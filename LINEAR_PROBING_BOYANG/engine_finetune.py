import math
import time
import sys
import copy
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassAUROC, BinaryAccuracy, BinaryAUROC, Dice
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

import torch
import torch.nn.functional as F
import torch.distributed as dist

from src.utils.misc import all_reduce_mean, plot_regression, \
    plot_pr_curve, save_checkpoint, MetricLogger

def train_one_epoch(
    config,
    model,
    classifier,
    loader,
    optimizers,
    schedulers,
    criterion,
    epoch,
    max_epoch,
    train_metric_collection,
    logger=None,
    device=None,
    use_amp=False,
    scaler=None,
    wandb_run=None,
):
    model_name = config.MODEL.NAME
    model.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    roi = config.MODEL.ROI
    num_patches = config.VIT.NUM_PATCHES
    hidden_dim = config.VIT.HIDDEN_SIZE
    for idx, batch_data in enumerate(loader):
        for optimizer in optimizers:
            optimizer.zero_grad()
        if model_name == 'vit':    
            data, target = batch_data['image'].to(device), batch_data['pred_label'].to(device)
        if model_name == 'unetr':    
            data, target = batch_data['image'].to(device), batch_data['label']
            target = target.reshape(-1, 1, roi[0], roi[1], roi[2]).int().to(device)
        
        batch_size = data.shape[0]
        
        # mix-precision
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            if model_name == 'vit':
                # aggregate batch and patch dimension
                batch_data_agg = data.reshape(-1, 1, roi[0], roi[1], roi[2])
                out = model(batch_data_agg)
                #print('data.shape', data.shape, 'batch_data_agg.shape', batch_data_agg.shape,  'out.shape', out.shape)
                #out, _ = model(batch_data_agg)
                # restore orignal batch/patch dimension and average across patch dimension
                if num_patches>1:
                    out = out.reshape(batch_size, num_patches, -1, hidden_dim)
                    #print('out.shape', out.shape)
                    out = out[:, :, :1, :].squeeze().reshape(batch_size, num_patches * hidden_dim)
                    #print('out.shape', out.shape)
                #out = out[:, :, :1, :].squeeze().reshape(batch_size, num_patches, hidden_dim)
                logits = classifier(out)
            elif model_name == 'unetr':
                # aggregate batch and patch dimension
                batch_data_agg = data.reshape(-1, 1, roi[0], roi[1], roi[2])
                out = model(batch_data_agg)
                #print('batch_data_agg', batch_data_agg.shape, 'out', out.shape)
                logits = classifier(batch_data_agg, out, [out]*4)
                #print('logits', logits.shape, 'target', target.shape)
                #print('segmentation labels', np.unique(target.detach().cpu().numpy()))
            else:
                raise NotImplementedError("Unknown model: {}".format(model_name))
        
        
        if config.VIT.NUM_CLASSES == 1:
            target = target.float().reshape(batch_size, 1)
        if config.VIT.NUM_CLASSES == 2:
            if config.VIT.SMOOTH_MCI_LABEL and model_name=='vit':
                target = torch.stack((1-target/2, target/2),1)
            else:
                target = torch.stack((1-target, target),1).float()
            loss = criterion(logits, target)
        elif config.VIT.NUM_CLASSES == 8:
            target_cognitive = target[:,0]
            if config.VIT.SMOOTH_MCI_LABEL and model_name=='vit':
                target_cognitive = torch.stack((1-target_cognitive/2, target_cognitive/2),1)
            else:
                target_cognitive = torch.stack((1-target_cognitive, target_cognitive),1).float()
            target_affective = target[:,1]
            target_hyperactivity = target[:,2]
            target_psychotic = target[:,3]
            loss = 3*criterion(logits[:,:2], target_cognitive) + criterion(logits[:,[2,3]], target_affective.float()) + criterion(logits[:,[4,5]], target_hyperactivity.float()) + criterion(logits[:,[6,7]], target_psychotic.float())
            
        #print('logits.dtype', logits.dtype, 'target.dtype', target.dtype)
        
        # metrics
        if model_name == 'vit':
            if config.VIT.NUM_CLASSES != 1:
                if config.VIT.NUM_CLASSES>2:
                    logits = logits[:,:2]
                probs = torch.tensor(F.softmax(logits, dim=1).detach())
                if not config.VIT.SMOOTH_MCI_LABEL:
                    probs_1D = probs[:,1]
                    target_1D = target[:,1]
                    train_metric_collection(probs_1D, torch.tensor(target_1D).int())
                else:
                    probs_CN_AD = probs[target[:,0]!=0.5][:,1]
                    target_CN_AD = target[target[:,0]!=0.5][:,1]
                    if target_CN_AD.shape[0]>1:# >=1 CN/AD sample in mini-batch
                        train_metric_collection(probs_CN_AD, torch.tensor(target_CN_AD).int())
            else:
                train_metric_collection(torch.tensor(logits.detach()), torch.tensor(target))
        elif model_name == 'unetr':
            logits = logits.detach().argmax(axis=1)
            train_metric_collection(logits, target.detach()[:,0,:,:,:])
        # loss
        scaler.scale(loss).backward()
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
        
        # gradient clipping
        if config.TRAIN.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.GRAD_CLIP)
            
        for optimizer in optimizers:
            scaler.step(optimizer)
            
        scaler.update()
        
        for scheduler in schedulers:
            scheduler.step()
            
        torch.cuda.synchronize()
        loss_value = all_reduce_mean(loss)
        
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        metric_logger.update(loss=loss_value)
        lr = optimizers[0].param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")
        if wandb_run != None and dist.get_rank() == 0:
            wandb_run.log({'Training Loss': (float)(loss_value), 'Training lr': lr})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(
    config,
    model,
    classifier,
    loader,
    epoch,
    max_epoch,
    val_metric_collection,
    criterion, 
    logger=None,
    device=None,
    use_amp=False,
    scaler=None,
    save_pred=False,
    save_embeddings=False,
):
    all_preds, all_targets  = [], []
    if save_embeddings:
        all_embeddings = []
    model_name = config.MODEL.NAME
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    roi = config.MODEL.ROI
    num_patches = config.VIT.NUM_PATCHES
    hidden_dim = config.VIT.HIDDEN_SIZE
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if model_name == 'vit':    
                if 'embedding' in config.MODEL.SAVE_NAME:
                    n_images = batch_data['image'].shape[0]
                    data, target = batch_data['image'].to(device), torch.zeros(n_images).to(device)
                else:
                    data, target = batch_data['image'].to(device), batch_data['pred_label'].to(device)
            if model_name == 'unetr':    
                data, target = batch_data['image'].to(device), batch_data['label']
                target = target.reshape(-1, 1, roi[0], roi[1], roi[2]).int().to(device)
            batch_size = data.shape[0]
            
            # mix-precision
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                if model_name == 'vit':
                    # aggregate batch and patch dimension
                    batch_data_agg = data.reshape(-1, 1, roi[0], roi[1], roi[2])
                    out = model(batch_data_agg)
                    #out, _ = model(batch_data_agg)
                    # restore orignal batch/patch dimension and average across patch dimension
                    if num_patches>1:
                        out = out.reshape(batch_size, num_patches, -1, hidden_dim)
                        out = out[:, :, :1, :].squeeze().reshape(batch_size, num_patches * hidden_dim)
                    #out = out[:, :, :1, :].squeeze().reshape(batch_size, num_patches, hidden_dim)
                    logits = classifier(out)
                    
                    if config.VIT.NUM_CLASSES == 1:
                        target = target.float().reshape(batch_size, 1)
                        loss = criterion(logits, target)
                    elif config.VIT.NUM_CLASSES == 2:
                        if config.VIT.SMOOTH_MCI_LABEL:
                            target = torch.stack((1-target/2, target/2),1)
                        else:
                            target = torch.stack((1-target, target),1).float()
                        loss = criterion(logits, target)
                    elif config.VIT.NUM_CLASSES == 8:
                        target_cognitive = target[:,0]
                        if config.VIT.SMOOTH_MCI_LABEL and model_name=='vit':
                            target_cognitive = torch.stack((1-target_cognitive/2, target_cognitive/2),1)
                        else:
                            target_cognitive = torch.stack((1-target_cognitive, target_cognitive),1).float()
                        target_affective = target[:,1]
                        target_hyperactivity = target[:,2]
                        target_psychotic = target[:,3]
                        loss = 3*criterion(logits[:,:2], target_cognitive) + criterion(logits[:,[2,3]], target_affective.float()) + criterion(logits[:,[4,5]], target_hyperactivity.float()) + criterion(logits[:,[6,7]], target_psychotic.float())
        
                    all_preds.append(logits.detach().cpu())
                    all_targets.append(target.detach().cpu())
                    if save_embeddings:
                        all_embeddings.append(out.detach().cpu())
                elif model_name == 'unetr':
                    batch_data_agg = data.reshape(-1, 1, roi[0], roi[1], roi[2])
                    out = model(batch_data_agg)
                    logits = classifier(batch_data_agg, out, [out]*4)
                    loss = criterion(logits, target)
                    #all_preds.append(logits.detach().cpu())
                    #all_targets.append(target.detach().cpu())
                    if save_pred:
                        images = logits.view(batch_size, num_patches, config.VIT.NUM_CLASSES, roi[0], roi[1], roi[2]).detach().cpu().numpy()
                        for i in range(batch_size):
                            filename = batch_data['image_meta_dict']['filename_or_obj'][i].replace('fsaverage.T1.mni152.mgz', config.MODEL.SAVE_NAME.replace('pt', 'npy'))
                            print(filename)
                            np.save(filename, images[i, ...])
                else:
                    raise NotImplementedError("Unknown model: {}".format(model_name))
            
            torch.cuda.synchronize()
            # metrics
            if model_name == 'vit':
                if config.VIT.NUM_CLASSES != 1:
                    if config.VIT.NUM_CLASSES != 2:
                        preds = logits[:, :2]
                    else:
                        preds = logits
                    probs = torch.tensor(F.softmax(preds, dim=1))
                    if not config.VIT.SMOOTH_MCI_LABEL:
                        #val_metric_collection(probs, torch.tensor(target))
                        probs_1D = probs[:,1]
                        target_1D = target[:,1]
                        val_metric_collection(probs_1D, torch.tensor(target_1D).int())
                    else:
                        probs_CN_AD = probs[target[:,0]!=0.5][:,1]
                        target_CN_AD = target[target[:,0]!=0.5][:,1]
                        if target_CN_AD.shape[0]>1:# >=1 CN/AD sample in mini-batch
                            val_metric_collection(probs_CN_AD, torch.tensor(target_CN_AD).int())
                else:
                    val_metric_collection(torch.tensor(logits), torch.tensor(target))
                    
            elif model_name == 'unetr':
                logits = logits.detach().argmax(axis=1)
                val_metric_collection(logits, target.detach()[:,0,:,:,:])
                
            
            loss_value = all_reduce_mean(loss)
            
            metric_logger.update(loss=loss_value)
            logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")

    
    percent = config.PERCENT

    if model_name=='vit':
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets).numpy()
        if config.VIT.NUM_CLASSES == 1:
            all_preds = all_preds.numpy()
            plot_regression(all_targets, all_preds, "Age", percent)
        elif config.VIT.NUM_CLASSES == 2:
            all_preds = F.softmax(all_preds.float(), dim=1)[:, 1].numpy()
            #plot_pr_curve(all_targets, all_preds, percent)
        elif config.VIT.NUM_CLASSES == 3:
            all_preds = F.softmax(all_preds.float(), dim=1)[:, 1].numpy()
        elif config.VIT.NUM_CLASSES == 8:
            all_preds_cognitive = F.softmax(all_preds[:, :1].float(), dim=1)[:, 1].numpy()
            all_preds_affective = F.softmax(all_preds[:, [2, 3]].float(), dim=1)[:, 1].numpy()
            all_preds_hyperactivity = F.softmax(all_preds[:, [4, 5]].float(), dim=1)[:, 1].numpy()
            all_preds_psychotic = F.softmax(all_preds[:, [6, 7]].float(), dim=1)[:, 1].numpy()
            all_preds = np.vstack([all_preds_cognitive, all_preds_affective, all_preds_hyperactivity, all_preds_psychotic]).T
        else:
            raise NotImplementedError("Unknown number of classes: {}".format(config.VIT.NUM_CLASSES))
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    if save_pred and model_name=='vit':
        all_preds = torch.Tensor(all_preds).to(device)
        all_targets = torch.Tensor(all_targets).to(device)
        all_preds_gathered = [torch.zeros_like(all_preds).to(device) for _ in range(dist.get_world_size())]
        all_targets_gathered = [torch.zeros_like(all_targets).to(device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_preds_gathered, all_preds)
        dist.all_gather(all_targets_gathered, all_targets)
        all_preds = torch.cat(all_preds_gathered).detach().cpu().numpy()
        all_targets = torch.cat(all_targets_gathered).detach().cpu().numpy()
        os.makedirs('/gpfs/scratch/by2026/BrainATLAS/mae/NPS_cog_pred/', exists_ok=True)
        np.save(f"/gpfs/scratch/by2026/BrainATLAS/mae/NPS_cog_pred/{config.MODEL.SAVE_NAME.replace('.pt', '_test_pred.npy')}", all_preds)
        np.save(f"/gpfs/scratch/by2026/BrainATLAS/mae/NPS_cog_pred/{config.MODEL.SAVE_NAME.replace('.pt', '_test_target.npy')}", all_targets)
    if save_embeddings and model_name=='vit':
        all_embeddings = torch.cat(all_embeddings)
        all_embeddings = torch.Tensor(all_embeddings).to(device)
        all_embeddings_gathered = [torch.zeros_like(all_embeddings).to(device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_embeddings_gathered, all_embeddings)
        all_embeddings = torch.cat(all_embeddings_gathered).detach().cpu().numpy()
        model_name = config.MODEL.PRETRAINED.split('/')[-1].replace('.pt','')
        pred_sample_name = config.DATA.TEST_CSV_PATH.split('/')[-1].replace('.csv','')
        os.makedirs('/gpfs/scratch/by2026/BrainATLAS/mae/embeddings/', exists_ok=True)
        with open(f"/gpfs/scratch/by2026/BrainATLAS/mae/embeddings/{model_name}_{pred_sample_name}.pkl", 'wb') as handle:
            pickle.dump(all_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def trainer(
    config,
    model,
    classifier,
    train_loader,
    val_loader,
    optimizers,
    schedulers,
    criterion,
    start_epoch=0,
    max_epochs=100,
    val_every=10,
    logger=None,
    device=None,
    wandb_run=None,
):
    model_name = config.MODEL.NAME
    if not config.TRAIN.LOCK or not config.TRAIN.LOCK_LAST_ATTENTION_MODULE:
        best_model = copy.deepcopy(model)
        best_classifier = copy.deepcopy(classifier)
    else:
        best_model = model
        best_classifier = classifier
        
    use_amp = config.AMP_ENABLE
    
    val_loss_min = float("inf")
    val_losses = []

    if model_name == 'vit':
        if config.VIT.NUM_CLASSES >2:
            train_metric_collection = MetricCollection([
                MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average=None),
                MulticlassAUROC(num_classes=config.VIT.NUM_CLASSES, average=None),
            ]).to(device)
            
            val_metric_collection = MetricCollection([
                MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average=None),
                MulticlassAUROC(num_classes=config.VIT.NUM_CLASSES, average=None),
            ]).to(device)
        elif config.VIT.NUM_CLASSES == 2:
            train_metric_collection = MetricCollection([
                BinaryAccuracy(),
                BinaryAUROC(),
            ]).to(device)
            
            val_metric_collection = MetricCollection([
                BinaryAccuracy(),
                BinaryAUROC(),
            ]).to(device)
        else:
            train_metric_collection = MetricCollection([
                MeanAbsoluteError(),
                MeanSquaredError(),
            ]).to(device)
            
            val_metric_collection = MetricCollection([
                MeanAbsoluteError(),
                MeanSquaredError(),
            ]).to(device)
    elif model_name == 'unetr':
        train_metric_collection = MetricCollection([
                Dice(num_classes=config.VIT.NUM_CLASSES, average='macro'),
                MulticlassPrecision(num_classes=config.VIT.NUM_CLASSES, average='macro'),
                MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average='macro')
            ]).to(device)
       
        val_metric_collection = MetricCollection([
                Dice(num_classes=config.VIT.NUM_CLASSES, average='macro'),
                MulticlassPrecision(num_classes=config.VIT.NUM_CLASSES, average='macro'),
                MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average='macro')
            ]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(start_epoch, max_epochs):
        logger.info(f"Epoch: {epoch+1}")
        epoch_time = time.time()
        
        train_stats = train_one_epoch(
            config,
            model,
            classifier,
            train_loader,
            optimizers,
            schedulers,
            criterion,
            epoch,
            max_epochs,
            train_metric_collection,
            logger=logger,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            wandb_run=wandb_run,
        )
        logger.info(
            f"Final training  {epoch+1}/{max_epochs}, loss: {train_stats['loss']}, \
                time {time.time() - epoch_time}s"
        )
        
        metric_out = train_metric_collection.compute()

        if model_name == 'vit':
            if config.VIT.NUM_CLASSES>2:
                acc_out, auroc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy(), \
                    metric_out["MulticlassAUROC"].detach().cpu().numpy()
                logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
                if wandb_run != None and dist.get_rank() == 0:
                    wandb_run.log({'Training MacroMulticlassAccuracy': (float)(np.mean(acc_out)), 'Training MacroMulticlassAUROC': np.mean(auroc_out)})
            elif config.VIT.NUM_CLASSES==2:
                acc_out, auroc_out = metric_out["BinaryAccuracy"].detach().cpu().numpy(), \
                    metric_out["BinaryAUROC"].detach().cpu().numpy()
                logger.info(f"BinaryAccuracy: {acc_out}, BinaryAUROC:{auroc_out}")
                if wandb_run != None and dist.get_rank() == 0:
                    wandb_run.log({'Training BinaryAccuracy': (float)(np.mean(acc_out)), 'Training BinaryAUROC': np.mean(auroc_out)})
            else:
                mae_out, mse_out = metric_out["MeanAbsoluteError"].detach().cpu().numpy(), \
                    metric_out["MeanSquaredError"].detach().cpu().numpy()
                logger.info(f"MeanAbsoluteError: {mae_out}, MeanSquaredError:{mse_out}")
                if wandb_run != None and dist.get_rank() == 0:
                    wandb_run.log({'Training MeanAbsoluteError': (float)(mae_out), 'Training MeanSquaredError': mse_out})
        elif model_name == 'unetr':    
            dice_out, precision_out, acc_out = metric_out["Dice"].detach().cpu().numpy(), \
                    metric_out["MulticlassPrecision"].detach().cpu().numpy(), \
                    metric_out["MulticlassAccuracy"].detach().cpu().numpy()
            logger.info(f"Dice: {dice_out}, MulticlassPrecision: {precision_out}, MulticlassAccuracy:{acc_out}")
            if wandb_run != None and dist.get_rank() == 0:
                wandb_run.log({'Training Dice': (float)(dice_out), 'Training MulticlassPrecision': precision_out, 'Training MulticlassAccuracy': acc_out})
        train_metric_collection.reset()
        
        if (epoch + 1) % val_every == 0 and epoch != 0:
            epoch_time = time.time()
            val_stats = val_one_epoch(
                config,
                model,
                classifier,
                val_loader,
                epoch,
                max_epochs,
                val_metric_collection,
                criterion,
                logger=logger,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )
            
            logger.info(
                f"Final validation {epoch+1}/{max_epochs} \
                    loss: {val_stats['loss']}, time {time.time() - epoch_time}s"
            )
            metric_out = val_metric_collection.compute()

            if model_name == 'vit':
                if config.VIT.NUM_CLASSES>2:
                    acc_out, auroc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy(), \
                        metric_out["MulticlassAUROC"].detach().cpu().numpy()
                    logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
                    if wandb_run != None and dist.get_rank() == 0:
                        wandb_run.log({'Validation MacroMulticlassAccuracy': (float)(np.mean(acc_out)), 'Validation MacroMulticlassAUROC': np.mean(auroc_out)})
                elif config.VIT.NUM_CLASSES==2:
                    acc_out, auroc_out = metric_out["BinaryAccuracy"].detach().cpu().numpy(), \
                        metric_out["BinaryAUROC"].detach().cpu().numpy()
                    logger.info(f"BinaryAccuracy: {acc_out}, BinaryAUROC:{auroc_out}")
                    if wandb_run != None and dist.get_rank() == 0:
                        wandb_run.log({'Validation BinaryAccuracy': (float)(np.mean(acc_out)), 'Validation BinaryAUROC': np.mean(auroc_out)})
                else:
                    mae_out, mse_out = metric_out["MeanAbsoluteError"].detach().cpu().numpy(), \
                        metric_out["MeanSquaredError"].detach().cpu().numpy()
                    logger.info(f"MeanAbsoluteError: {mae_out}, MeanSquaredError:{mse_out}")
                    if wandb_run != None and dist.get_rank() == 0:
                        wandb_run.log({'Validation MeanAbsoluteError': (float)(mae_out), 'Validation MeanSquaredError': mse_out})
            elif model_name == 'unetr':
                dice_out, precision_out, acc_out = metric_out["Dice"].detach().cpu().numpy(), \
                        metric_out["MulticlassPrecision"].detach().cpu().numpy(), \
                        metric_out["MulticlassAccuracy"].detach().cpu().numpy()
                logger.info(f"Dice: {dice_out}, MulticlassPrecision:{precision_out}, MulticlassAccuracy:{acc_out}")
                if wandb_run != None and dist.get_rank() == 0:
                    wandb_run.log({'Validation Dice': (float)(dice_out), 'Validation MulticlassPrecision': precision_out , 'Validation MulticlassAccuracy': acc_out})
            
            val_metric_collection.reset()
            
            if wandb_run != None and dist.get_rank() == 0:
                wandb_run.log({'Validation Loss': (float)(val_stats['loss'])})
            
            val_losses.append(val_stats['loss'])
            
            if val_stats['loss'] < val_loss_min:
                logger.info(f"new best ({val_loss_min} --> {val_stats['loss']}). ")
                val_loss_min = val_stats['loss']
                # only save model in main process since synchornization
                if dist.get_rank() == 0:
                    save_checkpoint(
                        model,
                        epoch,
                        optimizers[0],
                        schedulers[0],
                        best_loss=val_loss_min,
                        dir_add=config.MODEL.DIR,
                        filename=config.MODEL.SAVE_NAME,
                        logger=logger,
                    )
                #if not config.TRAIN.LOCK:
                best_model = copy.deepcopy(model)
                best_classifier = copy.deepcopy(classifier)
                
    logger.info(f"Training Finished !, Best Loss: {val_loss_min}")
    
    return val_loss_min, best_model, best_classifier


def tester(
    config,
    model,
    classifier,
    test_loader,
    criterion,
    logger=None,
    device=None,
    wandb_run=None,
):
    epoch_time = time.time()
    model_name = config.MODEL.NAME
    
    use_amp = config.AMP_ENABLE
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if model_name=='vit':
        if config.VIT.NUM_CLASSES>2:
            test_metric_collection = MetricCollection([
                MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average=None),
                MulticlassAUROC(num_classes=config.VIT.NUM_CLASSES, average=None),
            ]).to(device)
        elif config.VIT.NUM_CLASSES==2:
            test_metric_collection = MetricCollection([
                BinaryAccuracy(),
                BinaryAUROC(),
            ]).to(device)
        else:
            test_metric_collection = MetricCollection([
                MeanAbsoluteError(),
                MeanSquaredError(),
            ]).to(device)
    elif model_name=='unetr':
        test_metric_collection = MetricCollection([
                Dice(num_classes=config.VIT.NUM_CLASSES, average='macro'),
                MulticlassPrecision(num_classes=config.VIT.NUM_CLASSES, average='macro'),
                MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average='macro')
            ]).to(device)
        
    epoch, max_epoch = 0, 1
    if 'embedding' in config.MODEL.SAVE_NAME:
        save_embeddings = True
    else:
        save_embeddings = False
    test_stats = val_one_epoch(
        config,
        model,
        classifier,
        test_loader,
        epoch,
        max_epoch,
        test_metric_collection,
        criterion,
        logger=logger,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
        save_pred=True,# if model_name=='vit' else False,
        save_embeddings=save_embeddings,
    )
    
    logger.info(
        f"Final test loss: {test_stats['loss']}, time {time.time() - epoch_time}s"
    )
    
    metric_out = test_metric_collection.compute()

    if model_name == 'vit':
        if config.VIT.NUM_CLASSES>2:
            acc_out, auroc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy(), \
                metric_out["MulticlassAUROC"].detach().cpu().numpy()
            logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
        elif config.VIT.NUM_CLASSES == 2:
            acc_out, auroc_out = metric_out["BinaryAccuracy"].detach().cpu().numpy(), \
                metric_out["BinaryAUROC"].detach().cpu().numpy()
            logger.info(f"BinaryAccuracy: {acc_out}, BinaryAUROC:{auroc_out}")
        else:
            mae_out, mse_out = metric_out["MeanAbsoluteError"].detach().cpu().numpy(), \
                metric_out["MeanSquaredError"].detach().cpu().numpy()
            logger.info(f"MeanAbsoluteError: {mae_out}, MeanSquaredError:{mse_out}")
    elif model_name == 'unetr':
        dice_out, precision_out, acc_out = metric_out["Dice"].detach().cpu().numpy(), \
                        metric_out["MulticlassPrecision"].detach().cpu().numpy(), \
                        metric_out["MulticlassAccuracy"].detach().cpu().numpy()
        logger.info(f"Dice: {dice_out}, MulticlassPrecision:{precision_out}, MulticlassAccuracy:{acc_out}")
    test_metric_collection.reset()

    if wandb_run != None and dist.get_rank() == 0:
        wandb_run.log({'Test Loss': test_stats['loss']})

    return test_stats['loss']