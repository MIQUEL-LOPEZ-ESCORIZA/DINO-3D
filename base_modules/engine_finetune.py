import math
import time
import sys
import copy

import numpy as np
from sklearn.metrics import roc_auc_score

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
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
    
    for idx, batch_data in enumerate(loader):
        for optimizer in optimizers:
            optimizer.zero_grad()
            
        data, target = batch_data['image'].to(device), batch_data['pred_label'].to(device)
        batch_size = data.shape[0]
        
        roi = config.MODEL.ROI
        num_patches = config.VIT.NUM_PATCHES
        hidden_dim = config.VIT.HIDDEN_SIZE
        
        # mix-precision
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            if model_name == 'vit':
                # aggregate batch and patch dimension
                batch_data_agg = data.reshape(-1, 1, roi[0], roi[1], roi[2])
                out, _ = model(batch_data_agg)
                # restore orignal batch/patch dimension and average across patch dimension
                out = out.reshape(batch_size, num_patches, -1, hidden_dim)
            else:
                raise NotImplementedError("Unknown model: {}".format(model_name))
        
        out = out[:, :, :1, :].squeeze().reshape(batch_size, num_patches * hidden_dim)
        #out = out[:, :, :1, :].squeeze().reshape(batch_size, num_patches, hidden_dim)
        logits = classifier(out)
        
        if config.VIT.NUM_CLASSES == 1:
            target = target.float().reshape(batch_size, 1)
        loss = criterion(logits, target)
        
        # metrics
        if config.VIT.NUM_CLASSES != 1:
            train_metric_collection(torch.tensor(F.softmax(logits, dim=1).detach()), torch.tensor(target))
        else:
            train_metric_collection(torch.tensor(logits.detach()), torch.tensor(target))
            
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
):
    all_preds, all_targets = [], []
    model_name = config.MODEL.NAME
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data['image'].to(device), batch_data['pred_label'].to(device)
            batch_size = data.shape[0]
            
            roi = config.MODEL.ROI
            num_patches = config.VIT.NUM_PATCHES
            hidden_dim = config.VIT.HIDDEN_SIZE
            
            # mix-precision
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                if model_name == 'vit':
                    # aggregate batch and patch dimension
                    batch_data_agg = data.reshape(-1, 1, roi[0], roi[1], roi[2])
                    out, _ = model(batch_data_agg)
                    # restore orignal batch/patch dimension and average across patch dimension
                    out = out.reshape(batch_size, num_patches, -1, hidden_dim)
                    out = out[:, :, :1, :].squeeze().reshape(batch_size, num_patches * hidden_dim)
                    #out = out[:, :, :1, :].squeeze().reshape(batch_size, num_patches, hidden_dim)
                    logits = classifier(out)
                    
                    if config.VIT.NUM_CLASSES == 1:
                        target = target.float().reshape(batch_size, 1)
                    loss = criterion(logits, target)
                    all_preds.append(logits.detach().cpu())
                    all_targets.append(target.detach().cpu())
                else:
                    raise NotImplementedError("Unknown model: {}".format(model_name))
            
            torch.cuda.synchronize()
            # metrics
            if config.VIT.NUM_CLASSES != 1:
                val_metric_collection(torch.tensor(F.softmax(logits, dim=1)), torch.tensor(target))
            else:
                val_metric_collection(torch.tensor(logits), torch.tensor(target))
            
            loss_value = all_reduce_mean(loss)
            
            metric_logger.update(loss=loss_value)
            logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets).numpy()
    
    percent = config.PERCENT
    
    if config.VIT.NUM_CLASSES == 1:
        all_preds = all_preds.numpy()
        plot_regression(all_targets, all_preds, "Age", percent)
    elif config.VIT.NUM_CLASSES == 2:
        all_preds = F.softmax(all_preds.float(), dim=1)[:, 1].numpy()
        plot_pr_curve(all_targets, all_preds, percent)
    else:
        raise NotImplementedError("Unknown number of classes: {}".format(config.VIT.NUM_CLASSES))
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
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
    if not config.TRAIN.LOCK:
        best_model = copy.deepcopy(model)
        best_classifier = copy.deepcopy(classifier)
    else:
        best_model = model
        best_classifier = classifier
        
    use_amp = config.AMP_ENABLE
    
    val_loss_min = float("inf")
    val_losses = []
    
    if config.VIT.NUM_CLASSES != 1:
        train_metric_collection = MetricCollection([
            MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average=None),
            MulticlassAUROC(num_classes=config.VIT.NUM_CLASSES, average=None),
        ]).to(device)
        
        val_metric_collection = MetricCollection([
            MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average=None),
            MulticlassAUROC(num_classes=config.VIT.NUM_CLASSES, average=None),
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
        
        if config.VIT.NUM_CLASSES != 1:
            acc_out, auroc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy(), \
                metric_out["MulticlassAUROC"].detach().cpu().numpy()
            logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
        else:
            mae_out, mse_out = metric_out["MeanAbsoluteError"].detach().cpu().numpy(), \
                metric_out["MeanSquaredError"].detach().cpu().numpy()
            logger.info(f"MeanAbsoluteError: {mae_out}, MeanSquaredError:{mse_out}")
            
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
            
            if config.VIT.NUM_CLASSES != 1:
                acc_out, auroc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy(), \
                    metric_out["MulticlassAUROC"].detach().cpu().numpy()
                logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
            else:
                mae_out, mse_out = metric_out["MeanAbsoluteError"].detach().cpu().numpy(), \
                    metric_out["MeanSquaredError"].detach().cpu().numpy()
                logger.info(f"MeanAbsoluteError: {mae_out}, MeanSquaredError:{mse_out}")
                
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
                if not config.TRAIN.LOCK:
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
    
    use_amp = config.AMP_ENABLE
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if config.VIT.NUM_CLASSES != 1:
        test_metric_collection = MetricCollection([
            MulticlassAccuracy(num_classes=config.VIT.NUM_CLASSES, average=None),
            MulticlassAUROC(num_classes=config.VIT.NUM_CLASSES, average=None),
        ]).to(device)
    else:
        test_metric_collection = MetricCollection([
            MeanAbsoluteError(),
            MeanSquaredError(),
        ]).to(device)

    epoch, max_epoch = 0, 1

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
    )
    
    logger.info(
        f"Final test loss: {test_stats['loss']}, time {time.time() - epoch_time}s"
    )
    
    metric_out = test_metric_collection.compute()
    
    if config.VIT.NUM_CLASSES != 1:
        acc_out, auroc_out = metric_out["MulticlassAccuracy"].detach().cpu().numpy(), \
            metric_out["MulticlassAUROC"].detach().cpu().numpy()
        logger.info(f"MulticlassAccuracy: {acc_out}, MulticlassAUROC:{auroc_out}")
    else:
        mae_out, mse_out = metric_out["MeanAbsoluteError"].detach().cpu().numpy(), \
            metric_out["MeanSquaredError"].detach().cpu().numpy()
        logger.info(f"MeanAbsoluteError: {mae_out}, MeanSquaredError:{mse_out}")
        
    test_metric_collection.reset()

    if wandb_run != None and dist.get_rank() == 0:
        wandb_run.log({'Test Loss': test_stats['loss']})

    return test_stats['loss']