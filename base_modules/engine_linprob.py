import math
import time
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

import torch
import torch.nn.functional as F
import torch.distributed as dist

from src.utils.misc import all_reduce_mean, plot_regression, save_checkpoint, MetricLogger

def train_one_epoch(
    config,
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    epoch,
    max_epoch,
    train_embedding,
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
        optimizer.zero_grad()
        data, target = batch_data[0], batch_data[1].to(device)
        
        # gather embedding by file name
        embeddings = []
        for fname in data:
            embeddings.append(train_embedding[fname].squeeze())
        embeddings = torch.tensor(np.stack(embeddings, axis=0)).to(device)
        bs = embeddings.shape[0]
        embeddings = embeddings.reshape(bs, -1)
        
        # mix-precision
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            if model_name == 'vit':
                logits = model(embeddings).squeeze()
                if config.VIT.NUM_CLASSES == 1:
                    target = target.float()
                loss = criterion(logits, target)
            else:
                raise NotImplementedError("Unknown model: {}".format(model_name))
        
        # metrics
        if config.VIT.NUM_CLASSES != 1:
            train_metric_collection(F.softmax(logits, dim=1), target)
        else:
            train_metric_collection(logits, target)
            
        # loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # gradient clipping
        if config.TRAIN.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        torch.cuda.synchronize()
        loss_value = all_reduce_mean(loss)
        
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
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
    loader,
    epoch,
    max_epoch,
    val_embedding,
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
            data, target = batch_data[0], batch_data[1].to(device)
            
            # gather embedding by file name
            embeddings = []
            for fname in data:
                embeddings.append(val_embedding[fname].squeeze())
            embeddings = torch.tensor(np.stack(embeddings, axis=0)).to(device)
            bs = embeddings.shape[0]
            embeddings = embeddings.reshape(bs, -1)
            
            # mix-precision
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                if model_name == 'vit':
                    logits = model(embeddings).squeeze()
                    if config.VIT.NUM_CLASSES == 1:
                        target = target.float()
                    loss = criterion(logits, target)
                    all_preds.append(logits.detach().cpu())
                    all_targets.append(target.detach().cpu())
                else:
                    raise NotImplementedError("Unknown model: {}".format(model_name))
            
            torch.cuda.synchronize()
            # metrics
            if config.VIT.NUM_CLASSES != 1:
                val_metric_collection(F.softmax(logits, dim=1), target)
            else:
                val_metric_collection(logits, target)
            
            loss_value = all_reduce_mean(loss)
            
            metric_logger.update(loss=loss_value)
            logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    if config.VIT.NUM_CLASSES == 1:
        plot_regression(all_targets, all_preds, "Age")
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def trainer(
    config,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    train_embedding,
    val_embedding,
    start_epoch=0,
    max_epochs=100,
    val_every=10,
    logger=None,
    device=None,
    wandb_run=None,
):
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
            train_loader,
            optimizer,
            scheduler,
            criterion,
            epoch,
            max_epochs,
            train_embedding,
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

        if (epoch + 1) % val_every == 0 or epoch == 0:
            epoch_time = time.time()
            val_stats = val_one_epoch(
                config,
                model,
                val_loader,
                epoch,
                max_epochs,
                val_embedding,
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
                        optimizer,
                        scheduler,
                        best_loss=val_loss_min,
                        dir_add=config.MODEL.DIR,
                        filename=config.MODEL.SAVE_NAME,
                        logger=logger,
                    )
                
    logger.info(f"Training Finished !, Best Loss: {val_loss_min}")
    
    return val_loss_min


def tester(
    config,
    model,
    test_loader,
    test_embedding,
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
        test_loader,
        epoch,
        max_epoch,
        test_embedding,
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