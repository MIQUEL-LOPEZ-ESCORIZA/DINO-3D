import torch
import math
import time
import sys

import torch.distributed as dist

from src.utils.misc import all_reduce_mean, save_checkpoint, MetricLogger
from src.losses.vae_loss import L1Loss, KLDivergence

def train_one_epoch(
    config,
    model,
    loader,
    optimizer,
    scheduler,
    epoch,
    max_epoch,
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
        data = batch_data['image'].to(device)
        
        # mix-precision
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            if model_name == 'mae':
                loss, _, _ = model(data)
            elif model_name == 'vae':
                criterion_rec = L1Loss()
                criterion_dis = KLDivergence()
                y, z_mean, z_log_sigma = model(data)
                loss_rec = criterion_rec(data, y)
                loss_KL = criterion_dis(z_mean, z_log_sigma)
                loss = loss_rec + loss_KL
            else:
                raise NotImplementedError("Unknown model: {}".format(model_name))
        
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
    logger=None,
    device=None,
    use_amp=False,
    scaler=None,
):
    model_name = config.MODEL.NAME
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data['image'].to(device)

            # mix-precision
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                if model_name == 'mae':
                    loss, _, _ = model(data)
                elif model_name == 'vae':
                    criterion_rec = L1Loss()
                    criterion_dis = KLDivergence()
                    y, z_mean, z_log_sigma = model(data)
                    loss_rec = criterion_rec(data, y)
                    loss_KL = criterion_dis(z_mean, z_log_sigma)
                    loss = loss_rec + loss_KL
                else:
                    raise NotImplementedError("Unknown model: {}".format(model_name))
            
            loss_value = all_reduce_mean(loss)
            
            if not math.isfinite(loss_value):
                #logger.info("Loss is {}, stopping training".format(loss_value))
                logger.info("Loss is {}, ignored".format(loss_value))
                #sys.exit(1)
            
            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            
            logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")

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
            epoch,
            max_epochs,
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

        if (epoch + 1) % val_every == 0 or epoch == 0:
            epoch_time = time.time()
            val_stats = val_one_epoch(
                config,
                model,
                val_loader,
                epoch,
                max_epochs,
                logger=logger,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )
            
            logger.info(
                f"Final validation {epoch+1}/{max_epochs} \
                    loss: {val_stats['loss']}, time {time.time() - epoch_time}s"
            )
            
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
    logger=None,
    device=None,
    wandb_run=None,
):
    epoch_time = time.time()
    
    use_amp = config.AMP_ENABLE
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epoch, max_epoch = 0, 1

    test_stats = val_one_epoch(
        config,
        model,
        test_loader,
        epoch,
        max_epoch,
        logger=logger,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    
    logger.info(
        f"Final test loss: {test_stats['loss']}, time {time.time() - epoch_time}s"
    )

    if wandb_run != None and dist.get_rank() == 0:
        wandb_run.log({'Test Loss': test_stats['loss']})
        
    if wandb_run != None and dist.get_rank() == 0:
        wandb_run.log({'Test Loss': test_stats['loss']})

    return test_stats['loss']