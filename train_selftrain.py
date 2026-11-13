import os
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box
import lightning as L
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
import cv2
import argparse
from datasets import call_load_dataset
from utils.losses import DiceLoss, FocalLoss, ContraLoss
from utils.model import Model
from utils.eval_utils import AverageMeter, calc_iou, validate, get_prompts, calc_iou_matrix
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances



def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    teacher_model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    initial_validation_time: float,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    max_iou = 0.
    max_camo = 0.
    max_chame = 0.
    max_cod = 0.
    train_start = time.monotonic()
    total_batch_time = AverageMeter()
    validation_time = AverageMeter()
    validation_time.update(initial_validation_time)

    for epoch in range(1, cfg.num_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        num_iter = len(train_dataloader)

        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, gt_masks, img_paths= data
            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > 100:
                total_batch_time.update(time.time() - end)
                end = time.time()
                continue
            if num_insts > cfg.max_nums:
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            soft_image_embeds, soft_masks, soft_iou_predictions, soft_res_masks = teacher_model(images_weak, prompts)    # teacher
            pred_image_embeds, pred_masks, iou_predictions, pred_res_masks = model(images_strong, prompts)   # student

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, soft_mask, iou_prediction in zip(pred_masks, soft_masks, iou_predictions):
                soft_mask = (soft_mask > 0.).float()

                loss_focal += focal_loss(pred_mask, soft_mask, num_masks)
                loss_dice += dice_loss(pred_mask, soft_mask, num_masks)
                batch_iou = calc_iou(pred_mask, soft_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            fabric.backward(loss_total)

            # if iter + 1 % 5 == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            total_batch_time.update(batch_time.val)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)
            if iter%50==0:
                completed_steps = (epoch - 1) * num_iter + iter + 1
                remaining_steps = cfg.num_epochs * num_iter - completed_steps
                remaining_validations = cfg.num_epochs // cfg.eval_interval - (epoch - 1) // cfg.eval_interval
                eta_seconds = (remaining_steps * total_batch_time.avg
                               + remaining_validations * validation_time.avg)
                elapsed = timedelta(seconds=round(time.monotonic() - train_start))
                eta = timedelta(seconds=round(eta_seconds))
                fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                             f' | Elapsed [{elapsed}] | ETA [{eta}]'
                             f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                             f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                             f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                             f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                             f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                             f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            # loss_logger = {
            #     "Focal Loss": focal_losses.avg,
            #     "Dice Loss": dice_losses.avg,
            #     "IoU Loss": iou_losses.avg,
            #     "Total Loss": total_losses.avg
            # }
            # fabric.log_dict(loss_logger, num_iter * (epoch - 1) + iter)
            torch.cuda.empty_cache()

        if epoch % cfg.eval_interval == 0:
            validation_start = time.monotonic()
            iou, f1_score = validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
            validation_time.update(time.monotonic() - validation_start)
            if iou > max_iou:
                state = {"model": model, "optimizer": optimizer}
                fabric.save(os.path.join(cfg.out_dir, "save", "last-ckpt.pth"), state)
                max_iou = iou

def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.out_name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box) -> None:
    gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    num_devices = len(gpu_ids)
    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(cfg.seed + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    load_datasets = call_load_dataset(cfg)
    train_data, val_data = load_datasets(cfg, 1024)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.resume and cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(cfg.model.ckpt)
        model.load_state_dict(full_checkpoint["model"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])

    anchor_model = copy_model(model)
    print('-'*100)
    print('\033[92mDirect test on the original SAM.\033[0m') 
    print('-'*100)
    validation_start = time.monotonic()
    validate(fabric, cfg, anchor_model, val_data, name=cfg.name, epoch=0)
    initial_validation_time = time.monotonic() - validation_start
    train_sam(cfg, fabric, model, anchor_model, optimizer, scheduler, train_data, val_data, initial_validation_time)

    del model, anchor_model, train_data, val_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config file path')
    parser.add_argument('--prompt', help='the type of prompt')
    parser.add_argument('--num_points',type=int, help='the number of points')
    parser.add_argument('--out_dir', help='the dir to save logs and models')
    parser.add_argument('--load_type', help='the dir to save logs and models')      
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    args = parse_args()

    exec(f'from {args.cfg} import cfg')

    # transfer the args to a dict
    args_dict = vars(args)
    cfg.merge_update(args_dict)

    main(cfg)
    torch.cuda.empty_cache()
