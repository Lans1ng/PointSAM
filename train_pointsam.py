import os
import time
import argparse
import random
from datetime import timedelta
from abc import ABC

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.fabric import _FabricOptimizer

from box import Box
from datasets import call_load_dataset
from utils.model import Model
from utils.losses import DiceLoss, FocalLoss, Matching_Loss
from utils.eval_utils import AverageMeter, validate, get_prompts, calc_iou
from utils.tools import copy_model, create_csv, reduce_instances
from utils.utils import *
from utils.finch import FINCH, select_partition

vis = False

def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    target_pts,
    initial_validation_time: float,
):

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    max_iou = 0.
    mem_bank = Store(1, cfg.mem_bank_max_len) 
    match_interval = cfg.match_interval
    train_start = time.monotonic()
    total_batch_time = AverageMeter()
    validation_time = AverageMeter()
    validation_time.update(initial_validation_time)
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        match_losses = AverageMeter()
        end = time.time()
        num_iter = len(train_dataloader)

        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, gt_masks, img_paths= data
            del data  
            
            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)
            prompts = get_prompts(cfg, bboxes, gt_masks)
            match_ready = (iter + 1) % match_interval == 0 and len(mem_bank) + sum(len(m) for m in gt_masks) >= cfg.mem_bank_max_len

            #1. caculate pairwise IoUs of masks
            mask_ious, init_masks = cal_mask_ious(cfg, model, images_weak, prompts, gt_masks)
       
            #2. get new prompts through neg_prompt_calibration
            new_prompts = neg_prompt_calibration(cfg, mask_ious, prompts)

            #3. start training using new prompt
            with torch.set_grad_enabled(match_ready):
                soft_image_embeds, soft_masks, _, _ = model(images_weak, new_prompts)    # teacher
            
            if isinstance(soft_image_embeds, dict):
                soft_image_embeds = soft_image_embeds['vision_features']  
                
            _, pred_masks, iou_predictions, _= model(images_strong, prompts)   # student

            del _

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_match = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            
            for i, (embed, pred_mask, soft_mask,  gt_mask, prompt, iou_prediction) in enumerate(zip(soft_image_embeds, pred_masks, soft_masks, gt_masks, prompts, iou_predictions)):
                    
                soft_mask = (soft_mask > 0.).float()
                live_feats = generate_predict_feats(cfg, embed, soft_mask, prompt)
                target_pts_ = target_pts['target_pts']  
                for pred_feat in live_feats.detach().cpu().tolist():
                    mem_bank.add([[pred_feat]], [0])
                if len(mem_bank.retrieve(0)) >= cfg.mem_bank_max_len and (iter + 1) % match_interval == 0:
                    bank_feats = np.array(mem_bank.retrieve(0))

                    #FINCH
                    fin = FINCH(verbose=False)
                    results = fin.fit(bank_feats)
                    partition = select_partition(
                        results.partitions, len(target_pts_), cfg.get('match_partition', 'last'))
                    pred_count = partition['n_clusters']
                    target_count = len(target_pts_)
                    partition_counts = [p['n_clusters'] for p in results.partitions.values()]
                    fabric.print(
                        f'Match clusters: target={target_count} predicted={pred_count} '
                        f'levels={partition_counts} '
                        f'coverage={min(pred_count, target_count) / max(pred_count, target_count):.2f}')
                    cached_feats = torch.as_tensor(bank_feats[:-len(live_feats)], device=fabric.device, dtype=live_feats.dtype)
                    clustered_feats = torch.cat((cached_feats, live_feats), dim=0)
                    labels = torch.as_tensor(partition['labels'], device=fabric.device)
                    pred_pts = torch.stack([
                        clustered_feats[labels == cluster].mean(dim=0)
                        for cluster in range(partition['n_clusters'])
                    ])
                    loss_match += Matching_Loss(pred_pts, target_pts_, device = fabric.device)

                del embed
                
                if vis:
                    img_name = os.path.basename(img_paths[i]).split('.')[0]
                    
                    image_weak = images_weak[0].permute(1,2,0).cpu().numpy()* 255
                    image_weak = cv2.cvtColor(image_weak, cv2.COLOR_BGR2RGB)

                    if vis:
                        for j in range(len(soft_mask)):
                            mask_iou = torch.max(mask_ious[j])
                            image_weak_ = image_weak.copy()
                            mask_area = torch.sum(gt_mask[j])
                            
                            gt_mask_np = gt_mask[j].cpu().numpy() * 255 
                            gt_mask_img = cv2.cvtColor(gt_mask_np, cv2.COLOR_GRAY2RGB)
                            
                            init_prompt_po = prompts[0][0][j][:cfg.num_points]
                            init_prompt_ne = prompts[0][0][j][cfg.num_points:]
                            
                            for po in init_prompt_po:
                                cv2.circle(image_weak_, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
                                
                            init_mask_img = init_masks[j].cpu().detach().numpy() * 255
                            init_mask_img = cv2.cvtColor(init_mask_img, cv2.COLOR_GRAY2RGB)
                            for po,ne in zip(init_prompt_po,init_prompt_ne):
                                cv2.circle(init_mask_img, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
                                cv2.circle(init_mask_img, (int(ne[0]), int(ne[1])), 12, (0, 255, 0), -1)
                                
                            prompt_po = new_prompts[0][0][j][:cfg.num_points]
                            prompt_ne = new_prompts[0][0][j][cfg.num_points:]
                            soft_mask_img = soft_mask[j].cpu().detach().numpy() * 255  

                            soft_mask_img = cv2.cvtColor(soft_mask_img, cv2.COLOR_GRAY2RGB)

                            for po,ne in zip(prompt_po,prompt_ne):
                                cv2.circle(soft_mask_img, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
                                cv2.circle(soft_mask_img, (int(ne[0]), int(ne[1])), 12, (0, 255, 0), -1)

                            output_dir = "./save_mask_{}/{}/{}/".format(str(float(cfg.num_points)),cfg.dataset,str(epoch))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            merged_image = concatenate_images_with_padding([image_weak_, gt_mask_img, init_mask_img, soft_mask_img])
                            img_name_ = '{}_{}_iou{}.jpg'.format(img_name,str(j),str(mask_iou))
                            if mask_iou>float(cfg.iou_thr) and mask_area>3000:
                                cv2.imwrite(os.path.join(output_dir,img_name_), merged_image) 
                del init_masks, mask_ious

                loss_focal += focal_loss(pred_mask, soft_mask, num_masks)
                loss_dice += dice_loss(pred_mask, soft_mask, num_masks)
                batch_iou = calc_iou(pred_mask, soft_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks
                
            del soft_image_embeds, pred_masks, iou_predictions, gt_masks 
                
            loss_total = 20. * loss_focal + loss_dice + loss_iou + 0.1*loss_match            

            fabric.backward(loss_total)

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
            match_losses.update(loss_match.item(), batch_size)

            if (iter+1) %match_interval==0:
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
                             f' | Match Loss [{match_losses.val:.4f} ({match_losses.avg:.4f})]'
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
            iou, _= validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
            validation_time.update(time.monotonic() - validation_start)
            if iou > max_iou:
                state = {"model": model, "optimizer": optimizer}
                fabric.save(os.path.join(cfg.out_dir, "save", "best-ckpt.pth"), state)
                max_iou = iou
            del iou 
            
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
    train_data, val_data, pt_data = load_datasets(cfg, img_size=1024, return_pt = True)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    pt_data = fabric._setup_dataloader(pt_data)
    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.resume and cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(cfg.model.ckpt)
        model.load_state_dict(full_checkpoint["model"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])
    print('-'*100)
    print('\033[92mDirect test on the original SAM.\033[0m') 
    validation_start = time.monotonic()
    _, _, = validate(fabric, cfg, model, val_data, name=cfg.name, epoch=0)
    initial_validation_time = time.monotonic() - validation_start
    print('-'*100)
    del _     
    
    target_pts = offline_prototypes_generation(cfg, model, pt_data)
    
    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data, target_pts, initial_validation_time)

    del model, train_data, val_data


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
