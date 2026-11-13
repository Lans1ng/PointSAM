import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.ndimage import map_coordinates

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L

import segmentation_models_pytorch as smp

from box import Box
from utils.model import Model
from utils.sample_utils import get_point_prompts
from utils.tools import write_csv


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

def calc_iou_instance(pred_masks: torch.Tensor, gt_masks: torch.Tensor):
    iou_list = []
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        pred_mask = (pred_mask >= 0.5).float()
        # print(pred_mask.shape)
        intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(0, 1))
    
        union = torch.sum(pred_mask, dim=(0, 1)) + torch.sum(gt_mask, dim=(0, 1)) - intersection
        epsilon = 1e-7
        iou = intersection / (union + epsilon)
        # print(iou)
        # batch_iou = batch_iou.unsqueeze(1)
        iou_list.append(iou)
    return iou_list


#intersection
# mask1:            mask2:           intersection:
# [[1, 1, 0, 0],   [[1, 0, 0, 0],   [[1, 0, 0, 0],
#  [1, 0, 0, 0],    [1, 1, 0, 0],    [1, 0, 0, 0],
#  [1, 1, 0, 0],    [0, 1, 0, 0],    [0, 1, 0, 0],
#  [0, 0, 0, 0]]    [0, 0, 0, 0]]    [0, 0, 0, 0]]

#union
# mask1:            mask2:           union:
# [[1, 1, 0, 0],   [[1, 0, 0, 0],   [[1, 1, 0, 0],
#  [1, 0, 0, 0],    [1, 1, 0, 0],    [1, 1, 0, 0],
#  [1, 1, 0, 0],    [0, 1, 0, 0],    [1, 1, 0, 0],
#  [0, 0, 0, 0]]    [0, 0, 0, 0]]    [0, 0, 0, 0]]


def calculate_iou(mask1, mask2):

    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    return iou

def calc_iou_matrix(mask_list1, mask_list2):

    iou_matrix = torch.zeros((len(mask_list1), len(mask_list2)))
    for i, mask1 in enumerate(mask_list1):
        for j, mask2 in enumerate(mask_list2):
            iou_matrix[i, j] = calculate_iou(mask1, mask2)
    return iou_matrix

def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts

def validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    recall = AverageMeter()
    precision = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(val_dataloader, desc='Validation', ncols=100,
                                         disable=not sys.stderr.isatty())):
            images, bboxes, gt_masks, img_paths = data
            num_images = images.size(0)
            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)

            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    (pred_mask >= 0.5).int(),
                    gt_mask.int(),
                    mode='binary',
                )
                batch_recall = smp.metrics.recall(*batch_stats, reduction="micro-imagewise")
                batch_precision = smp.metrics.precision(*batch_stats, reduction="micro-imagewise")
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
                recall.update(batch_recall, num_images)
                precision.update(batch_precision, num_images)

            torch.cuda.empty_cache()

    fabric.print(
        f'Val: [{epoch}] - [{iter+1}/{len(val_dataloader)}]: IoU: [{ious.avg:.4f}] -- Recall: [{recall.avg:.4f}] -- Precision [{precision.avg:.4f}] -- F1: [{f1_scores.avg:.4f}]'
    )
    csv_dict = {"Prompt": cfg.prompt, "IoU": f"{ious.avg:.4f}","Recall": f"{recall.avg:.4f}", "Precision": f"{precision.avg:.4f}", "F1": f"{f1_scores.avg:.4f}", "epoch": epoch}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_dict, csv_head=cfg.csv_keys)
    return ious.avg, f1_scores.avg




        
