import random
import sys
from collections import deque
from tqdm import tqdm
from box import Box
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F

from .finch import FINCH
from .sample_utils import get_point_prompts

class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])

def concatenate_images_with_padding(images, padding=10, color=(255, 255, 255)):
    heights = [image.shape[0] for image in images]
    widths = [image.shape[1] for image in images]

    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights)
    
    if len(images[0].shape) == 3:
        new_image = np.full((max_height, total_width, 3), color, dtype=np.uint8)
    else:
        new_image = np.full((max_height, total_width), color[0], dtype=np.uint8)

    x_offset = 0
    for image in images:
        new_image[0:image.shape[0], x_offset:x_offset + image.shape[1]] = image
        x_offset += image.shape[1] + padding
    
    return new_image

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

def cal_mask_ious(
    cfg,
    model,
    images_weak,
    prompts,
    gt_masks,
):
    with torch.no_grad():
         _, soft_masks, _, _ = model(images_weak, prompts)   

    for i, (soft_mask, gt_mask) in enumerate(zip(soft_masks, gt_masks)):  
        soft_mask = (soft_mask > 0).float()
        mask_ious = calc_iou_matrix(soft_mask, soft_mask)
        indices = torch.arange(mask_ious.size(0))
        mask_ious[indices, indices] = 0.0
    return mask_ious, soft_mask


def neg_prompt_calibration(
    cfg,
    mask_ious,
    prompts,
):
    '''
    mask_ious:[mask_nums,mask_nums]
    '''
    point_list = []
    point_labels_list = []
    num_points = cfg.num_points
    for m in range(len(mask_ious)):
            
        pos_point_coords = prompts[0][0][m][:num_points].unsqueeze(0) 
        neg_point = prompts[0][0][m][num_points:].unsqueeze(0)  
        neg_points_list = []
        neg_points_list.extend(neg_point[0])

        indices = torch.nonzero(mask_ious[m] > float(cfg.iou_thr)).squeeze(1)

        if indices.numel() != 0:
            # neg_points_list = []
            for indice in indices:
                neg_points_list.extend(prompts[0][0][indice][:num_points])
            neg_points = random.sample(neg_points_list, num_points)
        else:
            neg_points =neg_points_list
            
        neg_point_coords = torch.tensor([p.tolist() for p in neg_points], device=neg_point.device).unsqueeze(0)

        point_coords = torch.cat((pos_point_coords, neg_point_coords), dim=1) 

        point_list.append(point_coords)
        pos_point_labels = torch.ones(pos_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        neg_point_labels = torch.zeros(neg_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        point_labels = torch.cat((pos_point_labels, neg_point_labels), dim=1)  
        point_labels_list.append(point_labels)

    point_ = torch.cat(point_list).squeeze(1)
    point_labels_ = torch.cat(point_labels_list)
    new_prompts = [(point_, point_labels_)]
    return new_prompts

def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts

def generate_predict_feats(cfg, embed, pseudo_label, gts):
    coords, lbls = gts
    selected_coords = []
    
    num_insts = len(pseudo_label)
    num_points = cfg.num_points
    for coord_grp, lbl_grp in zip(coords, lbls):
        for coord, lbl in zip(coord_grp, lbl_grp):  
            if lbl.item() == 1:  
                selected_coords.append(coord.tolist())

    # Downsample coordinates (SAM's stride is 16)
    coords = [[int(c // 16) for c in pair] for pair in selected_coords]

    embed = embed.permute(1, 2, 0)  # [H, W, C]

    pos_pts = [] 

    for index in range(0, num_insts * num_points, num_points):
        index += random.randint(0, num_points - 1)
        x, y = coords[index]
        pos_pt = embed[y, x]
        pos_pts.append(pos_pt)

    predict_feats = torch.stack(pos_pts, dim=0)

    return predict_feats


def offline_prototypes_generation(cfg, model, loader):
    model.eval()
    pts = []
    max_iters = 128 
    num_points = cfg.num_points

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Generating target prototypes', ncols=100,
                                       disable=not sys.stderr.isatty())):
            if i >= max_iters: 
                break
            imgs, boxes, masks, _ = batch
            prompts = get_prompts(cfg, boxes, masks)

            embeds, masks, _, _ = model(imgs, prompts) 
            del _

            if isinstance(embeds, dict):
                embeds = embeds['vision_features'] 

            for embed, prompt, mask in zip(embeds, prompts, masks):
                num_insts = len(mask)
                embed = embed.permute(1, 2, 0)  # [H, W, C]
                coords = []

                points, labels = prompt
                for point_grp, label_grp in zip(points, labels):
                    for point, label in zip(point_grp, label_grp): 
                        if label.item() == 1:  
                            coords.append(point.tolist())

                # 16 is the stride of SAM
                coords = [[int(pt / 16) for pt in pair] for pair in coords]
                for index in range(0, num_insts*num_points, num_points):
                    x, y = coords[index]
                    pt = embed[y,x]
                    pts.append(pt)

    fin = FINCH(verbose=True)
    pts = torch.stack(pts).cpu().numpy()
    res = fin.fit(pts)

    last_key = list(res.partitions.keys())[-1]
    pt_stats = {'target_pts': res.partitions[last_key]['cluster_centers']}
    return pt_stats


