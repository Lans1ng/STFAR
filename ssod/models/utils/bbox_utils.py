import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmdet.core.mask.structures import BitmapMasks
from torch.nn import functional as F
from mmcv.runner.fp16_utils import force_fp32
import ipdb

def resize_image(inputs, resize_ratio=0.5):
    down_inputs = F.interpolate(inputs, 
                                scale_factor=resize_ratio, 
                                mode='nearest')
    
    return down_inputs

# def evaluate_pseudo_label(det_bboxes, det_labels, gt_bboxes, 
#                             gt_labels, thres=0.5):
#     """ Perform evaluation on pseudo boxes. 
#     """
#     area1 = (det_bboxes[:, 2:4] - det_bboxes[:, 0:2]).prod(dim=1) 
#     area2 = (gt_bboxes[:, 2:4] - gt_bboxes[:, 0:2]).prod(dim=1) 
#     lt = torch.max(det_bboxes[:, None, :2], gt_bboxes[None, :, :2])
#     rb = torch.min(det_bboxes[:, None, 2:4], gt_bboxes[None, :, 2:4])
#     wh = torch.clamp(rb - lt, min=0)
    
#     overlaps = wh[..., 0] * wh[..., 1] 
#     ious = overlaps / (area1[:, None] + area2[None, :] - overlaps + 1e-8)
    
#     max_iou, argmax_iou = ious.max(dim=1)

#     flags = (max_iou > thres) & (det_labels == gt_labels[argmax_iou])        
#     return flags

# def get_pseudo_label_quality(det_bboxes, det_labels, gt_bboxes, gt_labels):
#     """ precision and recall of pseudo labels.  
#     """
#     TPs = []
#     for det_bbox, det_label, gt_bbox, gt_label in \
#                 zip(det_bboxes, det_labels, gt_bboxes, gt_labels):
#         if det_bbox.shape[0] == 0 or gt_bbox.shape[0] == 0:
#             pass
#         else:
#             TPs.append(evaluate_pseudo_label(det_bbox, det_label, 
#                                         gt_bbox, gt_label))
#     if torch.cat(det_bboxes, dim=0).shape[0] > 0 and len(TPs) > 0:
#         TPs = torch.cat(TPs, dim=0)
#         num_tp, num_fp = TPs.sum(), (~TPs).sum()
#         num_gts = sum([gt_bbox.shape[0] for gt_bbox in gt_bboxes])
#         precision = num_tp / (num_tp + num_fp)
#         recall = num_tp / torch.tensor(num_gts, dtype=num_tp.dtype, device=num_tp.device)
        
#     else:
#         precision = 0
#         recall = 0
#     return precision, recall

def evaluate_pseudo_label(det_bboxes, det_labels, gt_bboxes, gt_labels, thres=0.5):
    """ Perform evaluation on pseudo boxes. 
    """
    # 计算检测框和真实框的面积
    area1 = (det_bboxes[:, 2:4] - det_bboxes[:, 0:2]).prod(dim=1)  # 检测框面积
    area2 = (gt_bboxes[:, 2:4] - gt_bboxes[:, 0:2]).prod(dim=1)    # 真实框面积
    
    # 计算交集区域
    lt = torch.max(det_bboxes[:, None, :2], gt_bboxes[None, :, :2])  # 左上角最大值
    rb = torch.min(det_bboxes[:, None, 2:4], gt_bboxes[None, :, 2:4])  # 右下角最小值
    wh = torch.clamp(rb - lt, min=0)  # 宽高
    overlaps = wh[..., 0] * wh[..., 1]  # 交集面积
    
    # 计算 IoU
    ious = overlaps / (area1[:, None] + area2[None, :] - overlaps + 1e-8)
    
    # 找到每个检测框的最佳匹配
    max_iou, argmax_iou = ious.max(dim=1)
    
    # 判断是否为 True Positive
    flags = (max_iou > thres) & (det_labels == gt_labels[argmax_iou])
    return flags


def get_pseudo_label_quality(det_bboxes, det_labels, gt_bboxes, gt_labels):
    """ precision, recall, and statistics of pseudo labels.  
    """
    # 初始化统计量
    num_tp = 0
    num_fp = 0
    num_gts = 0
    
    # 遍历每张图片，逐图片计算 TPs
    for det_bbox, det_label, gt_bbox, gt_label in zip(
        det_bboxes, det_labels, gt_bboxes, gt_labels
    ):
        if det_bbox.shape[0] == 0 or gt_bbox.shape[0] == 0:
            # 如果没有检测框或真实框，仅累加真实框数量
            num_gts += gt_bbox.shape[0]
            continue
        
        # 计算单张图片的 TPs
        TPs = evaluate_pseudo_label(det_bbox, det_label, gt_bbox, gt_label)
        
        # 累加 TP 和 FP
        num_tp += TPs.sum().item()  # True Positives
        num_fp += (~TPs).sum().item()  # False Positives
        num_gts += gt_bbox.shape[0]  # Ground Truths
    
    # 返回 precision, recall 和统计量
    return num_tp, num_fp, num_gts

def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
        [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
    ).reshape(
        -1, 2
    )  # n*4,2


def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor."""
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int],
    dsize_dst: Tuple[int, int],
) -> torch.Tensor:
    check_is_tensor(dst_pix_trans_src_pix)

    if not (
        len(dst_pix_trans_src_pix.shape) == 3
        or dst_pix_trans_src_pix.shape[-2:] == (3, 3)
    ):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(
                dst_pix_trans_src_pix.shape
            )
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(
        dst_pix_trans_src_pix
    )
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix.float()).to(
        src_norm_trans_src_pix.dtype
    )
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(
        dst_pix_trans_src_pix
    )

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        raise TypeError(
            "Input src type is not a torch.Tensor. Got {}".format(type(src))
        )

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        message: str = (
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpAffine."
        )
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine

    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), dsize)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm.float())

    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        [B, C, dsize[0], dsize[1]],
        align_corners=align_corners,
    )

    return F.grid_sample(
        src.float(),
        grid,
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode,
    ).to(src.dtype)


class Transform2D:
    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_masks(
        mask: Union[BitmapMasks, List[BitmapMasks]],
        M: Union[torch.Tensor, List[torch.Tensor]],
        out_shape: Union[list, List[list]],
    ):
        if isinstance(mask, Sequence):
            assert len(mask) == len(M)
            return [
                Transform2D.transform_masks(b, m, o)
                for b, m, o in zip(mask, M, out_shape)
            ]
        else:
            if mask.masks.shape[0] == 0:
                return BitmapMasks(np.zeros((0, *out_shape)), *out_shape)
            mask_tensor = (
                torch.from_numpy(mask.masks[:, None, ...]).to(M.device).to(M.dtype)
            )
            return BitmapMasks(
                warp_affine(
                    mask_tensor,
                    M[None, ...].expand(mask.masks.shape[0], -1, -1),
                    out_shape,
                )
                .squeeze(1)
                .cpu()
                .numpy(),
                out_shape[0],
                out_shape[1],
            )

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]

            return (
                warp_affine(img.float(), M[None, ...], out_shape, mode="nearest")
                .squeeze()
                .to(img.dtype)
            )
# def filter_bboxes(proposal_list, proposal_label_list, average_cosine_sims, score_thr_1=0.7, score_thr_2=0.5, sim_thr=0.3):
#     """
#     二次筛选检测框：
#     1. 第一次筛选：保留分数大于score_thr_1的框。
#     2. 第二次筛选：对于分数在score_thr_2到score_thr_1之间的框，如果sim大于sim_thr，则保留。
#     """
#     filtered_proposal_list = []
#     filtered_proposal_label_list = []

#     for img_idx, (proposals, labels, avg_sim) in enumerate(zip(proposal_list, proposal_label_list, average_cosine_sims)):
#         scores = proposals[:, -1]

#         # 第一次筛选：分数大于0.7的框
#         proposals_1, labels_1, _, valid_indices_1 = filter_invalid(
#             proposals, label=labels, score=scores, thr=score_thr_1, return_inds=True
#         )

#         # 第二次筛选：分数在0.5到0.7之间且sim大于0.5的框
#         valid_indices_2 = (scores >= score_thr_2) & (scores <= score_thr_1) & (avg_sim > sim_thr)
#         proposals_2 = proposals[valid_indices_2]
#         labels_2 = labels[valid_indices_2]

#         # 合并两次筛选的结果
#         filtered_proposals = torch.cat([proposals_1, proposals_2], dim=0)
#         filtered_labels = torch.cat([labels_1, labels_2], dim=0)

#         filtered_proposal_list.append(filtered_proposals)
#         filtered_proposal_label_list.append(filtered_labels)

#     return filtered_proposal_list, filtered_proposal_label_list

import torch

def filter_bboxes(proposal_list, proposal_label_list, average_cosine_sims, score_thr_1=0.7, score_thr_2=0.5, sim_thr=0.3, return_inds=False):
    """
    二次筛选检测框：
    1. 第一次筛选：保留分数大于score_thr_1的框。
    2. 第二次筛选：对于分数在score_thr_2到score_thr_1之间的框，如果sim大于sim_thr，则保留。
    """
    filtered_proposal_list = []
    filtered_proposal_label_list = []
    filtered_indices_list = []

    for img_idx, (proposals, labels, avg_sim) in enumerate(zip(proposal_list, proposal_label_list, average_cosine_sims)):
        scores = proposals[:, -1]

        # 第一次筛选：分数大于score_thr_1的框
        _, _, _, valid_indices_1 = filter_invalid(
            proposals, label=labels, score=scores, thr=score_thr_1, return_inds=True
        )
#         print('valid_indices_1',valid_indices_1)#[boor,boor,...,boor]
        proposals_1 = proposals[valid_indices_1]
        labels_1 = labels[valid_indices_1]

        # 第二次筛选：分数在score_thr_2到score_thr_1之间且sim大于sim_thr的框
        valid_indices_2 = (scores >= score_thr_2) & (scores < score_thr_1) & (avg_sim > sim_thr)
#         print('valid_indices_2',valid_indices_2)
        proposals_2 = proposals[valid_indices_2]
        labels_2 = labels[valid_indices_2]

        # 将布尔掩码转换为索引数组
        indices_1 = torch.nonzero(valid_indices_1, as_tuple=True)[0]
        indices_2 = torch.nonzero(valid_indices_2, as_tuple=True)[0]

        # 合并索引数组并去重
        unique_indices = torch.unique(torch.cat([indices_1, indices_2]))
#         print(unique_indices)
        # 使用唯一索引提取结果
        filtered_proposals = proposals[unique_indices]
        filtered_labels = labels[unique_indices]

        filtered_proposal_list.append(filtered_proposals)
        filtered_proposal_label_list.append(filtered_labels)

        if return_inds:
            filtered_indices_list.append(unique_indices)

    if return_inds:
        return filtered_proposal_list, filtered_proposal_label_list, filtered_indices_list
    else:
        return filtered_proposal_list, filtered_proposal_label_list

def filter_invalid(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0, return_inds=False):
    bbox_ = bbox.clone()
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    idx_1 = torch.nonzero(valid).reshape(-1)

    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
        
        idx_2 = idx_1[valid] 
        idx = torch.zeros(bbox_.shape[0], device=idx_2.device).scatter_(
                    0, idx_2, torch.ones(idx_2.shape[0], device=idx_2.device)).bool()
    
    if not return_inds:
        return bbox, label, mask
    else:
        return bbox, label, mask, idx

def filter_invalid_classwise(bbox, label=None, score=None, class_acc=None, thr=0.0, min_size=0):
    if class_acc.max() > 0:
        class_acc = class_acc / class_acc.max()
    
    thres = thr * (0.375 * class_acc[label] + 0.625)
    select = score.ge(thres).bool()
    
    bbox = bbox[select]
    label = label[select]
    
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
    
    return bbox, label

def filter_invalid_scalewise(bbox, label=None, score=None, scale_acc=None, thr=0.0, min_size=0, return_inds=False):
    bbox_ = bbox.clone()
    
    bw = bbox[:, 2] - bbox[:, 0]
    bh = bbox[:, 3] - bbox[:, 1]
    area = bw * bh 
    
    scale_range = torch.pow(torch.linspace(0, 256, steps=9), 2)
    scale_range = torch.cat([scale_range, torch.tensor([1e8])])
    
    scale_label = - torch.ones(bbox.shape[0], dtype=torch.long)
    for idx in range(scale_range.shape[0] - 1):
        inds = (area > scale_range[idx]) & (area < scale_range[idx+1])
        scale_label[inds] = idx 

    # normalize 
    if scale_acc.max() > 0:
        scale_acc = scale_acc / scale_acc.max()
    thres = thr * (0.375 * scale_acc[scale_label] + 0.625)
    select = score.ge(thres).bool()
    bbox = bbox[select]
    label = label[select]
    idx_1 = torch.nonzero(select).reshape(-1)

    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        
        idx_2 = idx_1[valid] 
        idx = torch.zeros(bbox_.shape[0], device=idx_2.device).scatter_(
                    0, idx_2, torch.ones(idx_2.shape[0], device=idx_2.device)).bool()
    
    if not return_inds:
        return bbox, label
    else:
        return bbox, label, idx