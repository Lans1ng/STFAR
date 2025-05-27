# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
from torch import Tensor
from typing_extensions import Literal


@LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mlp_head_channels = 256
        self.fc_out_channels = 1024
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, self.mlp_head_channels))

        
    def forward(self,
                features: Tensor,
                prototypes: Tensor,
                labels: Tensor,
                ious: Tensor,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_weight = self.loss_weight
        if decay_rate is not None:
            loss_weight = self.loss_weight * decay_rate

        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        # ema_features是一个list，stack一下
        prototypes = torch.stack(prototypes).to(features.device)  # (C, D)

        # 找到前景样本（非背景类）
#         foreground_mask = labels != len(prototypes)
#         features = features[foreground_mask]
# #         print('1',labels.shape)
#         labels = labels[foreground_mask]
# #         print('2',labels.shape)
#         ious = ious[foreground_mask]
#         if weight is not None:
#             weight = weight[foreground_mask]
        
        # 如果没有前景样本，直接返回0
        if features.shape[0] == 0:
            return features.sum() * 0

        # 遍历features的label，按照每个样本的label取ema feature
        prototypes_expand = []
        for i, label in enumerate(labels):
            if label != len(prototypes):
                prototype = prototypes[label]
                prototypes_expand.append(prototype)
            else:
                prototypes_expand.append(features[i])                

        prototypes_expand = torch.stack(prototypes_expand, dim=0)  # (N, D)

        # 过MLP头+归一化
        features = self.contrastive_head(features)  # (N, K)
        features = F.normalize(features, dim=1)

        prototypes_expand = self.contrastive_head(prototypes_expand)  # (N, K)
        prototypes_expand = F.normalize(prototypes_expand, dim=1)

        # mask with shape [N, N], mask_{i, j}=1
        # if sample i and sample j have the same label
        labels = labels.unsqueeze(1)
        label_mask = torch.eq(labels, labels.T).float().to(features.device)
#         print(labels.shape)
#         print(label_mask)
#         print('3',label_mask.shape)
        similarity = torch.div(
            torch.matmul(features, prototypes_expand.T), self.temperature)
#         print(similarity.shape)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True) #对每一行的相似度（即每个样本与所有原型的相似度）求最大值，用于数值稳定性。
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
#         logits_mask = torch.ones_like(similarity)
#         logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

#         print(log_prob.shape)
#         print(label_mask.shape)
        per_label_log_prob = (log_prob *label_mask).sum(1) / label_mask.sum(1)

        keep = ious >= self.iou_threshold
#         print(keep.sum())
        if keep.sum() == 0:
            # return zero loss
            return per_label_log_prob.sum() * 0
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        coefficient = self.reweight_func(ious)
        coefficient = coefficient[keep]
        if weight is not None:
            weight = weight[keep]
        loss = loss * coefficient
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss_weight * loss


    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay