import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from mmdet.core import bbox2roi, multi_apply, bbox_overlaps
from mmdet.models import DETECTORS

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n
from ssod.models.utils import Transform2D, filter_invalid, filter_bboxes, get_pseudo_label_quality
from torch.distributions import MultivariateNormal

from .soft_teacher import SoftTeacher
from collections import deque

def post_process(bbox_results, proposals, rois, img_metas):
    img_shapes = tuple(meta['img_shape'] for meta in img_metas)
    scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
    cls_score = bbox_results['cls_score']
    bbox_pred = bbox_results['bbox_pred']

    num_proposals_per_img = tuple(len(p) for p in proposals)

    rois = rois.split(num_proposals_per_img, 0)
    cls_score = cls_score.split(num_proposals_per_img, 0)
    if bbox_pred is not None:
        if isinstance(bbox_pred, torch.Tensor):
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
    else:
        bbox_pred = (None,) * len(proposals)
    bbox_conv_feats = bbox_results['bbox_conv_feats'].split(num_proposals_per_img, 0)
    bbox_feats = bbox_results['bbox_feats'].split(num_proposals_per_img, 0)
    return rois, cls_score, bbox_pred, img_shapes, scale_factors, bbox_conv_feats, bbox_feats


def bbox2result(bboxes, labels, convbbox_feats, roi_feats, num_classes):
    if bboxes.shape[0] == 0:
        return (
            [torch.zeros([0, 5], dtype=torch.float).cuda() for _ in range(num_classes)],
            [torch.zeros([0, 1024], dtype=torch.float).cuda() for _ in range(num_classes)],
            [torch.zeros([0, 256], dtype=torch.float).cuda() for _ in range(num_classes)],
        )
    else:
        return (
            [bboxes[labels == i, :] for i in range(num_classes)],
            [convbbox_feats[labels == i, :] for i in range(num_classes)],
            [roi_feats[labels == i, :] for i in range(num_classes)],
        )

@DETECTORS.register_module()
class STFAR(SoftTeacher):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(STFAR, self).__init__(model, train_cfg, test_cfg)

        self.num_tp_global = 0
        self.num_fp_global = 0
        self.num_gt_global = 0
        
        self.iter_bias = 10
        self.iter = 0 
        self.ema_total_n = 0
        self.ema_total_n_cls = 0
        self.ema_length = train_cfg.get("ema_length", 128)
        self.ema_length_cls_global = train_cfg.get("ema_length_cls_global", 256)
        self.iter_bias = 10
        self.num_classes = train_cfg.get("num_classes", 20)
        self.align_loss_weight = train_cfg.get("align_loss_weight", 0.1)
        self.layers = train_cfg.get("backbone_layer", [4])
        self.bias = 0.01
        
        self.template_backbone_cov = torch.eye(256).cuda() * self.bias
        self.template_instance_cov = torch.eye(1024).cuda() * self.bias
        
        self.source_dist = [None for _ in range(5)]
        self.target_dist = [None for _ in range(5)]
        
        self.ema_total_n_cls_list = [0 for _ in range(self.num_classes)]
        self.class_instance_ema_mu = [None for _ in range(self.num_classes)]
        self.class_instance_ema_cov = [None for _ in range(self.num_classes)]
        self.class_instance_mu = [None for _ in range(self.num_classes)]
        self.class_instance_cov = [None for _ in range(self.num_classes)]
        self.class_instance_source_dist = [None for _ in range(self.num_classes)]
        self.class_instance_target_dist = [None for _ in range(self.num_classes)]

        self.init_layer_distributions(train_cfg.image_feature_path)
        self.init_instance_distributions(train_cfg.instance_feature_path)

        self.store = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.student_cls_rcnn_cfg = self.student.test_cfg.rcnn.copy()
        self.student_reg_rcnn_cfg = self.student.test_cfg.rcnn.copy()
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_ssod = self.train_cfg.unsup_ssod
            self.unsup_weight = self.train_cfg.unsup_weight
            self.layer = self.train_cfg.backbone_layer
            self.align_foreground = self.train_cfg.align_foreground
            self.align_global = self.train_cfg.align_global

        if self.train_cfg.frozen_afterbackbone:
            self.freeze_after_backbone('student')
            
    def init_layer_distributions(self, feature_path):
        self.ema_mu = [None for _ in range(5)]
        self.ema_cov = [None for _ in range(5)]
        self.mu = [None for _ in range(5)]
        self.cov = [None for _ in range(5)]        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(feature_path, 'rb') as file:
            all_features = pickle.load(file)

        for layer in self.layers:
            features = torch.tensor(
                np.array([f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in all_features[layer]]),
                dtype=torch.float32,
                device=device
            ).squeeze(1)
            print(features.shape)
            print('***'*22)
            mu, cov = features.mean(dim=0).to(device), self.compute_cov(features)
            self.mu[layer] = mu.clone()
            self.cov[layer] = cov.clone()
            
            self.ema_mu[layer] = mu.clone()
            self.ema_cov[layer] = cov.clone()
            
            cov= self.template_backbone_cov + cov
            self.source_dist[layer] = MultivariateNormal(mu, cov)
            self.target_dist[layer] = MultivariateNormal(mu, cov)

    def init_instance_distributions(self, feature_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(feature_path, 'rb') as file:
            all_features = pickle.load(file)
        print("Loaded feature keys (class names):", all_features.keys())

        all_class_features = []
        for features in all_features.values():
            all_class_features.extend(features)

        all_features_tensor = torch.tensor(
            np.array([f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in all_class_features]),
            dtype=torch.float32,
            device=device
        ).squeeze(1)

        mu = all_features_tensor.mean(dim=0).to(device)
        cov = self.compute_cov(all_features_tensor)
        cov = self.template_instance_cov + cov

        self.instance_source_dist = MultivariateNormal(mu, cov)
        self.instance_target_dist = MultivariateNormal(mu, cov)

        self.instance_ema_mu = mu.clone()
        self.instance_ema_cov = cov.clone()

        self.instance_mu = mu.clone()
        self.instance_cov = cov.clone()

        for class_name, features in all_features.items():
            class_features_tensor = torch.tensor(
                np.array([f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in features]),
                dtype=torch.float32,
                device=device
            ).squeeze(1)

            class_mu = class_features_tensor.mean(dim=0).to(device)
            class_cov = self.compute_cov(class_features_tensor)
            class_cov = self.template_instance_cov + class_cov
            
            self.class_instance_source_dist[class_name] = MultivariateNormal(class_mu, class_cov)
            self.class_instance_target_dist[class_name] = MultivariateNormal(class_mu, class_cov)
            
            self.class_instance_mu[class_name] = class_mu.clone()
            self.class_instance_cov[class_name] = class_cov.clone()
            self.class_instance_ema_mu[class_name]  = class_mu.clone()
            self.class_instance_ema_cov[class_name]  = class_mu.clone()
    
    def compute_cov(self, features):
        centered = features - features.mean(dim=0)
        covariance = (centered.T @ centered) / (centered.size(0) - 1)
        return covariance

    def update_target_distribution(self, backbone_features):
        self.ema_total_n += 1 
        for layer in self.layers:
            layer_features = torch.nn.AdaptiveAvgPool2d((1))(
                backbone_features[layer]).reshape(backbone_features[layer].shape[0], -1)

            target_mu = self.ema_mu[layer]
            target_cov = self.ema_cov[layer]
            
            delta_pre = layer_features - target_mu
            alpha = 1/self.ema_length
            delta = alpha * delta_pre.sum(dim=0)

            new_mu = target_mu + delta
            new_cov = (target_cov
                       + alpha * ((delta_pre.T @ delta_pre) - target_cov)
                       - delta[None, :].T @ delta[None, :])

            with torch.no_grad():
                self.ema_mu[layer] = new_mu.detach()
                self.ema_cov[layer] = new_cov.detach()
            self.target_dist[layer].loc = new_mu
            self.target_dist[layer].covariance_matrix = new_cov + self.template_backbone_cov
            regularized_cov = new_cov + self.template_backbone_cov 
            self.target_dist[layer]._unbroadcasted_scale_tril = torch.linalg.cholesky(regularized_cov)
    
    def align_target_source_with_kl(self):
        layer_kl_losses = {}

        for layer in self.layers:
            with torch.cuda.amp.autocast(False):
                kl_div = 0.5 * (
                    torch.distributions.kl_divergence(self.source_dist[layer], self.target_dist[layer]) +
                    torch.distributions.kl_divergence(self.target_dist[layer], self.source_dist[layer])
                )
            layer_kl_losses[layer] = kl_div

        return layer_kl_losses
       
    def forward_train(self, img, img_metas, **kwargs):
        self.iter = self.iter + 1
        
        kwargs.update({"img": img, "img_metas": img_metas, "tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        self.iter += 1
        loss = {}

        if "unsup_student" in data_groups:
            unsup_loss_init, _, backbone_feature_s, teacher_info_new, student_info_new, box_convfeat_cls, roi_feats = self.forward_unsup_train(
                data_groups["unsup_teacher"], data_groups["unsup_student"]
            )

            if self.unsup_ssod:
                unsup_loss = weighted_loss(unsup_loss_init, weight=self.unsup_weight)
                loss.update({f"unsup_{k}": v for k, v in unsup_loss.items()})

            if self.align_global:
                self.update_target_distribution(backbone_feature_s)
                layer_kl_losses = self.align_target_source_with_kl()
                for layer, kl_loss in layer_kl_losses.items():
                    loss[f"kl_alignment_loss_layer_{layer}"] = 0.1 * kl_loss

        return loss, teacher_info_new, student_info_new

    def forward_unsup_train(self, teacher_data, student_data):
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
                gt_bboxes = teacher_data['gt_bboxes'],
                gt_labels = teacher_data['gt_labels'],
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]
            
        M = student_info["transform_matrix"]
        gt_bboxes = self._transform_bbox(
            teacher_info['gt_bboxes'], 
            M,   
            [meta["img_shape"] for meta in student_info["img_metas"]],  
        )
        gt_labels = teacher_info['gt_labels']
        num_tp, num_fp, num_gt = get_pseudo_label_quality(
                         pseudo_bboxes, pseudo_labels, gt_bboxes, gt_labels)

        self.num_tp_global += num_tp
        self.num_fp_global += num_fp
        self.num_gt_global += num_gt
        precision = self.num_tp_global / (self.num_tp_global + self.num_fp_global + 1e-8)
        recall = self.num_tp_global / (self.num_gt_global + 1e-8)
        
        unsup_rcnn_cls_loss, box_convfeat_cls, roi_feats= self.unsup_rcnn_cls_loss(
                                                                    student_info["backbone_feature"],
                                                                    student_info["img_metas"],
                                                                    proposals,
                                                                    pseudo_bboxes,
                                                                    pseudo_labels,
                                                                    teacher_info["transform_matrix"],
                                                                    student_info["transform_matrix"],
                                                                    teacher_info["img_metas"],
                                                                    teacher_info["backbone_feature"],
                                                                    student_info=student_info,
                                                                )
        loss.update(unsup_rcnn_cls_loss)
        loss["precision"] = torch.tensor(precision).to(unsup_rcnn_cls_loss["loss_cls"])
        loss["recall"] = torch.tensor(recall).to(unsup_rcnn_cls_loss["loss_cls"])
        return loss , teacher_info['backbone_feature'], student_info["backbone_feature"], teacher_info, student_info, box_convfeat_cls,roi_feats

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]

        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        rois, cls_score, bbox_pred, img_shapes, scale_factors, bbox_conv_feats, bbox_feats = post_process(bbox_results, selected_bboxes, rois, img_metas)
        det_bboxes = []
        det_labels = []
        det_convbbox_feats = []
        det_roi_feats = []
        
        for i in range(len(selected_bboxes)):
            if rois[i].shape[0] == 0:
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if self.student.test_cfg.rcnn is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.student.roi_head.bbox_head.fc_cls.out_features))
            else:
                self.student_cls_rcnn_cfg.score_thr = 0.5
                self.student_cls_rcnn_cfg.nms = dict(type='nms', iou_threshold=1.0)
                self.student_cls_rcnn_cfg.max_per_img = 100
                det_bbox, det_label, keep_inds = self.student.roi_head.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=True,
                    cfg=self.student_cls_rcnn_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
                det_convboxfeat_select = bbox_conv_feats[i].expand(self.num_classes,-1,-1).permute(1,0,2).reshape(-1,1024)[keep_inds]
                bbox_feat=torch.nn.AdaptiveAvgPool2d((1, 1))(bbox_feats[i]).squeeze(-1).squeeze(-1)
                det_roi_feat_select = bbox_feat.expand(self.num_classes,-1,-1).permute(1,0,2).reshape(-1,256)[keep_inds]
                
                det_convbbox_feats.append(det_convboxfeat_select)   
                det_roi_feats.append(det_roi_feat_select)

        self.student_cls_rcnn_cfg.score_thr = 0.05
        self.student_cls_rcnn_cfg.nms = dict(type='nms', iou_threshold=0.5)
        self.student_cls_rcnn_cfg.max_per_img = 100
        
        bbox_results_test=[]
        bboxconv_feats = []
        roi_feats = []
        for i in range(len(det_bboxes)):
            _, bboxconv_feat, roi_feat = bbox2result(det_bboxes[i], det_labels[i], det_convbbox_feats[i], det_roi_feats[i], self.student.roi_head.bbox_head.num_classes)

            bboxconv_feats.append(bboxconv_feat)
            roi_feats.append(roi_feat)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        try:
            proposal_ious = []
            for res in sampling_results:
                single_pos_proposal_ious = bbox_overlaps(
                    res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
                single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                    single_pos_proposal_ious.device)
                single_proposal_ious[:res.pos_bboxes.
                                     size(0)] = single_pos_proposal_ious
                proposal_ious.append(single_proposal_ious)
            proposal_ious = torch.cat(proposal_ious, dim=0)
            loss_contrast = self.student.roi_head.bbox_head.loss_contrast(
                contrast_feat = bbox_results['contrast_feat'],
                ema_feat = self.class_instance_mu,
                proposal_ious = proposal_ious,
                labels=bbox_targets[0])
            loss.update(loss_contrast)
        except:
            pass
        
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss ,bboxconv_feats, roi_feats
    
    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox, bbox_results = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox['loss_bbox']}

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat, _ = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, gt_bboxes=None, gt_labels=None, **kwargs):
        teacher_info = {}
        teacher_info["gt_bboxes"] = gt_bboxes
        teacher_info['gt_labels'] = gt_labels
        feat, _ = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

        teacher_info['init_bboxes'] = proposal_list
        teacher_info['init_labels'] = proposal_label_list
        
        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]

        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info