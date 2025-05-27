import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid
import numpy as np

def post_process(bbox_results, proposals , rois, img_metas):
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
        bbox_pred = (None, ) * len(proposals)
    bbox_conv_feats = bbox_results['bbox_conv_feats'].split(num_proposals_per_img, 0)
    return rois, cls_score, bbox_pred, img_shapes, scale_factors, bbox_conv_feats

def bbox2result(bboxes, labels,convbbox_feats,num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    
    if bboxes.shape[0] == 0:
        return [torch.zeros([0, 5], dtype=torch.float).cuda() for i in range(num_classes)] ,[torch.zeros([0, 1024], dtype=torch.float).cuda() for i in range(num_classes)]
    else:
        return [bboxes[labels == i, :] for i in range(num_classes)] ,[convbbox_feats[labels == i, :] for i in range(num_classes)]




@DETECTORS.register_module()
class SoftTeacher_ORI(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
       
        super(SoftTeacher_ORI, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            
        )
        if train_cfg is not None:
            
            self.freeze("teacher")
            self.unsup_ssod = self.train_cfg.unsup_ssod
            self.unsup_weight = self.train_cfg.unsup_weight
 
            self.layer = self.train_cfg.backbone_layer

            self.align_global = self.train_cfg.align_global
            self.ema_length  = self.train_cfg.global_ema_length
            self.loss_scale_global = self.train_cfg.loss_scale_global
            self.cov_allfeat_baseline_all_path = self.train_cfg.cov_allfeat_baseline_all_path
            self.mean_allfeat_baseline_all_path = self.train_cfg.mean_allfeat_baseline_all_path


            self.align_globl_boxfeat = self.train_cfg.align_globl_boxfeat
            self.align_box_cls_global = self.train_cfg.align_box_cls_global
            self.align_box_reg_global = self.train_cfg.align_box_reg_global
            self.ema_length_cls_global  = self.train_cfg.ema_length_cls_global
            self.ema_length_reg_global  = self.train_cfg.ema_length_reg_global
            self.loss_scale_global_boxcls = self.train_cfg.loss_scale_global_boxcls
            self.loss_scale_global_boxreg =self.train_cfg.loss_scale_global_boxreg
            self.save_cov_boxfeat_mean_global_path = self.train_cfg.save_cov_boxfeat_mean_global_path
            self.save_cov_boxfeat_cov_global_path = self.train_cfg.save_cov_boxfeat_cov_global_path
            self.num_classes = self.train_cfg.num_classes

        if self.train_cfg.frozen_afterbackbone:
            self.freeze_after_backbone('student')
            print('print frozen')
            # frozen_backbone(model)

  
        cov_allfeat_baseline_all =  torch.from_numpy(np.load(self.cov_allfeat_baseline_all_path)).float().cuda()[self.layer]
        mean_allfeat_baseline_all = torch.from_numpy(np.load(self.mean_allfeat_baseline_all_path)).float().cuda()[self.layer]
        save_cov_boxfeat_mean_global = torch.from_numpy(np.load(self.save_cov_boxfeat_mean_global_path)).float().cuda()
        save_cov_boxfeat_cov_global = torch.from_numpy(np.load(self.save_cov_boxfeat_cov_global_path)).float().cuda()

        self.iter_bias = -1
        self.iter = 0 

        bias = 0.01
        # bias = torch.linalg.svdvals(cov_allfeat_baseline_all)[0] / 1000.
        print('bias', bias)
        self.template_backbone_cov = torch.eye(256).cuda() * bias
        ext_src_mu = mean_allfeat_baseline_all
        ext_src_cov = cov_allfeat_baseline_all +  self.template_backbone_cov
      
        # ext_src_mu = mean_allfeat_baseline_all
        # ext_src_cov = cov_allfeat_baseline_all+  self.template_backbone_cov
        self.source_component_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)
        self.target_compoent_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)
        self.ema_ext_mu = ext_src_mu.clone()
        self.ema_ext_cov = ext_src_cov.clone()
        self.ema_total_n = 0.

        bias_boxcls = 0.01
        # bias_boxcls = torch.linalg.svdvals(save_cov_boxfeat_cov_global)[0] / 1000. 
        print('bias_boxcls',bias_boxcls)
        self.template_global_boxclsfeat_cov = torch.eye(1024).cuda() * bias_boxcls
        global_boxcls_src_mu = save_cov_boxfeat_mean_global
        global_boxcls_src_cov =  save_cov_boxfeat_cov_global + self.template_global_boxclsfeat_cov 
        self.source_global_boxcls_distribution = torch.distributions.MultivariateNormal(global_boxcls_src_mu, global_boxcls_src_cov)
        self.target_global_boxcls_distribution = torch.distributions.MultivariateNormal(global_boxcls_src_mu, global_boxcls_src_cov)
        self.global_boxcls_ext_mu = global_boxcls_src_mu.clone()
        self.global_boxcls_ext_cov = global_boxcls_src_cov.clone()       
        self.ema_total_n_cls = 0.
        self.student_cls_rcnn_cfg = self.student.test_cfg.rcnn.copy()
        # self.student_cls_rcnn_cfg ['score_thr'] = 0.1
   
        bias_boxreg = 0.01
        self.template_global_boxregfeat_cov = torch.eye(1024).cuda() * bias_boxreg
        global_boxreg_src_mu = save_cov_boxfeat_mean_global
        global_boxreg_src_cov =  save_cov_boxfeat_cov_global + self.template_global_boxregfeat_cov 
        self.source_global_boxreg_distribution = torch.distributions.MultivariateNormal(global_boxreg_src_mu, global_boxreg_src_cov)
        self.target_global_boxreg_distribution = torch.distributions.MultivariateNormal(global_boxreg_src_mu, global_boxreg_src_cov)
        self.global_boxreg_ext_mu = global_boxreg_src_mu.clone()
        self.global_boxreg_ext_cov = global_boxreg_src_cov.clone()   
        self.ema_total_n_reg = 0.
        self.student_reg_rcnn_cfg = self.student.test_cfg.rcnn.copy()
        # self.student_reg_rcnn_cfg ['score_thr'] = 0.1

        # self.student_cls_rcnn_cfg ['nms'][ 'iou_threshold'] = 0.
        # self.student_cls_rcnn_cfg ['max_per_img'] = 1000
        # self.student_reg_rcnn_cfg ['score_thr'] = 0.
        # print('self.student.test_cfg.rcnn',self.student.test_cfg.rcnn)
        # print(' self.student_cls_rcnn_cfg', self.student_cls_rcnn_cfg)
        # print(' self.student_reg_rcnn_cfg', self.student_reg_rcnn_cfg)



    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        self.iter = self.iter + 1
        loss = {}
      
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            unsup_loss_init , backbone_feature ,  box_convfeat_cls, box_convfeat_reg, teacher_info_new, student_info_new =  self.foward_unsup_train(
            # backbone_feature ,  box_convfeat_cls, box_convfeat_reg =  self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                )
            if self.unsup_ssod:
                unsup_loss = weighted_loss(  unsup_loss_init,weight=self.unsup_weight,)
                unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
                loss.update(**unsup_loss)



        if self.align_global:
           
            self.ema_total_n += len(box_convfeat_cls)
          
            backbone_feature_pooling = torch.nn.AdaptiveAvgPool2d((1))(backbone_feature[self.layer]).reshape(backbone_feature[self.layer].shape[0], -1)
            delta_pre = (backbone_feature_pooling -  self.ema_ext_mu[None,:]) 
            alpha =  1. / self.ema_length if self.ema_total_n > self.ema_length  else 1. / (self.ema_total_n  + 1e-10)
            delta = alpha * delta_pre.sum(dim=0) # N, D
          
            new_component_mean = self.ema_ext_mu + delta
            new_component_cov = self.ema_ext_cov \
                                + alpha * ((delta_pre.T @ delta_pre) - self.ema_ext_cov) \
                                - delta[None,:].T @ delta[None,:]
            with torch.no_grad():
                self.ema_ext_mu = new_component_mean.detach()
                self.ema_ext_cov = new_component_cov.detach()
            
            if self.iter > self.iter_bias:
                self.target_compoent_distribution.loc = new_component_mean
                self.target_compoent_distribution.covariance_matrix = new_component_cov + self.template_backbone_cov
                self.target_compoent_distribution._unbroadcasted_scale_tril = torch.linalg.cholesky(new_component_cov + self.template_backbone_cov)
                dis_loss = {'align_global_loss': (torch.distributions.kl_divergence(self.source_component_distribution, self.target_compoent_distribution) \
                        + torch.distributions.kl_divergence(self.target_compoent_distribution, self.source_component_distribution)) * self.loss_scale_global} 
                loss.update( dis_loss )

        if self.align_globl_boxfeat: 
            if self.align_box_cls_global:
                box_convfeat_cls_list = []
                for i in range(len(box_convfeat_cls)):
                    boxcls_preimg = torch.cat([box_convfeat_cls[i][class_i] for class_i in range(len(box_convfeat_cls[0]))])
                    if boxcls_preimg.shape[0] != 0 :
                        boxcls_preimg_mean = boxcls_preimg.mean(0,True)
                        box_convfeat_cls_list.append(boxcls_preimg_mean)
                if len(box_convfeat_cls_list) > 0 :
                    box_convfeat_cls_list_global = torch.cat(box_convfeat_cls_list)

            if self.align_box_reg_global:
                box_convfeat_reg_list = []
                for i in range(len(box_convfeat_cls)):
                    boxreg_preimg = torch.cat([box_convfeat_reg[i][class_i] for class_i in range(len(box_convfeat_reg[0]))])
                    if boxreg_preimg.shape[0] != 0 :
                        box_convfeat_reg_list.append(boxreg_preimg.mean(0,True))
                if len(box_convfeat_reg_list) > 0 :
                    box_convfeat_reg_list_global = torch.cat(box_convfeat_reg_list)               


        if self.align_globl_boxfeat: 

            if self.align_box_cls_global:
                if len(box_convfeat_cls_list) > 0 :
                    self.ema_total_n_cls += box_convfeat_cls_list_global.shape[0]
                    delta_pre = ( box_convfeat_cls_list_global -  self.global_boxcls_ext_mu) 
                    alpha =  1. / self.ema_length_cls_global if self.ema_total_n_cls > self.ema_length_cls_global  else 1. / (self.ema_total_n_cls  + 1e-10)
                    delta = alpha * delta_pre.sum(dim=0) # N, D
                    # print('delta',delta.shape)
                    new_component_mean = self.global_boxcls_ext_mu + delta
                    new_component_cov = self.global_boxcls_ext_cov \
                                        + alpha * ((delta_pre.T @ delta_pre) - self.global_boxcls_ext_cov) \
                                        - delta[None,:].T @ delta[None,:]
                    with torch.no_grad():
                        self.global_boxcls_ext_mu = new_component_mean.detach()
                        self.global_boxcls_ext_cov = new_component_cov.detach()
                    
                    if self.iter > self.iter_bias:
                        self.target_global_boxcls_distribution.loc = new_component_mean
                        self.target_global_boxcls_distribution.covariance_matrix = new_component_cov + self.template_global_boxclsfeat_cov
                        # print('new_component_cov + self.template_global_boxclsfeat_cov', (new_component_cov + self.template_global_boxclsfeat_cov).shape, torch.all((new_component_cov + self.template_global_boxclsfeat_cov) == 0 ))
                        # if torch.all((new_component_cov + self.template_global_boxclsfeat_cov) == 0 ):
                        
                            # print('new_component_cov + self.template_global_boxclsfeat_cov',(new_component_cov + self.template_global_boxclsfeat_cov).shape)
                        try:
                            self.target_global_boxcls_distribution._unbroadcasted_scale_tril = torch.linalg.cholesky(new_component_cov + self.template_global_boxclsfeat_cov)
                        except:
                            import ipdb;ipdb.set_trace()
                            print("未知异常")
                        global_boxcls_value =(torch.distributions.kl_divergence(self.source_global_boxcls_distribution, self.target_global_boxcls_distribution) \
                                + torch.distributions.kl_divergence(self.target_global_boxcls_distribution, self.source_global_boxcls_distribution)) * self.loss_scale_global_boxcls
                        # if self.iter < 500:
                        #     global_boxcls_value = global_boxcls_value * 0.0
                        dis_cls_loss = {'align_global_boxcls_loss': global_boxcls_value} 
                        
                        loss.update( dis_cls_loss )
            
            if self.align_box_reg_global:
                if len(box_convfeat_reg_list) > 0 :
                    self.ema_total_n_reg += box_convfeat_reg_list_global.shape[0]
                    delta_pre = ( box_convfeat_reg_list_global -  self.global_boxreg_ext_mu) 
                    alpha =  1. / self.ema_length_reg_global if self.ema_total_n_reg > self.ema_length_reg_global  else 1. / (self.ema_total_n_reg  + 1e-10)
                    delta = alpha * delta_pre.sum(dim=0) # N, D
                    # print('delta',delta.shape)
                    new_component_mean = self.global_boxreg_ext_mu + delta
                    new_component_cov = self.global_boxreg_ext_cov \
                                        + alpha * ((delta_pre.T @ delta_pre) - self.global_boxreg_ext_cov) \
                                        - delta[None,:].T @ delta[None,:]
                    with torch.no_grad():
                        self.global_boxreg_ext_mu = new_component_mean.detach()
                        self.global_boxreg_ext_cov = new_component_cov.detach()
                    
                    if self.iter > self.iter_bias:
                        self.target_global_boxreg_distribution.loc = new_component_mean
                        self.target_global_boxreg_distribution.covariance_matrix = new_component_cov + self.template_global_boxregfeat_cov
                        self.target_global_boxreg_distribution._unbroadcasted_scale_tril = torch.linalg.cholesky(new_component_cov + self.template_global_boxregfeat_cov)
                        dis_reg_loss = {'align_global_boxreg_loss': (torch.distributions.kl_divergence(self.source_global_boxreg_distribution, self.target_global_boxreg_distribution) \
                                + torch.distributions.kl_divergence(self.target_global_boxreg_distribution, self.source_global_boxreg_distribution)) * self.loss_scale_global_boxreg} 
                        loss.update( dis_reg_loss )

        # print('loss', loss)
        return loss, teacher_info_new, student_info_new



    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
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
        teacher_info['pseudo_bboxes'] = pseudo_bboxes
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

        
        unsup_rcnn_cls_loss, box_convfeat_cls= self.unsup_rcnn_cls_loss(
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
        
        unsup_rcnn_reg_loss,  box_convfeat_reg= self.unsup_rcnn_reg_loss(
                                                                            student_info["backbone_feature"],
                                                                            student_info["img_metas"],
                                                                            proposals,
                                                                            pseudo_bboxes,
                                                                            pseudo_labels,
                                                                            student_info=student_info,
                                                                        )
        loss.update(unsup_rcnn_reg_loss)
       
        # backbone_feature_res = student_info["backbone_feature_res"]
    
        backbone_feature =  student_info["backbone_feature"]



        return loss , backbone_feature ,  box_convfeat_cls, box_convfeat_reg, teacher_info, student_info
        # return backbone_feature ,  box_convfeat_cls, box_convfeat_reg

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
                    ],  # TODO: replace with foreground score, here is classification score,
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
        rois, cls_score, bbox_pred, img_shapes, scale_factors, bbox_conv_feats =post_process( bbox_results, selected_bboxes , rois, img_metas )
        det_bboxes = []
        det_labels = []
        det_convbbox_feats = []
        
        for i in range(len(selected_bboxes)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if self.student.test_cfg.rcnn is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.student.roi_head.bbox_head.fc_cls.out_features))
            else:
                det_bbox, det_label,keep_inds = self.student.roi_head.bbox_head.get_bboxes(
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
                det_convbbox_feats.append(det_convboxfeat_select)    

        bbox_results_test=[]
        bboxconv_feats = []
        for i in range(len(det_bboxes)):
            bbox_results_test.append(bbox2result(det_bboxes[i], det_labels[i],det_convbbox_feats[i], self.student.roi_head.bbox_head.num_classes)[0])
            bboxconv_feats.append(bbox2result(det_bboxes[i], det_labels[i],det_convbbox_feats[i], self.student.roi_head.bbox_head.num_classes)[1])
        #    len( bboxconv_feats)      4
        #   len( bboxconv_feats[0])     80
        
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
        return loss ,bboxconv_feats

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

        rois, cls_score, bbox_pred, img_shapes, scale_factors, bbox_conv_feats =post_process( bbox_results, bbox_results['proposals'] , bbox_results['rois'], img_metas )
        det_bboxes = []
        det_labels = []
        det_convbbox_feats = []
        
        for i in range(len( bbox_results['proposals'])):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if self.student.test_cfg.rcnn is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.student.roi_head.bbox_head.fc_cls.out_features))
            else:
                det_bbox, det_label,keep_inds = self.student.roi_head.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=True,
                    cfg=self.student_reg_rcnn_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
                det_convboxfeat_select = bbox_conv_feats[i].expand(self.num_classes,-1,-1).permute(1,0,2).reshape(-1,1024)[keep_inds]
                det_convbbox_feats.append(det_convboxfeat_select)    

        bbox_results_test=[]
        bboxconv_feats = []
        for i in range(len(det_bboxes)):
            bbox_results_test.append(bbox2result(det_bboxes[i], det_labels[i],det_convbbox_feats[i], self.student.roi_head.bbox_head.num_classes)[0])
            bboxconv_feats.append(bbox2result(det_bboxes[i], det_labels[i],det_convbbox_feats[i], self.student.roi_head.bbox_head.num_classes)[1])
        #    len( bboxconv_feats)      4
        #   len( bboxconv_feats[0])     80

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
        return {"loss_bbox": loss_bbox["loss_bbox"]} , bboxconv_feats

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

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
     
        feat, feat_res = self.student.extract_feat(img)
        
        student_info["backbone_feature_res"] = feat_res
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

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat, feat_res= self.teacher.extract_feat(img)
        teacher_info["backbone_feature_res"] = feat_res
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

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]

        teacher_info['init_bboxes'] = proposal_list #used to calculate mAP
        teacher_info['init_labels'] = proposal_label_list #used to calculate mAP

        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
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

    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]
  
        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
