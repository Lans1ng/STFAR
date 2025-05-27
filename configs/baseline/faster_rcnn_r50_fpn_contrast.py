custom_imports = dict(imports=['tta'], allow_failed_imports=False)
mmdet_base = "../../thirdparty/mmdetection/configs/_base_"
_base_ = [
    f"{mmdet_base}/models/faster_rcnn_r50_fpn.py",
]
model = dict(
    roi_head=dict(
        type='ContrastiveRoIHead',
        bbox_head=dict(
            type='ContrastiveBBoxHead',
            num_shared_fcs=2,
            mlp_head_channels=128,
            with_weight_decay=True,
            loss_contrast=dict(
                type='ContrastiveLoss',
                temperature=0.5, #默认0.2，temperature越大，softmax 更平缓，正负样本之间不容易被拉开
                iou_threshold=0.8,
                loss_weight=0.1,
                reweight_type='none'),
            scale=20,#20
            learnable_scale=True,
#             init_cfg=[
#                 dict(
#                     type='Caffe2Xavier',
#                     override=dict(type='Caffe2Xavier', name='shared_fcs')),
#                 dict(
#                     type='Normal',
#                     override=dict(type='Normal', name='fc_cls', std=0.01)),
#                 dict(
#                     type='Normal',
#                     override=dict(type='Normal', name='fc_reg', std=0.001)),
#                 dict(
#                     type='Xavier',
#                     override=dict(
#                         type='Xavier', name='contrastive_head'))
#             ]
        )),
#     train_cfg=dict(
#         rpn_proposal=dict(max_per_img=2000),
#         rcnn=dict(
#             assigner=dict(pos_iou_thr=0.4, neg_iou_thr=0.4, min_pos_iou=0.4),
#             sampler=dict(num=256)))
)