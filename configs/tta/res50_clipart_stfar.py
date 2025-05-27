custom_imports = dict(imports=['_ttt'], allow_failed_imports=False)
mmdet_base = "../../thirdparty/mmdetection/configs/_base_"
_base_ = [
    f"../baseline/faster_rcnn_r50_fpn_contrast.py",
    f"{mmdet_base}/datasets/voc0712.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py",
]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
    ),
    roi_head=dict(bbox_head=dict(num_classes=20)))


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1000, 400), (1000, 600)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1000, 400), (1000, 600)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type=k)
                            for k in [
                                "Identity",
                                "AutoContrast",
                                "RandEqualize",
                                "RandSolarize",
                                "RandColor",
                                "RandContrast",
                                "RandBrightness",
                                "RandSharpness",
                                "RandPosterize",
                            ]
                        ],
                    ),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1000, 400), (1000, 600)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]

unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'ClipartDataset'
data_root = '/root/autodl-fs/_DATASETS/clipart/'


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ClipartDataset',
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root ,
        pipeline=unsup_pipeline,
        filter_empty_gt=False,),
    val=dict(
        type='ClipartDataset',
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type='ClipartDataset',
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root ,
        pipeline=test_pipeline))


semi_wrapper = dict(
    type="STFAR",
    model="${model}",
    train_cfg=dict(
        align_loss_weight=0.1,
        ema_length=64,
        ema_length_instance = 64,
        use_teacher_proposal=False,
        frozen_afterbackbone = True,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.7,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        align_global = True,
        unsup_ssod= True,
        num_classes=20,
        backbone_layer = [0,1,2,3,4],
        image_feature_path='./feat_statistics/voc_image_feature.pkl',
        instance_feature_path = './feat_statistics/voc_instance_feature.pkl',
    ),
    test_cfg=dict(inference_on="student"),
)


custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]

#evaluation = dict(type="SubModulesDistEvalHook", interval=1)  # Change interval to 1 for epoch-based evaluation

evaluation = dict(type="TTAEvalHook", interval=1, classwise=False)

optimizer = dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0001)

# Update learning rate schedule to be epoch-based
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.00001,
    step=[8, 11]  # Epoch-based schedule; these should be epochs (e.g., 8 and 11)
)

# Use EpochBasedRunner
runner = dict(_delete_ = True, type="EpochBasedRunner", max_epochs=10)

checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=1)  # Change by_epoch to True

fp16 = dict(loss_scale="dynamic")

load_from = 'pretrain/res50_voc.pth'
work_dir = "work_dirs/${cfg_name}/"
log_config = dict(
    interval=30,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="pre_release",
        #         name="${cfg_name}",
        #         config=dict(
        #             fold="${fold}",
        #             percent="${percent}",
        #             work_dirs="${work_dir}",
        #             total_step="${runner.max_iters}",
        #         ),
        #     ),
        #     by_epoch=False,
        # ),
    ],
)