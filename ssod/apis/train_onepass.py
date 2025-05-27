import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmcv.runner.hooks import HOOKS
from mmcv.utils import build_from_cfg
from mmdet.core import EvalHook
from mmdet.datasets import build_dataset, replace_ImageToTensor

from ssod.datasets import build_dataloader
from ssod.utils import find_latest_checkpoint, get_root_logger, patch_runner
from ssod.utils.hooks import DistEvalHook
import numpy as np
# from mmcv.runner import IterLoader
import torch.utils.data as torch_data
import json
import os
# import sys
# sys.setrecursionlimit(9000000) #这里设置大一些




def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def train_detector(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    logger = get_root_logger(log_level=cfg.log_level)

    # dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    optimizer = build_optimizer(model, cfg.optimizer)

    mini_batch_indices = []
    correct = []
    if len(dataset[0].CLASSES) == 8:
        # Cityscape
        bs = 128
    else:
        bs = 256
    mini_batch_length = 4096

    train_loader = build_dataloader(dataset[0], bs, cfg.data.workers_per_gpu,len(cfg.gpu_ids),dist=distributed,
                                             seed=cfg.seed,sampler_cfg=cfg.data.get("sampler", {}).get("train", {}), shuffle=False)

    accumulate_category_aps_teacher = None
    accumulate_category_aps_student = None
    accumulate_counts_teacher = None
    accumulate_counts_student = None
    
    for idx, data_train in enumerate(train_loader):
        # print('type', type(data_train))  #dict_keys(['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'idx'])
        indices = data_train['idx'].unique()
        del data_train
        mini_batch_indices.extend(indices.tolist())
        mini_batch_indices = mini_batch_indices[-mini_batch_length:]
        print('-----------------------len--------------------------', len(mini_batch_indices))
        
        tr_dataset_subset = torch_data.Subset(dataset[0], mini_batch_indices)
        tr_dataloader = [build_dataloader(tr_dataset_subset, cfg.data.samples_per_gpu,cfg.data.workers_per_gpu,len(cfg.gpu_ids),dist=distributed,
                                             seed=cfg.seed,sampler_cfg=cfg.data.get("sampler", {}).get("train", {}))]

        # epoch_times = 2
        # import ipdb; ipdb.set_trace()
        epoch_times =  cfg['model']['train_cfg']['epoch_times']                                                                                                                                                                          
        cfg.runner[ 'max_iters'] =  int (len(mini_batch_indices)  / cfg.data.samples_per_gpu * epoch_times )  
        cfg.log_config['interval'] = int (len(mini_batch_indices)  / cfg.data.samples_per_gpu ) 
        # cfg.runner[ 'max_iters'] = 1

        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta,
            ),
        )

        # an ugly workaround to make .log and .log.json filenames the same
        runner.timestamp = timestamp

        # fp16 setting
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        elif distributed and "type" not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

  
        # register hooks

        # cfg.lr_config
        flag = 0
        for lr_key in list(cfg.lr_config.keys()):
            if lr_key is ('policy'):
                flag = 1  #表示已经有policy这个了
                cfg.lr_config['policy'] = 'fixed'
            else:
                cfg.lr_config.pop(lr_key)

        if flag == 0 :
            cfg.lr_config['policy'] = 'fixed'


        runner.register_training_hooks(
            cfg.lr_config,
            optimizer_config,
            cfg.checkpoint_config,
            cfg.log_config,
            cfg.get("momentum_config", None),
        )
        if distributed:
            if isinstance(runner, EpochBasedRunner):
                runner.register_hook(DistSamplerSeedHook())

        # register eval hooks
        if validate:
            # Support batch_size > 1 in validation
            val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
            if val_samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
            data_val_cfg = cfg.data.val.copy()
            # data_val_cfg['indices'] = range(dataset[0].__len__())
            data_val_cfg['indices'] = indices
            val_dataset = build_dataset(data_val_cfg, dict(test_mode=True))
            del data_val_cfg

            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=val_samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
            )
            
            eval_cfg = cfg.get("evaluation", {})
            
            # eval_cfg['interval'] =  int (len(mini_batch_indices)  / cfg.data.samples_per_gpu )  
            eval_cfg['interval'] = cfg.runner[ 'max_iters']

            eval_cfg["by_epoch"] = eval_cfg.get(
                "by_epoch", cfg.runner["type"] != "IterBasedRunner"
            )
            if "type" not in eval_cfg:
                eval_hook = DistEvalHook if distributed else EvalHook
                eval_hook = eval_hook(val_dataloader, **eval_cfg)
            else:
                eval_hook = build_from_cfg(
                    eval_cfg, HOOKS, default_args=dict(dataloader=val_dataloader)
                )

            runner.register_hook(eval_hook, priority=80)

        # user-defined hooks
        if cfg.get("custom_hooks", None):
            custom_hooks = cfg.custom_hooks
            assert isinstance(
                custom_hooks, list
            ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
            for hook_cfg in cfg.custom_hooks:
                assert isinstance(hook_cfg, dict), (
                    "Each item in custom_hooks expects dict type, but got "
                    f"{type(hook_cfg)}"
                )
                hook_cfg = hook_cfg.copy()
                priority = hook_cfg.pop("priority", "NORMAL")
                hook = build_from_cfg(hook_cfg, HOOKS)
                runner.register_hook(hook, priority=priority)

        runner = patch_runner(runner)
        # resume_from = None

        # if cfg.get("auto_resume", True):
        #     resume_from = find_latest_checkpoint(cfg.work_dir)
        # if resume_from is not None:
        #     cfg.resume_from = resume_from

        # if cfg.resume_from:
        #     runner.resume(cfg.resume_from)
        # elif cfg.load_from:
        if idx == 0:
            runner.load_checkpoint(cfg.load_from)

       
        runner.run(tr_dataloader, cfg.workflow)

        del tr_dataset_subset
        del tr_dataloader
        del val_dataset
        del val_dataloader
        # import ipdb;ipdb.set_trace()

        # eval_categorical_info_teacher = getattr(runner,'eval_res_teacher')['categorical_info'] # AP, Count
        # eval_categorical_info_teacher['Count'] = np.array(eval_categorical_info_teacher['Count'])
        # eval_categorical_info_teacher['AP'] = np.array(eval_categorical_info_teacher['AP'])
        # if accumulate_category_aps_teacher is None:
        #     accumulate_category_aps_teacher = eval_categorical_info_teacher['AP']
        #     accumulate_counts_teacher = eval_categorical_info_teacher['Count']
        # else:
        #     accumulate_category_aps_teacher = (eval_categorical_info_teacher['AP'] * eval_categorical_info_teacher['Count'] + accumulate_category_aps_teacher * accumulate_counts_teacher) / (accumulate_counts_teacher + eval_categorical_info_teacher['Count'])
        #     accumulate_counts_teacher += eval_categorical_info_teacher['Count']
        #     accumulate_category_aps_teacher[accumulate_counts_teacher == 0] = 0.

        # print('[INFO] TTT accumulative accuracy for teacher:', accumulate_category_aps_teacher.sum() / (accumulate_counts_teacher > 0).sum()) 




        eval_categorical_info_teacher = getattr(runner, 'eval_res_teacher')['categorical_info']  # AP, Count
        eval_categorical_info_teacher['Count'] = np.array(eval_categorical_info_teacher['Count'])
        eval_categorical_info_teacher['AP'] = np.array(eval_categorical_info_teacher['AP'])
        if accumulate_category_aps_teacher is None:
            accumulate_category_aps_teacher = eval_categorical_info_teacher['AP']
            accumulate_counts_teacher = eval_categorical_info_teacher['Count']
        else:
            accumulate_category_aps_teacher = (eval_categorical_info_teacher['AP'] * eval_categorical_info_teacher['Count'] + accumulate_category_aps_teacher * accumulate_counts_teacher) / (accumulate_counts_teacher + eval_categorical_info_teacher['Count'])
            accumulate_counts_teacher += eval_categorical_info_teacher['Count']
            accumulate_category_aps_teacher[accumulate_counts_teacher == 0] = 0.

        print('[INFO] TTT accumulative accuracy for teacher:', accumulate_category_aps_teacher.sum() / (accumulate_counts_teacher > 0).sum())


        eval_categorical_info_student = getattr(runner, 'eval_res_student')['categorical_info']  # AP, Count
        eval_categorical_info_student['Count'] = np.array(eval_categorical_info_student['Count'])
        eval_categorical_info_student['AP'] = np.array(eval_categorical_info_student['AP'])
        if accumulate_category_aps_student is None:
            accumulate_category_aps_student = eval_categorical_info_student['AP']
            accumulate_counts_student = eval_categorical_info_student['Count']
        else:
            accumulate_category_aps_student = (eval_categorical_info_student['AP'] * eval_categorical_info_student['Count'] + accumulate_category_aps_student * accumulate_counts_student) / (accumulate_counts_student + eval_categorical_info_student['Count'])
            accumulate_counts_student += eval_categorical_info_student['Count']
            accumulate_category_aps_student[accumulate_counts_student == 0] = 0.

        print('[INFO] TTT accumulative accuracy for student:', accumulate_category_aps_student.sum() / (accumulate_counts_student > 0).sum())

