# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
from mmcv.runner.fp16_utils import force_fp32
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmdet.core.evaluation import EvalHook
from ssod.models.utils import Transform2D, filter_invalid
# from thirdparty.mmdetection.mmdet.datasets.coco import COCO

@HOOKS.register_module()
class TTAEvalHook(EvalHook):
    def __init__(self, 
                 dataloader, 
                 interval=1, 
                 **eval_kwargs):  
        super().__init__(dataloader, **eval_kwargs)
        self.dataloader = dataloader
        self.classes = dataloader.dataset.CLASSES
        self.interval = interval
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes    
        
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]
        
    # def before_run(self, runner):
    #     self.eval_hook.before_train_epoch(runner)
        
    def before_train_epoch(self, runner):
        self.total_results = []
        
    # def before_train_iter(self, runner):
    #     self.eval_hook.after_train_iter(runner)
        
    def after_train_iter(self, runner):
        teacher_info = runner.outputs['teacher_info']
        
        # Extract the transformation matrices
        M = teacher_info["transform_matrix"]
        # print('######',teacher_info["init_bboxes"])

        init_bboxes = self._transform_bbox(
            teacher_info["init_bboxes"],  # Original bounding boxes
            [m.inverse() for m in M],    # Inverse of each transformation matrix
            [meta["img_shape"] for meta in teacher_info["img_metas"]],  # Image shapes
        )

        init_labels = teacher_info["init_labels"]
         
        for boxes, labels in zip(init_bboxes,init_labels):
            image_results = [np.empty((0, 5)) for i in range(len(self.classes))]
            for box, label in zip(boxes,labels):
                label = label.item()
                box = box.cpu().numpy()
                image_results[label] = np.concatenate([image_results[label], [box]], axis=0)
            self.total_results.append(image_results)

    # def evaluate(self, runner, results, prefix=""):
    #     eval_res = runner.data_loader.dataset.evaluate(
    #         results, logger=runner.logger
    #     )
    #     for name, val in eval_res.items():
    #         runner.log_buffer.output[name] = val
    #     runner.log_buffer.ready = True

    #     if self.save_best is not None:
    #         if self.key_indicator == 'auto':
    #             # infer from eval_results
    #             self._init_rule(self.rule, list(eval_res.keys())[0])
    #         return eval_res[self.key_indicator]

    #     return None
        
    def after_train_epoch(self, runner):
        print('starting evaluation')
        print('--'*20)
        self.evaluate(runner, self.total_results)









