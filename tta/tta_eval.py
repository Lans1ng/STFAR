# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import os
import pickle
import cv2
import torch
from mmcv.runner.fp16_utils import force_fp32
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmdet.core.evaluation import EvalHook
from ssod.models.utils import Transform2D, filter_invalid

@HOOKS.register_module()
class TTAEvalHook(EvalHook):
    def __init__(self, 
                 dataloader, 
                 interval=1, 
                 classwise=True,
                 cityscape=False,
                 **eval_kwargs):  
        super().__init__(dataloader, **eval_kwargs)
        self.dataloader = dataloader
        self.classes = dataloader.dataset.CLASSES
        self.interval = interval
        self.classwise = classwise
        self.cityscape = cityscape
        
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes    
        
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]
        
    def before_train_epoch(self, runner):
        self.total_results = []
        
    def after_train_iter(self, runner):
        #the keys of teacher_info:
        #['backbone_feature_res', 'backbone_feature', 'proposals', 'init_bboxes', 'init_labels', 'det_bboxes', 'det_labels', 'transform_matrix', 'img_metas', 'pseudo_bboxes'])
        
        teacher_info = runner.outputs['teacher_info']
#         print(teacher_info.keys())
        # Extract the transformation matrices
        M = teacher_info["transform_matrix"]

        init_bboxes = self._transform_bbox(
            teacher_info["init_bboxes"],  # Original bounding boxes
            [m.inverse() for m in M],    # Inverse of each transformation matrix
            [meta["img_shape"] for meta in teacher_info["img_metas"]],  # Image shapes
        )

        init_labels = teacher_info["init_labels"]

        #det_bboxes是阈值大于0.7的box
        det_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],  # Original bounding boxes
            [m.inverse() for m in M],    # Inverse of each transformation matrix
            [meta["img_shape"] for meta in teacher_info["img_metas"]],  # Image shapes
        )
#         print(det_bboxes)

        det_labels = teacher_info['det_labels']
    
        for boxes, labels in zip(init_bboxes, init_labels):
            image_results = [np.empty((0, 5)) for i in range(len(self.classes))]
            for box, label in zip(boxes,labels):
                label = label.item()
                box = box.cpu().numpy()
                image_results[label] = np.concatenate([image_results[label], [box]], axis=0)
            self.total_results.append(image_results)
            
        def draw_bboxes_on_image(image, bboxes, labels, classes, color=(0, 255, 0), thickness=1, font_scale=0.5, font_thickness=1):
            """
            在图像上绘制边界框，并显示类别名称。
            :param image: 输入图像 (numpy array)
            :param bboxes: 边界框列表，每个边界框的格式为 [x1, y1, x2, y2]
            :param labels: 边界框对应的类别标签
            :param classes: 类别名称列表
            :param color: 边界框颜色 (B, G, R)
            :param thickness: 边界框线条宽度
            :param font_scale: 文本字体大小
            :param font_thickness: 文本线条宽度
            :return: 绘制后的图像
            """
            font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体

            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2, *_ = bbox  # 解包 bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 转换为整数

                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                # 创建文本内容
                class_name = classes[label]  # 获取类别名称
                text = class_name

                # 获取文本的宽度和高度
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                # 计算文本框的坐标
                text_x = x1
                text_y = y1 - text_height - 5  # 在边界框上方绘制文本

                # 绘制文本框背景（黑色底色）
                cv2.rectangle(image, (text_x, text_y), (text_x + text_width, text_y + text_height), (0, 0, 0), -1)

                # 绘制文本（白色字体）
                cv2.putText(image, text, (text_x, text_y + text_height), font, font_scale, (255, 255, 255), font_thickness)

            return image
        
        def draw_det_bboxes_on_image(image, bboxes, labels, classes, color=(0, 255, 0), thickness=1, font_scale=0.5, font_thickness=1):
            """
            在图像上绘制边界框，并显示类别、score 和 sim。
            :param image: 输入图像 (numpy array)
            :param bboxes: 检测框列表，每个检测框的格式为 [x1, y1, x2, y2, score, ..., sim]
            :param labels: 检测框对应的类别标签
            :param classes: 类别名称列表
            :param color: 边界框颜色 (B, G, R)
            :param thickness: 边界框线条宽度
            :param font_scale: 文本字体大小
            :param font_thickness: 文本线条宽度
            :return: 绘制后的图像
            """
            font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体

            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2, score, *_ = bbox  # 解包 bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 转换为整数

                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                # 创建文本内容
                class_name = classes[label]  # 获取类别名称
                text = f"{class_name}: {score:.2f}"

                # 获取文本的宽度和高度
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                # 计算文本框的坐标
                text_x = x1
                text_y = y1 - text_height - 5  # 在边界框上方绘制文本

                # 绘制文本框背景（黑色底色）
                cv2.rectangle(image, (text_x, text_y), (text_x + text_width, text_y + text_height), (0, 0, 0), -1)

                # 绘制文本（白色字体）
                cv2.putText(image, text, (text_x, text_y + text_height), font, font_scale, (255, 255, 255), font_thickness)

            return image
        # 获取转换后的 proposals, init_bboxes, 和 gt_boxes
        # 原圖上的gt boxes
        gt_boxes = teacher_info['gt_bboxes']
        gt_labels = teacher_info['gt_labels']
        
        # 如果是 tensor，则转换为 numpy 数组
        if isinstance(det_bboxes, torch.Tensor):
            det_bboxes = det_bboxes.cpu().numpy()
#         print(det_bboxes[0].shape)
            
        if isinstance(init_bboxes, torch.Tensor):
            init_bboxes = init_bboxes.cpu().numpy()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        
        # 加载图像
        img = cv2.imread(teacher_info['img_metas'][0]['filename'])
        
        # 复制图像以分别绘制 proposals, init_bboxes, 和 gt_boxes
        img_det_bboxes = img.copy()
        img_init_bboxes = img.copy()
        img_gt_boxes = img.copy()
        
        # 在第一张图像上绘制 proposals (绿色)
        img_with_det_bboxes = draw_det_bboxes_on_image(
                                img, det_bboxes[0], det_labels[0], self.classes, color=(0, 255, 0), thickness=2
                                                            )
        
        # 在第二张图像上绘制 init_bboxes (红色)
        img_with_init_bboxes = draw_bboxes_on_image(img_init_bboxes, init_bboxes[0], init_labels[0], self.classes, color=(0, 0, 255), thickness=2)
        
        # 在第三张图像上绘制 gt_boxes (蓝色)
        img_with_gt_boxes = draw_bboxes_on_image(img_gt_boxes, gt_boxes[0], gt_labels[0], self.classes, color=(255, 0, 0), thickness=2)
        
        # 设置保存目录
        save_dir_det_bboxes = os.path.join(runner.work_dir, 'vis_det_bboxes')
        save_dir_init_bboxes = os.path.join(runner.work_dir, 'vis_init_bboxes')
        save_dir_gt_boxes = os.path.join(runner.work_dir, 'vis_gt_bboxes')
        
        # 创建目录（如果不存在）
        os.makedirs(save_dir_det_bboxes, exist_ok=True)
        os.makedirs(save_dir_init_bboxes, exist_ok=True)
        os.makedirs(save_dir_gt_boxes, exist_ok=True)
        
        # 获取文件名，不包括扩展名
        basename = os.path.basename(teacher_info['img_metas'][0]['filename'])
        filename_without_ext = os.path.splitext(basename)[0]
        
#         # # 保存 proposal 图像
#         save_path_det_bboxes = os.path.join(save_dir_det_bboxes, f"{filename_without_ext}.jpg")
#         cv2.imwrite(save_path_det_bboxes, img_with_det_bboxes)
        
        # 保存 init_bboxes 图像
#         save_path_init_bboxes = os.path.join(save_dir_init_bboxes, f"{filename_without_ext}.jpg")
#         cv2.imwrite(save_path_init_bboxes, img_with_init_bboxes)
        
        # 保存 gt_boxes 图像
#         save_path_gt_boxes = os.path.join(save_dir_gt_boxes, f"{filename_without_ext}.jpg")
#         cv2.imwrite(save_path_gt_boxes, img_with_gt_boxes)

        # 将三张图像横向拼接
        merged_image = np.hstack(( img_with_init_bboxes,img_with_det_bboxes, img_with_gt_boxes))
        
        # 设置保存目录
        save_dir_merged = os.path.join(runner.work_dir, 'vis_merged_bboxes')
        
        # 创建目录（如果不存在）
        os.makedirs(save_dir_merged, exist_ok=True)
        
        # 获取文件名，不包括扩展名
        basename = os.path.basename(teacher_info['img_metas'][0]['filename'])
        filename_without_ext = os.path.splitext(basename)[0]
        
        # 保存合并后的图像
        save_path_merged = os.path.join(save_dir_merged, f"{filename_without_ext}_merged.jpg")
#         cv2.imwrite(save_path_merged, merged_image)
        
    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        if self.classwise:
#             eval_res = self.dataloader.dataset.evaluate(
#                 results, logger=runner.logger, classwise=True, **self.eval_kwargs, iou_thrs=np.arange(0.5, 0.96, 0.05))
            #for cityscape
            if self.cityscape:
                eval_res = self.dataloader.dataset.evaluate(
                    results, logger=runner.logger, classwise=True, **self.eval_kwargs, iou_thrs=np.arange(0.5, 0.51, 0.05))
            else:
                eval_res = self.dataloader.dataset.evaluate(
                    results, logger=runner.logger, classwise=True, **self.eval_kwargs, iou_thrs=np.arange(0.5, 0.96, 0.05))
        else:
            eval_res = self.dataloader.dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)            
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None

    def after_train_epoch(self, runner):
        print('--' * 20)
        print('starting tta evaluation')
        print('--' * 20)
        
        self.evaluate(runner, self.total_results)
    
        work_dir = runner.work_dir
        
        results_file = os.path.join(work_dir, 'tta_results.pkl')
            
        with open(results_file, 'wb') as f: 
            pickle.dump(self.total_results, f)
        
        print(f'tta_results saved to {results_file}')










