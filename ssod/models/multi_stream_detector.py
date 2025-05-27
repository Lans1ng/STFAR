from typing import Dict
from mmdet.models import BaseDetector, TwoStageDetector
from mmcv.runner import auto_fp16
import torch
from copy import deepcopy
import torch.nn as nn

class MultiSteamDetector(BaseDetector):
    def __init__(
        self, model: Dict[str, TwoStageDetector], train_cfg=None, test_cfg=None
    ):
        # import ipdb ; ipdb.set_trace()
        # model['bbox_head'][ 'loss_cls']['class_weight']= 2.0  
        super(MultiSteamDetector, self).__init__()
        self.submodules = list(model.keys())
        for k, v in model.items():
            setattr(self, k, v)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.inference_on = self.test_cfg.get("inference_on", self.submodules[0])

    def model(self, **kwargs) -> TwoStageDetector:
        
        if "submodule" in kwargs:
            assert (
                kwargs["submodule"] in self.submodules
            ), "Detector does not contain submodule {}".format(kwargs["submodule"])
            model: TwoStageDetector = getattr(self, kwargs["submodule"])
        else:
            model: TwoStageDetector = getattr(self, self.inference_on)
        return model

    def freeze(self, model_ref: str):
        assert model_ref in self.submodules
        model = getattr(self, model_ref)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def freeze_after_backbone(self, model_ref: str):
        model = getattr(self, model_ref)
#         print(model)
        # 冻结 neck 模块
        model_neck = getattr(model, 'neck')
        model_neck.eval()
        for param in model_neck.parameters():
            param.requires_grad = False

        # 冻结 rpn_head 模块
        model_rpn = getattr(model, 'rpn_head')
        model_rpn.eval()
        for param in model_rpn.parameters():
            param.requires_grad = False

        # 冻结 roi_head 模块
        model_roi = getattr(model, 'roi_head')
        model_roi.eval()
        for param in model_roi.parameters():
            param.requires_grad = False

        # 打印可训练的模块
#         print("可训练的模块：")
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 print(f"  {name}")

    def train_step(self, data, optimizer):
        # import ipdb;ipdb.set_trace()
        losses, teacher_info, student_info = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, teacher_info = teacher_info,student_info = student_info , log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs 
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, extract_feat=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if extract_feat:
            return self.extract_feat(img[0])[0]
            
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)   
        
    def forward_test(self, imgs, img_metas, **kwargs):
        return self.model(**kwargs).forward_test(imgs, img_metas, **kwargs)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        return self.model(**kwargs).aforward_test(img, img_metas, **kwargs)

    def extract_feat(self, imgs):
        return self.model().extract_feat(imgs)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        return self.model(**kwargs).aforward_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.model(**kwargs).aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.model(**kwargs).simple_test(img, img_metas, **kwargs)

    async def async_simple_test(self, img, img_metas, **kwargs):
        return self.model(**kwargs).async_simple_test(img, img_metas, **kwargs)

    def show_result(self, *args, **kwargs):
        self.model().CLASSES = self.CLASSES
        return self.model().show_result(*args, **kwargs)
