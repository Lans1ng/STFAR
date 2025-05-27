from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n
import torch
from copy import deepcopy

@HOOKS.register_module()
class StoRestore(Hook):
    def __init__(
        self,
        source_model,
        rst=0.01,
    ):
        self.rst = rst
        source_model = torch.load(source_model)
        if is_module_wrapper(source_model):
            source_model = source_model.module 
        self.source_model_state = deepcopy(source_model['state_dict'])
        # print(self.source_model_state.keys())
    def before_run(self, runner):
        target_model = runner.model
        if is_module_wrapper(target_model):
            target_model = target_model.module   
        self.target_model = target_model.student
    # def before_train_iter(self, runner):
    #     print('use_SR')
    def after_train_iter(self, runner):
        # print('use_SR')
        for nm, m  in self.target_model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape)<self.rst).float().cuda().to(p.device) 
                    with torch.no_grad():
                        p.data = self.source_model_state[f"student.{nm}.{npp}"].to(p.device) * mask + p * (1.-mask)

