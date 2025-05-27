from mmdet.datasets import DATASETS, CityscapesDataset
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.custom import CustomDataset
from .coco_with_idx import CocoDatasetWithIdx


@DATASETS.register_module()
class CityscapesDatasetWithIdx(CityscapesDataset):
    def __init__(
        self,
        with_idx=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.with_idx=with_idx
        return
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.with_idx:
            if type(data) == dict:
                data['idx'] = idx
            else:
                for i in data:
                    i['idx'] = idx
        return data
    

