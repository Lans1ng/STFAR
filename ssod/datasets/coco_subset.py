from mmdet.datasets import DATASETS
from .coco_with_idx import CocoDatasetWithIdx
import numpy as np


@DATASETS.register_module()
class CocoDatasetSubset(CocoDatasetWithIdx):
    def __init__(
        self,
        indices,
        with_idx=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.indices = indices
        self.with_idx = with_idx
        self.img_ids = np.array(self.img_ids)[self.indices]
        return
    
    def __getitem__(self, idx):
        data = super().__getitem__(self.indices[idx])
        if self.with_idx is False:
            if type(data) == dict:
                if 'idx' in data:
                    del data['idx']
            else:
                for i in data:
                    if 'idx' in data:
                        del i['idx']
        return data

    def __len__(self):
        return len(self.indices)