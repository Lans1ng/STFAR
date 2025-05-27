from mmdet.datasets import DATASETS
from .cityscapes_with_idx import CityscapesDatasetWithIdx
from .coco_subset import CocoDatasetSubset
import numpy as np


@DATASETS.register_module()
class CityscapesDataseSubset(CityscapesDatasetWithIdx):
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


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 outfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in Cityscapes/COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            outfile_prefix (str | None): The prefix of output file. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with COCO protocol, it would be the
                prefix of output json file. For example, the metric is 'bbox'
                and 'segm', then json files would be "a/b/prefix.bbox.json" and
                "a/b/prefix.segm.json".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output txt/png files. The output files would be
                png images under folder "a/b/prefix/xxx/" and the file name of
                images would be written into a txt file
                "a/b/prefix/xxx_pred.txt", where "xxx" is the video name of
                cityscapes. If not specified, a temp file will be created.
                Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: COCO style evaluation metric or cityscapes mAP \
                and AP@50.
        """
        
        eval_results = dict()

        metrics = metric.copy() if isinstance(metric, list) else [metric]

        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, outfile_prefix, logger))
            metrics.remove('cityscapes')

        # left metrics are all coco metric
        if len(metrics) > 0:
            # create CocoDataset with CityscapesDataset annotation

            self_coco = CocoDatasetSubset(ann_file = self.ann_file, pipeline = self.pipeline.transforms,
                                    classes = None, data_root = self.data_root, img_prefix = self.img_prefix,
                                    seg_prefix = self.seg_prefix, proposal_file = self.proposal_file,
                                    test_mode = self.test_mode, filter_empty_gt = self.filter_empty_gt,
                                    with_idx=False,indices =self.indices)
            # TODO: remove this in the future
            # reload annotations of correct class
            self_coco.CLASSES = self.CLASSES
            self_coco.data_infos = self_coco.load_annotations(self.ann_file)
            self_coco.img_ids = np.array(self_coco.img_ids)[self.indices]
            eval_results.update(
                self_coco.evaluate(results=results, metric=metrics, logger=logger, jsonfile_prefix=outfile_prefix,
                                   classwise=True, proposal_nums=proposal_nums, iou_thrs=iou_thrs))
        

        return eval_results