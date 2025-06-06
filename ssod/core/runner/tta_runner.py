
import os.path as osp
import platform
import shutil
import time
import warnings

from mmcv.runner import get_host_info
import torch
import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint

from mmcv.parallel import is_module_wrapper
from mmcv.runner import IterBasedRunner, EpochBasedRunner


@RUNNERS.register_module()
class TTAEpochBasedRunner(EpochBasedRunner):

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
            # result = self.model(return_loss=False, rescale=True, **data_batch)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            # student_info = self.outputs['student_info'].keys()

            # self.outputs['teacher_info'].keys()
            #(['backbone_feature_res', 'backbone_feature', 'proposals', 'det_bboxes', 
            # 'det_labels', 'transform_matrix', 'img_metas', 'pseudo_bboxes'])

            # print('det_bboxes',det_bboxes)
            # print('det_labels',det_labels)
            # print('img_metas',img_metas)
            self.call_hook('after_train_iter')
            # if i == 1:
            #     break
            # self.run_iter(data_batch, train_mode=False, **kwargs)
            # print('val_out',self.outputs)
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
        
    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break
        
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
    # def save_checkpoint(self,
    #                     out_dir,
    #                     filename_tmpl='iter_{}.pth',
    #                     meta=None,
    #                     save_optimizer=True,
    #                     create_symlink=True):
    #     if meta is None:
    #         meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
    #     elif isinstance(meta, dict):
    #         meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
    #     else:
    #         raise TypeError(
    #             f'meta should be a dict or None, but got {type(meta)}')
    #     if self.meta is not None:
    #         meta.update(self.meta)

    #     filename = filename_tmpl.format(self.iter + 1)
    #     filepath = osp.join(out_dir, filename)
    #     optimizer = self.optimizer if save_optimizer else None
    #     save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
    #     filepath_ema = filepath[:-4] + '_ema.pth'
    #     if is_module_wrapper(self.model):
    #         use_ema = hasattr(self.model.module, 'ema_model') and self.model.module.ema_model is not None
    #         if use_ema:
    #             save_checkpoint(self.model.module.ema_model, filepath_ema, optimizer=optimizer, meta=meta)
    #     else:
    #         use_ema = hasattr(self.model, 'ema_model') and self.model.ema_model is not None
    #         if use_ema:
    #             save_checkpoint(self.model.ema_model, filepath_ema, optimizer=optimizer, meta=meta)
    #     # in some environments, `os.symlink` is not supported, you may need to
    #     # set `create_symlink` to False
    #     if create_symlink:
    #         dst_file = osp.join(out_dir, 'latest.pth')
    #         if platform.system() != 'Windows':
    #             mmcv.symlink(filename, dst_file)
    #         else:
    #             shutil.copy(filepath, dst_file)