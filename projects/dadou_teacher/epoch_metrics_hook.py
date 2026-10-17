"""Write one machine-readable training record per epoch."""

import csv
import os
from collections import OrderedDict, defaultdict
from numbers import Number

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class EpochMetricsCsvHook(Hook):
    """Aggregate batch losses and merge validation metrics into one CSV.

    The file is rewritten after every train/validation epoch, so an interrupted
    run still leaves usable loss-vs-epoch data for plotting and reporting.
    """

    priority = 'NORMAL'

    def __init__(self, filename='epoch_metrics.csv'):
        self.filename = filename
        self.path = None
        self.rows = OrderedDict()
        self.sums = defaultdict(float)
        self.weights = defaultdict(float)

    def before_run(self, runner):
        if not is_main_process():
            return
        self.path = os.path.join(runner.work_dir, self.filename)
        if os.path.isfile(self.path):
            with open(self.path, encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    if row.get('epoch'):
                        self.rows[int(row['epoch'])] = dict(row)

    def before_train_epoch(self, runner):
        self.sums.clear()
        self.weights.clear()

    @staticmethod
    def _batch_size(data_batch):
        if not isinstance(data_batch, dict):
            return 1
        samples = data_batch.get('data_samples')
        if samples is not None:
            return len(samples)
        inputs = data_batch.get('inputs')
        if isinstance(inputs, (list, tuple)):
            return len(inputs)
        if isinstance(inputs, torch.Tensor) and inputs.ndim > 0:
            return int(inputs.shape[0])
        return 1

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        if isinstance(value, Number):
            return float(value)
        return None

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        if not is_main_process() or not isinstance(outputs, dict):
            return
        weight = self._batch_size(data_batch)
        for name, value in outputs.items():
            scalar = self._scalar(value)
            if scalar is None:
                continue
            self.sums[name] += scalar * weight
            self.weights[name] += weight

    def after_train_epoch(self, runner):
        if not is_main_process():
            return
        epoch = int(runner.epoch) + 1
        row = {'epoch': epoch, 'global_iter': int(runner.iter) + 1}
        for name in self.sums:
            row[f'train_{name}'] = self.sums[name] / self.weights[name]

        optimizer = runner.optim_wrapper.optimizer
        for index, group in enumerate(optimizer.param_groups):
            row[f'learning_rate_group_{index}'] = float(group['lr'])
        self.rows[epoch] = row
        self._write()

    def after_val_epoch(self, runner, metrics=None):
        if not is_main_process():
            return
        epoch = int(runner.epoch)
        row = self.rows.setdefault(epoch, {'epoch': epoch})
        for name, value in (metrics or {}).items():
            scalar = self._scalar(value)
            if scalar is not None:
                row[f'val_{name}'] = scalar
        self._write()

    def _write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fieldnames = ['epoch']
        all_keys = set().union(*(row.keys() for row in self.rows.values()))
        fieldnames.extend(sorted(all_keys - {'epoch'}))
        with open(self.path, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for epoch in sorted(self.rows):
                writer.writerow(self.rows[epoch])
