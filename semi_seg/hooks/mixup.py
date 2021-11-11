from contextlib import nullcontext
from functools import lru_cache
from typing import Union

import numpy as np
import torch
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats
from deepclustering2.loss import KL_div
from deepclustering2.utils import class2one_hot
from loguru import logger
from torch import Tensor

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.utils import fix_all_seed_within_context
from semi_seg.hooks import meter_focus


def mixup_data(x, y, *, alpha=1.0, device: Union[str, torch.device]):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y


class MixUpHook(TrainerHook):
    def __init__(self, *, hook_name: str, weight: float, enable_bn=True):
        super().__init__(hook_name)
        self._weight = weight
        self._enable_bn = enable_bn
        logger.debug(f"created {self.__class__.__name__} with weight: {self._weight} and enable_bn: {self._enable_bn}")

    def __call__(self, **kwargs):
        return _MixUpEpocherHook(name="mix_reg", weight=self._weight, criterion=KL_div(verbose=False),
                                 enable_bn=self._enable_bn)


class _MixUpEpocherHook(EpocherHook):
    def __init__(self, *, name: str, weight: float, alpha: float = 1.0, criterion, enable_bn=True) -> None:
        super().__init__(name)
        self._weight = weight
        self._alpha = alpha
        self._criterion = criterion
        self._enable_bn = enable_bn

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        meters = super(_MixUpEpocherHook, self).configure_meters(meters)
        meters.register_meter("mixup_ls", AverageValueMeter())
        return meters

    @meter_focus
    def __call__(self, *, labeled_image: Tensor,
                 labeled_image_tf: Tensor,
                 labeled_target: Tensor,
                 labeled_target_tf: Tensor, seed: int, **kwargs):
        labeled_target_oh = class2one_hot(labeled_target, C=self.num_classes)
        labeled_target_tf_oh = class2one_hot(labeled_target_tf, C=self.num_classes)

        with fix_all_seed_within_context(seed):
            mixed_image, mixed_target, = mixup_data(x=torch.cat([labeled_image, labeled_image_tf], dim=0),
                                                    y=torch.cat([labeled_target_oh, labeled_target_tf_oh], dim=0),
                                                    alpha=1, device=self.device)
        with self.bn_context_manger(self._model):
            mixed_pred = self._model(mixed_image)
        reg_loss = self._criterion(mixed_pred.softmax(1), mixed_target.squeeze())
        self.meters["mixup_ls"].add(reg_loss.item())
        return reg_loss * self._weight

    @property
    def _model(self):
        return self.epocher._model  # noqa

    @property
    def device(self):
        return self.epocher.device

    @property
    def num_classes(self):
        return self.epocher.num_classes

    @property
    @lru_cache()
    def bn_context_manger(self):
        return _disable_tracking_bn_stats if not self._enable_bn else nullcontext
