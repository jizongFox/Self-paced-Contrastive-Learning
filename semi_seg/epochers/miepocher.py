from typing import Callable, Iterable

from deepclustering2.meters2 import AverageValueMeter, MeterInterface
from deepclustering2.type import T_loss
from torch import Tensor

from ._mixins import _ConsistencyMixin, _MIMixin
from .base import TrainEpocher


class ConsistencyTrainEpocher(_ConsistencyMixin, TrainEpocher, ):

    # noinspection Mypy
    def _regularization(
        self,
        *,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, **kwargs
    ):
        reg_loss = self._consistency_regularization(unlabeled_tf_logits=unlabeled_tf_logits,
                                                    unlabeled_logits_tf=unlabeled_logits_tf)
        return reg_loss


class MITrainEpocher(_MIMixin, TrainEpocher, ):

    def _regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, **kwargs):
        reg_loss = self._mi_regularization(unlabeled_logits_tf=unlabeled_logits_tf,
                                           unlabeled_tf_logits=unlabeled_tf_logits,
                                           seed=seed)
        return reg_loss


class ConsistencyMIEpocher(_ConsistencyMixin, _MIMixin, TrainEpocher,
                           ):

    def _init(self, *, mi_weight: float, consistency_weight: float,  # noqa
              mi_estimator_array: Iterable[Callable[[Tensor, Tensor], Tensor]], reg_criterion: T_loss,  # noqa
              enforce_matching=False, **kwargs):
        super()._init(reg_weight=1.0, mi_estimator_array=mi_estimator_array, enforce_matching=enforce_matching,
                      reg_criterion=reg_criterion, **kwargs)
        self._mi_weight = mi_weight  # noqa
        self._cons_weight = consistency_weight  # noqa
        self._reg_criterion = reg_criterion  # noqa
        assert self._reg_weight == 1.0, self._reg_weight

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("consistency", AverageValueMeter())
        meters.register_meter("mi_weight", AverageValueMeter())
        meters.register_meter("cons_weight", AverageValueMeter())
        return meters

    def _regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, **kwargs):
        self.meters["mi_weight"].add(self._mi_weight)
        self.meters["cons_weight"].add(self._cons_weight)
        iic_loss = MITrainEpocher._regularization(
            self,
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed
        )
        cons_loss = ConsistencyTrainEpocher._regularization(
            self,  # noqa
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed,
        )
        return self._cons_weight * cons_loss + self._mi_weight * iic_loss
