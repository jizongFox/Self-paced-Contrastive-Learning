from copy import deepcopy

import torch
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.models import ema_updater
from torch import nn

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.meters import AverageValueMeter, MeterInterface
from semi_seg.hooks import meter_focus


class MeanTeacherTrainerHook(TrainerHook):

    def __init__(self, name: str, weight: float, model: nn.Module):
        super().__init__(name)
        self._weight = weight
        self._criterion = nn.MSELoss()
        self._updater = ema_updater()
        self._teacher_model = deepcopy(model)
        for p in self._teacher_model.parameters():
            p.detach_()

    def __call__(self):
        return _MeanTeacherEpocherHook(name=self._hook_name, weight=self._weight, criterion=self._criterion,
                                       teacher_model=self._teacher_model, updater=self._updater)

    @property
    def teacher_model(self):
        return self._teacher_model


class _MeanTeacherEpocherHook(EpocherHook):
    def __init__(self, name: str, weight: float, criterion, teacher_model, updater) -> None:
        super().__init__(name)
        self._weight = weight
        self._criterion = criterion
        self._teacher_model = teacher_model
        self._updater = updater

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("loss", AverageValueMeter())

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                 **kwargs):
        student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        teacher_unlabeled_prob = self._teacher_model(unlabeled_image)
        with FixRandomSeed(seed):
            teacher_unlabeled_prob_tf = torch.stack([affine_transformer(x) for x in teacher_unlabeled_prob], dim=0)
        loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
        self.meters["loss"].add(loss.item())
        self._updater(ema_model=self._teacher_model, student_model=self.epocher._model)  # noqa
        return self._weight * loss
