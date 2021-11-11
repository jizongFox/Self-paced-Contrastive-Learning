from functools import partial
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.schedulers.customized_scheduler import WeightScheduler
from torch import nn

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.contrast_loss3 import SelfPacedSupConLoss, SupConLoss1, switch_plt_backend
from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.writer import get_tb_writer
from semi_seg.arch.hook import SingleFeatureExtractor
from semi_seg.mi_estimator.base import decoder_names, encoder_names
from .utils import get_label, meter_focus


def get_n_point_coordinate(h, w, n):
    return [(x, y) for x, y in zip(np.random.choice(range(h), n, replace=False),
                                   np.random.choice(range(w), n, replace=False))]


def figure2board(tensor, name, criterion, writer, epocher):
    with switch_plt_backend("agg"):
        fig1 = plt.figure()
        plt.imshow(tensor.detach().cpu().numpy(), cmap="gray")
        plt.colorbar()
        dest = "/".join([criterion.__class__.__name__, name])
        writer.add_figure(tag=dest, figure=fig1, global_step=epocher._cur_epoch)  # noqa


class PScheduler(WeightScheduler):
    def __init__(self, max_epoch, begin_value=0.0, end_value=1.0, p=0.5):
        super().__init__()
        self.max_epoch = max_epoch
        self.begin_value = float(begin_value)
        self.end_value = float(end_value)
        self.epoch = 0
        self.p = p

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(self.epoch)

    def get_lr(self, cur_epoch):
        return self.begin_value + (self.end_value - self.begin_value) * np.power(
            cur_epoch / self.max_epoch, self.p
        )


class INFONCEHook(TrainerHook):

    @property
    def learnable_modules(self) -> List[nn.Module]:
        return [self._projector, ]

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0, spatial_size=None,
                 data_name: str, contrast_on: str) -> None:
        super().__init__(hook_name=name)
        assert feature_name in encoder_names + decoder_names, feature_name
        self._feature_name = feature_name
        self._weight = weight

        self._extractor = SingleFeatureExtractor(model, feature_name=feature_name)  # noqa
        input_dim = model.get_channel_dim(feature_name)
        if feature_name in encoder_names:
            spatial_size = spatial_size or (1, 1)
        else:
            spatial_size = spatial_size or (10, 10)
        self._projector = self.init_projector(input_dim=input_dim, spatial_size=spatial_size)
        self._criterion = self.init_criterion()
        self._label_generator = partial(get_label, contrast_on=contrast_on, data_name=data_name)
        self._learnable_models = (self._projector,)

    def __call__(self):
        if self.is_encoder:
            hook = _INFONCEEpochHook(
                name=self._hook_name, weight=self._weight, extractor=self._extractor, projector=self._projector,
                criterion=self._criterion, label_generator=self._label_generator
            )
            return hook
        return _INFONCEDenseHook(
            name=self._hook_name, weight=self._weight, extractor=self._extractor, projector=self._projector,
            criterion=self._criterion, label_generator=self._label_generator
        )

    def init_criterion(self) -> SupConLoss1:
        self._criterion = SupConLoss1()
        return self._criterion

    def init_projector(self, *, input_dim, spatial_size):
        projector = self.projector_class(input_dim=input_dim, hidden_dim=256, output_dim=256, head_type="mlp",
                                         normalize=True, spatial_size=spatial_size)
        return projector

    @property
    def projector_class(self):
        from contrastyou.projectors.heads import ProjectionHead, DenseProjectionHead
        if self.is_encoder:
            return ProjectionHead
        return DenseProjectionHead

    @property
    def is_encoder(self):
        return self._feature_name in encoder_names


class SelfPacedINFONCEHook(INFONCEHook):

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0, spatial_size=(1, 1),
                 data_name: str, contrast_on: str, mode="soft", p=0.5, begin_value=1e6, end_value=1e6,
                 correct_grad: bool = False, max_epoch: int) -> None:
        self._mode = mode
        self._p = float(p)
        self._begin_value = float(begin_value)
        self._end_value = float(end_value)
        self._max_epoch = int(max_epoch)
        self._correct_grad = correct_grad
        super().__init__(name=name, model=model, feature_name=feature_name, weight=weight, spatial_size=spatial_size,
                         data_name=data_name, contrast_on=contrast_on)

    def init_criterion(self) -> SelfPacedSupConLoss:
        self._scheduler = PScheduler(max_epoch=self._max_epoch, begin_value=self._begin_value,
                                     end_value=self._end_value, p=self._p)
        self._criterion = SelfPacedSupConLoss(weight_update=self._mode, correct_grad=self._correct_grad)
        return self._criterion

    def __call__(self):
        gamma = self._scheduler.value
        self._scheduler.step()
        self._criterion.set_gamma(gamma)
        hook = _SPINFONCEEpochHook(
            name=self._hook_name, weight=self._weight, extractor=self._extractor, projector=self._projector,
            criterion=self._criterion, label_generator=self._label_generator
        )
        return hook


class _INFONCEEpochHook(EpocherHook):

    def __init__(self, *, name: str, weight: float, extractor, projector,
                 criterion: Union[SupConLoss1, SelfPacedSupConLoss],
                 label_generator) -> None:
        super().__init__(name)
        self._extractor = extractor
        self._extractor.bind()
        self._weight = weight
        self._projector = projector
        self._criterion = criterion
        self._label_generator = label_generator
        self._n = 0

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        meters = super().configure_meters(meters)
        meters.register_meter("loss", AverageValueMeter())
        return meters

    def before_forward_pass(self, **kwargs):
        self._extractor.clear()
        self._extractor.set_enable(True)

    def after_forward_pass(self, **kwargs):
        self._extractor.set_enable(False)

    @meter_focus
    def __call__(self, *, affine_transformer, seed, unlabeled_tf_logits, unlabeled_logits_tf, partition_group,
                 label_group, **kwargs):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self._extractor.feature()[-n_unl * 2:]
        unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        with FixRandomSeed(seed):
            unlabeled_features_tf = torch.stack([affine_transformer(x) for x in unlabeled_features], dim=0)
        norm_features_tf, norm_tf_features = torch.chunk(
            self._projector(torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0)), 2)
        labels = self._label_generator(partition_group=partition_group, label_group=label_group)
        loss = self._criterion(norm_features_tf, norm_tf_features, target=labels)
        self.meters["loss"].add(loss.item())

        sim_exp = self._criterion.sim_exp
        sim_logits = self._criterion.sim_logits
        pos_mask = self._criterion.pos_mask
        if self._n == 0:
            writer = get_tb_writer()
            figure2board(pos_mask, "mask", self._criterion, writer, self.epocher)
            figure2board(sim_exp, "sim_exp", self._criterion, writer, self.epocher)
            figure2board(sim_logits, "sim_logits", self._criterion, writer, self.epocher)

        self._n += 1
        return loss * self._weight

    def close(self):
        self._extractor.remove()


class _INFONCEDenseHook(_INFONCEEpochHook):
    @meter_focus
    def __call__(self, *, affine_transformer, seed, unlabeled_tf_logits, unlabeled_logits_tf, partition_group,
                 label_group, **kwargs):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self._extractor.feature()[-n_unl * 2:]
        unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        with FixRandomSeed(seed):
            unlabeled_features_tf = torch.stack([affine_transformer(x) for x in unlabeled_features], dim=0)
        norm_features_tf, norm_tf_features = torch.chunk(
            self._projector(torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0)), 2)
        with FixRandomSeed(seed):
            norm_features_tf_selected = self.region_extractor(norm_features_tf, point_nums=5)
        with FixRandomSeed(seed):
            norm_tf_features_selected = self.region_extractor(norm_tf_features, point_nums=5)

        labels = list(range(norm_features_tf_selected.shape[0]))
        loss = self._criterion(norm_features_tf_selected, norm_tf_features_selected, target=labels)
        self.meters["loss"].add(loss.item())

        sim_exp = self._criterion.sim_exp
        sim_logits = self._criterion.sim_logits
        pos_mask = self._criterion.pos_mask
        if self._n == 0:
            writer = get_tb_writer()
            figure2board(pos_mask, "mask", self._criterion, writer, self.epocher)
            figure2board(sim_exp, "sim_exp", self._criterion, writer, self.epocher)
            figure2board(sim_logits, "sim_logits", self._criterion, writer, self.epocher)

        self._n += 1
        return loss * self._weight

    @staticmethod
    def region_extractor(normalize_features, point_nums=5):
        def get_feature_selected(feature_map, n_point_coordinate):
            return torch.stack([feature_map[:, n[0], n[1]] for n in n_point_coordinate], dim=0)

        h, w = normalize_features.shape[2:]
        return torch.cat(
            [get_feature_selected(single_feature, get_n_point_coordinate(n=point_nums, h=h, w=w)) for single_feature in
             normalize_features], dim=0)


class _SPINFONCEEpochHook(_INFONCEEpochHook):
    _criterion: SelfPacedSupConLoss

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        meters = super().configure_meters(meters)
        meters.register_meter("sp_weight", AverageValueMeter())
        meters.register_meter("age_param", AverageValueMeter())
        return meters

    @meter_focus
    def __call__(self, *, affine_transformer, seed, unlabeled_tf_logits, unlabeled_logits_tf, partition_group,
                 label_group, **kwargs):
        loss = super().__call__(affine_transformer=affine_transformer, seed=seed,
                                unlabeled_tf_logits=unlabeled_tf_logits, unlabeled_logits_tf=unlabeled_logits_tf,
                                partition_group=partition_group, label_group=label_group, **kwargs)
        self.meters["sp_weight"].add(self._criterion.downgrade_ratio)
        self.meters["age_param"].add(self._criterion.age_param)

        sp_mask = self._criterion.sp_mask
        if self._n == 1:
            writer = get_tb_writer()
            figure2board(sp_mask, "sp_mask", self._criterion, writer, self.epocher)

        return loss
