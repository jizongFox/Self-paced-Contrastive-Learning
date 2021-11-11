from typing import List

import torch
from deepclustering2.decorator import FixRandomSeed
from torch import nn

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.meters import AverageValueMeter
from semi_seg.arch.hook import SingleFeatureExtractor
from semi_seg.hooks.utils import meter_focus
from semi_seg.mi_estimator.base import decoder_names, encoder_names


class DiscreteMITrainHook(TrainerHook):

    @property
    def learnable_modules(self) -> List[nn.Module]:
        return [self._projector, ]

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0, num_clusters=20,
                 num_subheads=5, padding=None) -> None:
        super().__init__(hook_name=name)
        assert feature_name in encoder_names + decoder_names, feature_name
        self._feature_name = feature_name
        self._weight = weight

        self._extractor = SingleFeatureExtractor(model, feature_name=feature_name)  # noqa

        input_dim = model.get_channel_dim(feature_name)
        self._projector = self.init_projector(input_dim=input_dim, num_clusters=num_clusters, num_subheads=num_subheads)
        self._criterion = self.init_criterion(padding=padding)

    def __call__(self):
        return _DiscreteMIEpochHook(name=self._hook_name, weight=self._weight, extractor=self._extractor,
                                    projector=self._projector, criterion=self._criterion)

    def init_projector(self, *, input_dim, num_clusters, num_subheads=5):
        projector = self.projector_class(input_dim=input_dim, num_clusters=num_clusters,
                                         num_subheads=num_subheads, head_type="linear", T=1, normalize=False)
        return projector

    def init_criterion(self, padding: int = None):
        if self._feature_name in encoder_names:
            return self._init_criterion()
        return self._init_dense_criterion(padding=padding or 0)

    def _init_dense_criterion(self, padding: int = 0):
        criterion = self.criterion_class(padding=padding)
        return criterion

    def _init_criterion(self):
        criterion = self.criterion_class()

        def criterion_wrapper(*args, **kwargs):
            return criterion(*args, **kwargs)[0]

        return criterion_wrapper

    @property
    def projector_class(self):
        from contrastyou.projectors.heads import DenseClusterHead, ClusterHead
        if self._feature_name in encoder_names:
            return ClusterHead
        return DenseClusterHead

    @property
    def criterion_class(self):
        from contrastyou.losses.iic_loss import IIDLoss, IIDSegmentationLoss
        if self._feature_name in encoder_names:
            return IIDLoss
        return IIDSegmentationLoss


class _DiscreteMIEpochHook(EpocherHook):

    def __init__(self, *, name: str, weight: float, extractor, projector, criterion) -> None:
        super().__init__(name)
        self._extractor = extractor
        self._extractor.bind()
        self._weight = weight
        self._projector = projector
        self._criterion = criterion

    @meter_focus
    def configure_meters(self, meters):
        meters.register_meter("mi", AverageValueMeter())

    def before_forward_pass(self, **kwargs):
        self._extractor.clear()
        self._extractor.set_enable(True)

    def after_forward_pass(self, **kwargs):
        self._extractor.set_enable(False)

    @meter_focus
    def __call__(self, *, unlabeled_image, unlabeled_image_tf, affine_transformer, seed, **kwargs):
        n_unl = len(unlabeled_image)
        feature_ = self._extractor.feature()[-n_unl * 2:]
        proj_feature, proj_tf_feature = torch.chunk(feature_, 2, dim=0)
        assert proj_feature.shape == proj_tf_feature.shape
        with FixRandomSeed(seed):
            proj_feature_tf = torch.stack([affine_transformer(x) for x in proj_feature], dim=0)

        prob1, prob2 = list(
            zip(*[torch.chunk(x, 2, 0) for x in self._projector(
                torch.cat([proj_feature_tf, proj_tf_feature], dim=0)
            )])
        )
        loss = sum([self._criterion(x1, x2) for x1, x2 in zip(prob1, prob2)]) / len(prob1)
        self.meters["mi"].add(loss.item())
        return loss * self._weight

    def close(self):
        self._extractor.remove()
