import random
from typing import Callable, Iterable
from typing import Union, List

import torch
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter, MultipleAverageValueMeter
from deepclustering2.models import ema_updater as EMA_Updater
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.tqdm import tqdm, item2str
from deepclustering2.type import T_loss
from deepclustering2.utils import simplex
from loguru import logger
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from contrastyou.utils import weighted_average_iter, get_dataset
from semi_seg.arch.hook import FeatureExtractor
from .helper import unl_extractor, preprocess_input_with_twice_transformation


class _FeatureExtractorMixin:
    forward_pass: Callable

    @staticmethod
    def __refactorize(feature_position, feature_importance):
        assert isinstance(feature_position, list) and isinstance(feature_position[0], str), feature_position
        assert isinstance(feature_importance, list) and isinstance(feature_importance[0],
                                                                   (int, float)), feature_importance

        assert len(feature_position) == len(feature_importance), \
            (len(feature_position), len(feature_importance))
        return feature_position, feature_importance

    def __init__(self, feature_position: Union[List[str], str], feature_importance: Union[float, List[float]], *args,
                 **kwargs):
        super(_FeatureExtractorMixin, self).__init__(*args, **kwargs)
        feature_position, feature_importance = self.__refactorize(feature_position, feature_importance)
        logger.debug("Initializing {} with features and weights: {}", self.__class__.__name__,
                     item2str({k: v for k, v in zip(feature_position, feature_importance)}))

        self._feature_position = feature_position
        self._feature_importance = feature_importance

    def run(self, *args, **kwargs):
        with FeatureExtractor(self._model, self._feature_position) as self._fextractor:  # noqa
            logger.debug(f"create feature extractor for {', '.join(self._feature_position)} ")
            return super(_FeatureExtractorMixin, self).run(*args, **kwargs)  # noqa

    def forward_pass(self, *args, **kwargs):
        self._fextractor.clear()
        with self._fextractor.enable_register():
            return super(_FeatureExtractorMixin, self).forward_pass(*args, **kwargs)  # noqa


class _MIMixin(_FeatureExtractorMixin):
    meters: MeterInterface
    _affine_transformer: Callable[[Tensor], Tensor]
    _feature_importance: List[float]
    _fextractor: FeatureExtractor
    _feature_position: List[str]

    def _init(self, *, mi_estimator_array: Iterable[Callable[[Tensor, Tensor], Tensor]], **kwargs):
        super(_MIMixin, self)._init(**kwargs)  # noqa
        self._mi_estimator_array = mi_estimator_array

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(_MIMixin, self)._configure_meters(meters)  # noqa
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

    def _mi_regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, ):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        def calculate_iic(unlabeled_features, mi_estimator: Callable[[Tensor, Tensor], Tensor]):
            unlabeled_features, unlabeled_tf_features = torch.chunk(unlabeled_features, 2, dim=0)

            with FixRandomSeed(seed):
                unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features], dim=0)
            assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
                (unlabeled_tf_features.shape, unlabeled_tf_features.shape)

            loss = mi_estimator(unlabeled_tf_features, unlabeled_features_tf)
            return loss

        iic_losses = [calculate_iic(f, mi) for f, mi in zip(
            unl_extractor(self._fextractor, n_uls=n_uls), self._mi_estimator_array
        )]

        reg_loss = weighted_average_iter(iic_losses, self._feature_importance)

        with torch.no_grad():
            self.meters["mi"].add(-reg_loss.item())
            self.meters["individual_mis"].add(**dict(zip(
                self._feature_position,
                [-x.item() for x in iic_losses]
            )))
        return reg_loss


# mean teacher mixin
class _MeanTeacherMixin:
    _affine_transformer: Callable[[Tensor], Tensor]
    _model: nn.Module

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__mt_initialized__ = False
        self.__mt_updated__ = False

    def _init(self, *, teacher_model: nn.Module, reg_criterion: T_loss,
              ema_updater: EMA_Updater, **kwargs):
        super(_MeanTeacherMixin, self)._init(**kwargs)  # noqa
        self._reg_criterion = reg_criterion
        self._teacher_model = teacher_model
        self._ema_updater = ema_updater
        self._teacher_model.train()
        self.__mt_initialized__ = True

    def run(self, *args, **kwargs):
        if not self.__mt_initialized__:
            raise RuntimeError("MTMixin should be initialized first")
        result = super(_MeanTeacherMixin, self).run(*args, **kwargs)  # noqa
        if not self.__mt_updated__:
            raise RuntimeError("You may forget updating the mean teacher")
        return result

    def _mt_regularization(
        self,
        *,
        unlabeled_tf_logits: Tensor,
        seed: int,
        unlabeled_image: Tensor,
    ):
        with torch.no_grad():
            teacher_unlabeled_logit = self._teacher_model(unlabeled_image)
        with FixRandomSeed(seed):
            teacher_unlabeled_logit_tf = torch.stack(
                [self._affine_transformer(x) for x in teacher_unlabeled_logit], dim=0)

        # compare teacher_unlabeled_logit_tf and student unlabeled_tf_logits
        if simplex(teacher_unlabeled_logit):
            raise RuntimeError("`teacher_unlabeled_logit` should be logits, instead of simplex.")
        reg_loss = self._reg_criterion(unlabeled_tf_logits.softmax(1), teacher_unlabeled_logit_tf.softmax(1).detach())
        return reg_loss

    def update_meanteacher(self, teacher_model, student_model):
        self.__mt_updated__ = True
        self._ema_updater(teacher_model, student_model)


class _ConsistencyMixin:
    meters: MeterInterface

    def _init(self, *, reg_criterion: T_loss, **kwargs):  # noqa
        super(_ConsistencyMixin, self)._init(**kwargs)  # noqa
        self._reg_criterion = reg_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(_ConsistencyMixin, self)._configure_meters(meters)  # noqa
        meters.register_meter("consistency", AverageValueMeter())
        return meters

    def _consistency_regularization(self, *, unlabeled_tf_logits: Tensor,
                                    unlabeled_logits_tf: Tensor):
        if simplex(unlabeled_logits_tf):
            raise RuntimeError("unlabeled_logits_tf should not be simplex.")

        reg_loss = self._reg_criterion(
            unlabeled_tf_logits.softmax(1),
            unlabeled_logits_tf.softmax(1).detach()
        )
        self.meters["consistency"].add(reg_loss.item())
        return reg_loss


# ======== base pretrain epocher mixin ================
class _PretrainEpocherMixin:
    """
    PretrainEpocher makes all images goes to regularization, permitting to use the other classes to create more pretrain
    models
    """
    meters: MeterInterface
    _model: nn.Module
    _optimizer: Optimizer
    _indicator: tqdm
    _labeled_loader: DataLoader
    _unlabeled_loader: DataLoader
    _unzip_data: Callable[..., torch.device]
    _device: torch.device
    _affine_transformer: Callable[[Tensor], Tensor]
    on_master: Callable[[], bool]
    regularization: Callable[..., Tensor]
    forward_pass: Callable

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meter = super()._configure_meters(meters)  # noqa
        meter.delete_meters(["sup_loss", "sup_dice", "reg_weight"])
        return meter

    def _init(self, *, chain_dataloader, monitor_dataloader, **kwargs):
        # extend the interface for original class with chain_dataloader
        super()._init(**kwargs)  # noqa
        self._chain_dataloader = chain_dataloader
        self._monitor_dataloader = monitor_dataloader

    def _assertion(self):
        labeled_set = get_dataset(self._labeled_loader)
        labeled_transform = labeled_set.transforms
        assert labeled_transform._total_freedom  # noqa

        if self._unlabeled_loader is not None:
            unlabeled_set = get_dataset(self._unlabeled_loader)
            unlabeled_transform = unlabeled_set.transforms
            assert unlabeled_transform._total_freedom  # noqa

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        assert self._model.training, self._model.training
        return self._run_pretrain(*args, **kwargs)

    def _run_pretrain(self, *args, **kwargs):
        for self.cur_batch_num, data in zip(self._indicator, self._chain_dataloader):
            seed = random.randint(0, int(1e7))
            (unlabeled_image, unlabeled_image_tf), _, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(data, self._device)

            unlabel_logits, unlabel_tf_logits = self.forward_pass(
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            # regularized part
            reg_loss = self.regularization(
                unlabeled_tf_logits=unlabel_tf_logits,
                unlabeled_logits_tf=unlabel_tf_logits,
                seed=seed,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                unlabeled_filename=unlabeled_filename,
            )
            total_loss = reg_loss
            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            if self.on_master():
                with torch.no_grad():
                    self.meters["reg_loss"].add(reg_loss.item())
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)

        report_dict = self.meters.tracking_status(final=True)
        return report_dict

    def _forward_pass(self, unlabeled_image, unlabeled_image_tf):
        n_l, n_unl = 0, len(unlabeled_image)
        # hightlight: this is only for training for encoder.
        predict_logits = self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0), until="Conv5")

        unlabel_logits, unlabel_tf_logits = torch.split(predict_logits, [n_unl, n_unl], dim=0)
        return unlabel_logits, unlabel_tf_logits

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return (image, image_ct), None, filename, partition, group


class _PretrainMonitorEpocherMxin(_PretrainEpocherMixin):
    @torch.no_grad()
    def _monitor_pretrain(self):
        for self.cur_batch_num, data in enumerate(self._monitor_dataloader):
            seed = random.randint(0, int(1e7))
            unlabeled_image, unlabeled_target, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(data, self._device)

            with FixRandomSeed(seed):
                unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image], dim=0)
            assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                (unlabeled_image_tf.shape, unlabeled_image.shape)

            unlabel_logits, unlabel_tf_logits = self.forward_pass(
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            with FixRandomSeed(seed):
                unlabel_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabel_logits], dim=0)

            assert unlabel_logits_tf.shape == unlabel_tf_logits.shape, (
                unlabel_logits_tf.shape, unlabel_tf_logits.shape)

            # regularized part
            reg_loss = self.regularization(
                unlabeled_tf_logits=unlabel_tf_logits,
                unlabeled_logits_tf=unlabel_logits_tf,
                seed=seed,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                unlabeled_filename=unlabeled_filename,
            )
            if self.cur_batch_num > 30:
                break

        report_dict = self.meters.tracking_status(final=True)
        return report_dict

    def monitor_pretrain(self):
        previous_value = self._affine_transformer._threshold  # noqa
        self._affine_transformer._threshold = 0
        result = self._monitor_pretrain()
        self._affine_transformer._threshold = previous_value  # noqa
        return result

    _run = monitor_pretrain
