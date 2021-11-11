import math
from abc import abstractmethod
from functools import lru_cache, partial
from itertools import cycle
from typing import Callable, Iterable, List

import torch
from deepclustering2.configparser._utils import get_config  # noqa
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats as disable_bn  # noqa
from deepclustering2.loss import Entropy
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, MeterInterface, MultipleAverageValueMeter
from deepclustering2.models import ema_updater as EMA_Updater
from deepclustering2.schedulers.customized_scheduler import RampScheduler
from deepclustering2.type import T_loss
from deepclustering2.writer.SummaryWriter import get_tb_writer
from loguru import logger
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from contrastyou.losses.contrast_loss2 import is_normalized, SupConLoss1
from contrastyou.utils.utils import ntuple
from contrastyou.projectors.heads import ProjectionHead
from contrastyou.projectors.nn import Normalize
from contrastyou.utils import average_iter, weighted_average_iter
from semi_seg.arch.hook import FeatureExtractor
from semi_seg.utils import ContrastiveProjectorWrapper
from .helper import unl_extractor
from ._mixins import _FeatureExtractorMixin, _MeanTeacherMixin
from .base import TrainEpocher
from .miepocher import MITrainEpocher, ConsistencyTrainEpocher


class MeanTeacherEpocher(_MeanTeacherMixin, TrainEpocher):

    def _regularization(
        self,
        *,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed: int,
        unlabeled_image: Tensor,
        unlabeled_image_tf: Tensor, **kwargs
    ):
        mt_reg = self._mt_regularization(
            unlabeled_tf_logits=unlabeled_tf_logits,
            seed=seed, unlabeled_image=unlabeled_image
        )
        self.update_meanteacher(self._teacher_model, self._model)
        return mt_reg


class UCMeanTeacherEpocher(MeanTeacherEpocher, ):

    def _init(self, *, reg_weight: float, teacher_model: nn.Module, reg_criterion: T_loss,  # noqa
              ema_updater: EMA_Updater, threshold: RampScheduler = None, **kwargs):  # noqa
        super()._init(reg_weight=reg_weight, teacher_model=teacher_model, reg_criterion=reg_criterion,
                      ema_updater=ema_updater, **kwargs)
        assert isinstance(threshold, RampScheduler), threshold
        self._threshold: RampScheduler = threshold
        self._entropy_loss = Entropy(reduction="none")

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(UCMeanTeacherEpocher, self)._configure_meters(meters)
        meters.register_meter("uc_weight", AverageValueMeter())
        meters.register_meter("uc_ratio", AverageValueMeter())
        return meters

    def _regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int,
                        unlabeled_image: Tensor, unlabeled_image_tf: Tensor, **kwargs):
        # override completely the reguarization function.
        @torch.no_grad()
        def get_teacher_pred_with_tf(uimage, noise=None):
            if noise is not None:
                uimage += noise
            teacher_unlabeled_logit = self._teacher_model(uimage)
            with FixRandomSeed(seed):
                teacher_unlabeled_logit_tf = torch.stack(
                    [self._affine_transformer(x) for x in teacher_unlabeled_logit], dim=0)
            return teacher_unlabeled_logit_tf

        # compare teacher_unlabeled_logit_tf and student unlabeled_tf_logits
        self._reg_criterion.reduction = "none"  # here the self._reg_criterion should be nn.MSELoss or KLDiv()
        teacher_unlabeled_logit_tf = get_teacher_pred_with_tf(unlabeled_image)
        reg_loss = self._reg_criterion(unlabeled_tf_logits.softmax(1), teacher_unlabeled_logit_tf.softmax(1).detach())

        # uncertainty:
        with disable_bn(self._teacher_model):
            uncertainty_predictions = [
                get_teacher_pred_with_tf(unlabeled_image, 0.05 * torch.randn_like(unlabeled_image)) for _ in range(8)
            ]

        average_prediction = average_iter(uncertainty_predictions)

        entropy = self._entropy_loss(average_prediction.softmax(1)) / math.log(average_prediction.shape[1])
        th = self._threshold.value
        mask = (entropy <= th).float()

        self.meters["uc_weight"].add(th)
        self.meters["uc_ratio"].add(mask.mean().item())

        # update teacher model here.
        self._ema_updater(self._teacher_model, self._model)
        return (reg_loss.mean(1) * mask).mean()


class MIMeanTeacherEpocher(MITrainEpocher, ):

    def _init(self, *, mi_estimator_array: Iterable[Callable[[Tensor, Tensor], Tensor]],
              teacher_model: nn.Module = None, ema_updater: EMA_Updater = None, mt_weight: float = None,
              mi_weight: float = None, enforce_matching=False, reg_criterion: T_loss = None, **kwargs):
        super(MIMeanTeacherEpocher, self)._init(reg_weight=1.0, mi_estimator_array=mi_estimator_array,
                                                enforce_matching=enforce_matching, **kwargs)
        assert reg_criterion is not None
        assert teacher_model is not None
        assert ema_updater is not None
        assert mt_weight is not None
        assert mi_weight is not None

        self._reg_criterion = reg_criterion  # noqa
        self._teacher_model = teacher_model  # noqa
        self._ema_updater = ema_updater  # noqa
        self._mt_weight = float(mt_weight)  # noqa
        self._mi_weight = float(mi_weight)  # noqa

    def _set_model_state(self, model) -> None:
        model.train()
        self._teacher_model.train()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(MIMeanTeacherEpocher, self)._configure_meters(meters)
        meters.register_meter("consistency", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        with FeatureExtractor(self._teacher_model, self._feature_position) as self._teacher_fextractor:  # noqa
            return super(MIMeanTeacherEpocher, self)._run()

    def _regularization(
        self,
        *,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed: int,
        unlabeled_image: Tensor = None,
        unlabeled_image_tf: Tensor = None,
        **kwargs
    ):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        # clear feature cache
        self._teacher_fextractor.clear()
        with torch.no_grad():
            teacher_logits = self._teacher_model(unlabeled_image)
        with FixRandomSeed(seed):
            teacher_logits_tf = torch.stack([self._affine_transformer(x) for x in teacher_logits], dim=0)
        assert teacher_logits_tf.shape == teacher_logits.shape, (teacher_logits_tf.shape, teacher_logits.shape)

        def generate_iic(student_f, teacher_f, mi_estimator: Callable[[Tensor, Tensor], Tensor]):
            _, student_tf_features = torch.chunk(student_f, 2, dim=0)
            with FixRandomSeed(seed):
                teacher_f_tf = torch.stack([self._affine_transformer(x) for x in teacher_f], dim=0)

            assert teacher_f.shape == teacher_f_tf.shape, (teacher_f.shape, teacher_f_tf.shape)
            loss = mi_estimator(student_f, teacher_f_tf)
            return loss

        loss_list = [
            generate_iic(s, t, mi) for s, t, mi in zip(
                unl_extractor(self._fextractor, n_uls=n_uls),
                self._teacher_fextractor, self._mi_estimator_array)
        ]

        reg_loss = weighted_average_iter(loss_list, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            self._feature_position,
            [-x.item() for x in loss_list]
        )))
        uda_loss = ConsistencyTrainEpocher._regularization(  # noqa
            self,  # noqa
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=teacher_logits_tf.detach(),
            seed=seed,
        )

        # update ema
        self._ema_updater(self._teacher_model, self._model)

        return self._mt_weight * uda_loss + self._mi_weight * reg_loss


class MIDLPaperEpocher(ConsistencyTrainEpocher, ):

    def init(self, *, mi_weight: float, consistency_weight: float, iic_segcriterion: T_loss,  # noqa
             reg_criterion: T_loss,  # noqa
             **kwargs):  # noqa
        super().init(reg_weight=1.0, reg_criterion=reg_criterion, **kwargs)
        self._iic_segcriterion = iic_segcriterion  # noqa
        self._mi_weight = mi_weight  # noqa
        self._consistency_weight = consistency_weight  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(MIDLPaperEpocher, self)._configure_meters(meters)
        meters.register_meter("iic_mi", AverageValueMeter())
        return meters

    def _regularization(
        self,
        *,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, **kwargs
    ):
        uda_loss = super(MIDLPaperEpocher, self)._regularization(
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed, **kwargs
        )
        iic_loss = self._iic_segcriterion(unlabeled_tf_logits.softmax(1), unlabeled_logits_tf.softmax(1).detach())
        self.meters["iic_mi"].add(iic_loss.item())
        return uda_loss * self._consistency_weight + iic_loss * self._mi_weight


class EntropyMinEpocher(TrainEpocher, ):

    def init(self, *, reg_weight: float, **kwargs):
        super().init(reg_weight=reg_weight, **kwargs)
        self._entropy_criterion = Entropy()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(EntropyMinEpocher, self)._configure_meters(meters)
        meters.register_meter("entropy", AverageValueMeter())
        return meters

    def _regularization(
        self,
        *,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, **kwargs
    ):
        reg_loss = self._entropy_criterion(unlabeled_logits_tf.softmax(1))
        self.meters["entropy"].add(reg_loss.item())
        return reg_loss


class _InfoNCEBasedEpocher(_FeatureExtractorMixin, TrainEpocher, ):
    """base epocher class for infonce like method"""

    def __init__(self, *args, **kwargs):
        super(_InfoNCEBasedEpocher, self).__init__(*args, **kwargs)
        self.__set_global_contrast_done__ = False

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: List[T_loss] = None, **kwargs):
        assert projectors_wrapper is not None and infoNCE_criterion is not None, (projectors_wrapper, infoNCE_criterion)
        super()._init(reg_weight=reg_weight, **kwargs)
        assert projectors_wrapper is not None and infoNCE_criterion is not None, (projectors_wrapper, infoNCE_criterion)
        # here we take the infonce as the criterion array (list)
        config = get_config(scope="base")
        assert len(config["ProjectorParams"]["GlobalParams"]["feature_names"]) == len(infoNCE_criterion)

        self._projectors_wrapper: ContrastiveProjectorWrapper = projectors_wrapper  # noqa
        self._encoder_criterion_generator = cycle(infoNCE_criterion)  # noqa
        self._normal_criterion = SupConLoss1()

    def set_global_contrast_method(self, *, contrast_on_list):
        assert isinstance(contrast_on_list, (tuple, list))
        for e in contrast_on_list:
            assert e in ("partition", "patient", "cycle", "self"), e
        config = get_config(scope="base")
        assert len(config["ProjectorParams"]["GlobalParams"]["feature_names"]) == len(contrast_on_list)
        self.__encoder_contrast_name_list = contrast_on_list
        logger.debug("{} set global contrast method to be {}", self.__class__.__name__, ", ".join(contrast_on_list))
        self._encoder_contrastive_name_generator = cycle(self.__encoder_contrast_name_list)
        self.__set_global_contrast_done__ = True

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

    def run(self, *args, **kwargs):
        if not self.__set_global_contrast_done__:
            raise RuntimeError(f"`set_global_contrast_method` should be called first for {self.__class__.__name__}.")
        return super(_InfoNCEBasedEpocher, self).run(*args, **kwargs)

    def unlabeled_projection(self, unl_features, projector, seed, return_unlabeled_features=False):
        unlabeled_features, unlabeled_tf_features = torch.chunk(unl_features, 2, dim=0)
        with FixRandomSeed(seed):
            unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features], dim=0)
        assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
            (unlabeled_tf_features.shape, unlabeled_tf_features.shape)

        proj_tf_feature, proj_feature_tf = torch.chunk(
            projector(torch.cat([unlabeled_tf_features, unlabeled_features_tf], dim=0)), 2, dim=0
        )
        if return_unlabeled_features:
            return proj_tf_feature, proj_feature_tf, unlabeled_tf_features, unlabeled_features_tf
        return proj_tf_feature, proj_feature_tf

    @lru_cache()
    def global_label_generator(self, dataset_name: str, contrast_on: str):
        from .helper import PartitionLabelGenerator, PatientLabelGenerator, ACDCCycleGenerator, SIMCLRGenerator
        if dataset_name == "acdc":
            logger.debug("initialize {} label generator for encoder training", contrast_on)
            if contrast_on == "partition":
                return PartitionLabelGenerator()
            elif contrast_on == "patient":
                return PatientLabelGenerator()
            elif contrast_on == "cycle":
                return ACDCCycleGenerator()
            elif contrast_on == "self":
                return SIMCLRGenerator()
            else:
                raise NotImplementedError(contrast_on)
        elif dataset_name == "prostate":
            if contrast_on == "partition":
                return PartitionLabelGenerator()
            elif contrast_on == "patient":
                return PatientLabelGenerator()
            elif contrast_on == "self":
                return SIMCLRGenerator()
            else:
                raise NotImplementedError(contrast_on)
        elif dataset_name == "mmwhs":
            if contrast_on == "partition":
                return PartitionLabelGenerator()
            elif contrast_on == "patient":
                return PatientLabelGenerator()
            elif contrast_on == "self":
                return SIMCLRGenerator()
            else:
                raise NotImplementedError(contrast_on)
        else:
            NotImplementedError(dataset_name)

    @lru_cache()
    def local_label_generator(self):
        from contrastyou.epocher._utils import LocalLabelGenerator  # noqa
        return LocalLabelGenerator()

    def _regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                        partition_group, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        losses = [
            self.generate_infonce(
                feature_name=n, features=f, projector=p, seed=seed, partition_group=partition_group,
                label_group=label_group) for n, f, p in
            zip(self._feature_position, unl_extractor(self._fextractor, n_uls=n_uls), self._projectors_wrapper)
        ]
        reg_loss = weighted_average_iter(losses, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            [f"{p}|{i}" for i, p in enumerate(self._feature_position)],
            [-x.item() for x in losses]
        )))
        return reg_loss

    def generate_infonce(self, *, feature_name, features, projector, seed, partition_group, label_group) -> Tensor:
        proj_tf_feature, proj_feature_tf = self.unlabeled_projection(unl_features=features, projector=projector,
                                                                     seed=seed)

        if isinstance(projector, ProjectionHead):
            # it goes to **global** representation here.
            return self._global_infonce(
                feature_name=feature_name,
                proj_tf_feature=proj_tf_feature,
                proj_feature_tf=proj_feature_tf,
                partition_group=partition_group,
                label_group=label_group
            )
        # it goes to a **dense** representation on pixels
        return self._dense_based_infonce(
            feature_name=feature_name,
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            partition_group=partition_group,
            label_group=label_group
        )

    @abstractmethod
    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group) -> Tensor:
        ...

    @abstractmethod
    def _dense_based_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                             label_group) -> Tensor:
        ...

    @staticmethod
    def _reshape_dense_feature(proj_tf_feature: Tensor, proj_feature_tf: Tensor):
        """reshape a feature map from [b,c h,w] to [b, hw, c]"""
        b, c, *hw = proj_tf_feature.shape
        proj_tf_feature = proj_tf_feature.view(b, c, -1).permute(0, 2, 1)
        proj_feature_tf = proj_feature_tf.view(b, c, -1).permute(0, 2, 1)
        return proj_tf_feature, proj_feature_tf


class InfoNCEEpocher(_InfoNCEBasedEpocher):
    """INFONCE that implements SIMCLR and SupContrast
        This class take the feature maps (global and/or dense) to perform contrastive pretraining.
        Dense feature and global feature can take from the same position and multiple times.
    """
    from contrastyou.losses.contrast_loss import SupConLoss2 as SupConLoss1
    _encoder_criterion_generator: List[SupConLoss1]

    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group) -> Tensor:
        """methods go for global vectors"""
        assert len(proj_tf_feature.shape) == 2, proj_tf_feature.shape
        assert is_normalized(proj_feature_tf) and is_normalized(proj_feature_tf)

        # generate simclr or supcontrast labels
        contrast_on = next(self._encoder_contrastive_name_generator)
        criterion = next(self._encoder_criterion_generator)  # only for encoder
        config = get_config(scope="base")
        if config["Data"]["name"] == "acdc":
            labels = self.global_label_generator(dataset_name="acdc", contrast_on=contrast_on) \
                (partition_list=partition_group,
                 patient_list=[p.split("_")[0] for p in label_group],
                 experiment_list=[p.split("_")[1] for p in label_group])
        elif config["Data"]["name"] == "prostate":
            labels = self.global_label_generator(dataset_name="prostate", contrast_on=contrast_on) \
                (partition_list=partition_group,
                 patient_list=[p.split("_")[0] for p in label_group])
        elif config["Data"]["name"] in ("mmwhsct", "mmwhsmr"):
            labels = self.global_label_generator(dataset_name="mmwhs", contrast_on=contrast_on) \
                (partition_list=partition_group,
                 patient_list=label_group)
        elif config["Data"]["name"] == "prostate_md":
            labels = self.global_label_generator(dataset_name="prostate", contrast_on=contrast_on) \
                (partition_list=partition_group,
                 patient_list=label_group)
        else:
            raise NotImplementedError(config["Data"]["name"])
        if self.cur_batch_num == 0:  # noqa
            with criterion.register_writer(
                get_tb_writer(), epoch=self._cur_epoch,
                extra_tag=f"{feature_name}/{contrast_on}"
            ):
                return criterion(proj_feature_tf, proj_tf_feature, target=labels)
        return criterion(proj_feature_tf, proj_tf_feature, target=labels)

    def _dense_based_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                             label_group) -> Tensor:
        if "Conv" in feature_name:
            # this is the dense feature from encoder
            return self._dense_infonce_for_encoder(
                feature_name=feature_name,
                proj_tf_feature=proj_tf_feature,
                proj_feature_tf=proj_feature_tf,
                partition_group=partition_group,
                label_group=label_group
            )
        return self._dense_infonce_for_decoder(
            feature_name=feature_name,
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            partition_group=partition_group,
            label_group=label_group
        )

    def _dense_infonce_for_encoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, **kwargs):
        """here the dense prediction does not consider the spatial neighborhood"""
        # the mask of this dense metric would be image-wise simclr
        # usually the spatial size of feature map is very small
        assert "Conv" in feature_name, feature_name

        proj_tf_feature, proj_feature_tf = self._reshape_dense_feature(proj_tf_feature, proj_feature_tf)

        b, hw, c = proj_feature_tf.shape

        if not (is_normalized(proj_feature_tf, dim=2) and is_normalized(proj_feature_tf, dim=2)):
            proj_feature_tf = Normalize(dim=2)(proj_feature_tf)
            proj_tf_feature = Normalize(dim=2)(proj_tf_feature)

        if self.cur_batch_num == 0 and self.trainerself._cur_epoch < 50 and self.trainerself._cur_epoch % 4 == 0:  # noqa
            with self._normal_criterion.register_writer(
                get_tb_writer(),
                epoch=self._cur_epoch,
                extra_tag=f"{feature_name}/dense"
            ):
                return self._normal_criterion(proj_feature_tf.reshape(-1, c), proj_tf_feature.reshape(-1, c))
        return self._normal_criterion(proj_feature_tf.reshape(-1, c), proj_tf_feature.reshape(-1, c))

    def _dense_infonce_for_decoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, **kwargs):
        """here the dense predictions consider the neighborhood information, and the content similarity"""
        assert "Up" in feature_name, feature_name
        b, c, *hw = proj_feature_tf.shape
        config = get_config(scope="base")

        output_size = (12, 12)
        method = config["ProjectorParams"]["DenseParams"]["pool_method"]

        sampled_norm_tf_feature, sampled_norm_feature_tf = self._dense_featuremap_tailoring(
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            output_size=output_size,
            method=method
        )
        assert sampled_norm_tf_feature.shape == torch.Size([b, c, *output_size])
        assert is_normalized(sampled_norm_tf_feature) and is_normalized(sampled_norm_feature_tf)

        n_tf_feature, n_feature_tf = self._reshape_dense_feature(sampled_norm_tf_feature, sampled_norm_feature_tf)
        return self._normal_criterion(n_tf_feature.reshape(-1, c), n_feature_tf.reshape(-1, c))

    def _dense_featuremap_tailoring(self, *, proj_tf_feature: Tensor, proj_feature_tf: Tensor, output_size=(9, 9),
                                    method="adaptive_avg"):
        """
        it consists of
        1. downsampling the feature map to a pre-defined size
        2. sampling fixed position
        3. reshaping the feature map
        4. create a relation mask
        """
        b, c, *hw = proj_feature_tf.shape
        # 1. upsampling
        proj_feature_tf = self._resize_featuremap(output_size=output_size, method=method)(proj_feature_tf)
        proj_tf_feature = self._resize_featuremap(output_size=output_size, method=method)(proj_tf_feature)
        # output features are [b,c,h_,w_] with h_, w_ as the reduced size
        if not (is_normalized(proj_feature_tf, dim=1) and is_normalized(proj_feature_tf, dim=1)):
            proj_feature_tf = Normalize(dim=1)(proj_feature_tf)
            proj_tf_feature = Normalize(dim=1)(proj_tf_feature)
        return proj_tf_feature, proj_feature_tf

    @lru_cache()
    def _resize_featuremap(self, output_size, method="adaptive_avg"):
        if method == "bilinear":
            return partial(F.interpolate, size=output_size, align_corners=True, mode="bilinear")
        elif method == "adaptive_avg":
            return nn.AdaptiveAvgPool2d(output_size=output_size)
        elif method == "adaptive_max":
            return nn.AdaptiveMaxPool2d(output_size=output_size)
        else:
            raise ValueError(method)

    @lru_cache()
    def generate_relation_masks(self, output_size) -> Tensor:
        _pair = ntuple(2)
        output_size = _pair(output_size)
        size = output_size[0] * output_size[1]
        mask = torch.zeros(size, size, dtype=torch.float, device=self._device)
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                relation = torch.ones(*output_size, dtype=torch.float, device=self._device) * -1
                relation[
                max(i - 1, 0):i + 2,
                max(j - 1, 0):j + 2
                ] = 0
                relation[i, j] = 1
                mask[i * output_size[0] + j] = relation.view(1, -1)
        return mask


class InfoNCEMeanTeacherEpocher(_MeanTeacherMixin, InfoNCEEpocher):

    def _init(self, *, infonce_weight: float = None, mt_weight: float = None,
              projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, teacher_model: nn.Module, reg_criterion: T_loss,
              ema_updater: EMA_Updater, **kwargs):
        super()._init(teacher_model=teacher_model, reg_criterion=reg_criterion, ema_updater=ema_updater,
                      reg_weight=1.0, projectors_wrapper=projectors_wrapper, infoNCE_criterion=infoNCE_criterion,
                      **kwargs)
        if self._reg_weight != 1.0:
            raise RuntimeError(self._reg_weight)
        assert isinstance(infonce_weight, (int, float))
        assert isinstance(mt_weight, (int, float))

        self._infonce_w = infonce_weight
        self._mt_w = mt_weight

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(InfoNCEMeanTeacherEpocher, self)._configure_meters(meters)
        meters.register_meter("mt_w", AverageValueMeter())
        meters.register_meter("mi_w", AverageValueMeter())
        return meters

    def _regularization(self, *, unlabeled_image=None,
                        unlabeled_tf_logits: Tensor,
                        unlabeled_logits_tf: Tensor,
                        seed: int, label_group,
                        partition_group, **kwargs):
        infonce_reg = super(InfoNCEMeanTeacherEpocher, self)._regularization(
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf, seed=seed,
            label_group=label_group,
            partition_group=partition_group)
        mt_reg = self._mt_regularization(unlabeled_tf_logits=unlabeled_tf_logits,
                                         seed=seed,
                                         unlabeled_image=unlabeled_image)
        self.update_meanteacher(teacher_model=self._teacher_model, student_model=self._model)
        self.meters["mi_w"].add(self._infonce_w)
        self.meters["mt_w"].add(self._mt_w)

        return self._infonce_w * infonce_reg + self._mt_w * mt_reg
