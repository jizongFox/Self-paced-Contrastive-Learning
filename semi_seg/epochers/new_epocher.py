import random
from abc import ABC
from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import torch
from deepclustering2.augment.tensor_augment import TensorRandomFlip
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.type import T_loader, T_loss, T_optim
from deepclustering2.utils import class2one_hot
from loguru import logger
from torch import nn, Tensor

from contrastyou.epochers.base import EpocherBase as _EpocherBase
from contrastyou.meters import MeterInterface, UniversalDice, AverageValueMeter
from contrastyou.utils import get_dataset
from semi_seg.epochers.helper import preprocess_input_with_twice_transformation, \
    preprocess_input_with_single_transformation
from semi_seg.utils import _num_class_mixin


class EpocherBase(_EpocherBase, _num_class_mixin, ABC):

    def init(self):
        self._assertion()

    def _assertion(self):
        pass

    def forward_pass(self, **kwargs):
        for h in self._hooks:
            h.before_forward_pass(**kwargs)
        result = self._forward_pass(**kwargs)
        for h in self._hooks:
            h.after_forward_pass(**kwargs, result_dict=result)
        return result

    def _forward_pass(self, **kwargs) -> Any:
        ...

    def regularization(self, **kwargs):
        for h in self._hooks:
            h.before_regularization(**kwargs)
        result = self._regularization(**kwargs)
        for h in self._hooks:
            h.after_regularization(**kwargs, result_dict=result)
        return result

    def _regularization(self, **kwargs):
        return torch.tensor(0, dtype=torch.float, device=self._device)


class EvalEpocher(EpocherBase):
    meter_focus = "eval"

    def get_score(self):
        metric = self.meters._get_meter(name="dice", group_name=self.meter_focus).summary()  # noqa
        return metric["DSC_mean"]

    def __init__(self, *, model: nn.Module, loader: T_loader, sup_criterion: T_loss, cur_epoch=0, device="cpu") -> None:
        super().__init__(model=model, num_batches=len(loader), cur_epoch=cur_epoch, device=device)
        self._loader = loader
        self._sup_criterion = sup_criterion

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis))
        return meters

    def _run(self, **kwargs):
        self._model.eval()
        return self._run_eval(**kwargs)

    @torch.no_grad()
    def _run_eval(self, **kwargs):
        for i, eval_data in zip(self.indicator, self._loader):
            eval_img, eval_target, file_path, _, group = self._unzip_data(eval_data, self._device)
            eval_logits = self._model(eval_img)
            onehot_target = class2one_hot(eval_target.squeeze(1), self.num_classes)

            eval_loss = self._sup_criterion(eval_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(eval_loss.item())
            self.meters["dice"].add(eval_logits.max(1)[1], eval_target.squeeze(1), group_name=group)

            report_dict = self.meters.statistics()
            self.indicator.set_postfix_statics(report_dict)

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class SemiSupervisedEpocher(EpocherBase, ABC):
    meter_focus = "semi"

    def __init__(self, *, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader,
                 unlabeled_loader: T_loader, sup_criterion: T_loss, num_batches: int, cur_epoch=0,
                 device="cpu", two_stage: bool = False, disable_bn: bool = False, **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)

        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._sup_criterion = sup_criterion
        self._affine_transformer = TensorRandomFlip(axis=[1, 2], threshold=0.8)

        self._two_stage = two_stage  # highlight: this is the parameter to use two stage training
        logger.opt(depth=1).trace("{} set to be using {} stage training", self.__class__.__name__,
                                  "two" if self._two_stage else "single")
        self._disable_bn = disable_bn  # highlight: disable the bn accumulation
        if self._disable_bn:
            logger.debug("{} set to disable bn tracking", self.__class__.__name__)

    def _assertion(self):
        labeled_set = get_dataset(self._labeled_loader)
        labeled_transform = labeled_set.transforms
        assert labeled_transform._total_freedom is False  # noqa

        if self._unlabeled_loader is not None:
            unlabeled_set = get_dataset(self._unlabeled_loader)
            unlabeled_transform = unlabeled_set.transforms
            assert unlabeled_transform._total_freedom is False  # noqa

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(SemiSupervisedEpocher, self).configure_meters(meters)
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis))
        meters.register_meter("reg_loss", AverageValueMeter())
        return meters

    def _run(self, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        self._model.train()
        return self._run_semi(**kwargs)

    def _run_semi(self, **kwargs):
        for self.cur_batch_num, labeled_data, unlabeled_data in zip(self.indicator, self._labeled_loader,
                                                                    self._unlabeled_loader):
            seed = random.randint(0, int(1e7))
            (labeled_image, _), labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)
            (unlabeled_image, unlabeled_image_cf), _, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(unlabeled_data, self._device)
            if self.cur_batch_num < 5:
                logger.trace(f"{self.__class__.__name__}--"
                             f"cur_batch:{self.cur_batch_num}, labeled_filenames: {','.join(labeled_filename)}, "
                             f"unlabeled_filenames: {','.join(unlabeled_filename)}")

            with FixRandomSeed(seed):
                unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image_cf], dim=0)
            assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                (unlabeled_image_tf.shape, unlabeled_image.shape)

            label_logits, unlabeled_logits, unlabeled_tf_logits = self.forward_pass(
                labeled_image=labeled_image,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            with FixRandomSeed(seed):
                unlabeled_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_logits], dim=0)

            # supervised part
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
            # regularized part
            reg_loss = self.regularization(
                unlabeled_tf_logits=unlabeled_tf_logits,
                unlabeled_logits_tf=unlabeled_logits_tf,
                seed=seed,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                unlabeled_filename=unlabeled_filename,
                labeled_filename=labeled_filename,
                affine_transformer=self._affine_transformer
            )

            total_loss = sup_loss + reg_loss
            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            if self.on_master():
                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                                group_name=label_group)
                    self.meters["reg_loss"].add(reg_loss.item())

                report_dict = self.meters.statistics()
                self.indicator.set_postfix_statics(report_dict, cache_time=10)

    def _forward_pass(self, labeled_image, unlabeled_image, unlabeled_image_tf):
        n_l, n_unl = len(labeled_image), len(unlabeled_image)
        if not self._two_stage:
            # if train with only single stage
            predict_logits = self._model(torch.cat([labeled_image, unlabeled_image, unlabeled_image_tf], dim=0))
            label_logits, unlabeled_logits, unlabeled_tf_logits = \
                torch.split(predict_logits, [n_l, n_unl, n_unl], dim=0)
        else:
            # train with two stages, while their feature extractions are the same
            label_logits = self._model(labeled_image)

            bn_context = self._get_bn_context()
            with bn_context(self._model):
                unlabeled_logits, unlabeled_tf_logits = \
                    torch.split(self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0)),
                                [n_unl, n_unl], dim=0)

        return label_logits, unlabeled_logits, unlabeled_tf_logits

    @lru_cache()
    def _get_bn_context(self):
        return _disable_tracking_bn_stats if self._disable_bn else nullcontext

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return (image, image_ct), target, filename, partition, group

    def _regularization(self, **kwargs):
        if len(self._hooks) > 0:
            reg_losses = [h(**kwargs) for h in self._hooks]
            return sum(reg_losses)
        return torch.tensor(0, device=self.device, dtype=torch.float)


class FineTuneEpocher(SemiSupervisedEpocher, ABC):

    def __init__(self, *, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, sup_criterion: T_loss,
                 num_batches: int, cur_epoch=0, device="cpu", **kwargs) -> None:
        super().__init__(model=model, optimizer=optimizer, labeled_loader=labeled_loader,
                         sup_criterion=sup_criterion, num_batches=num_batches,  # noqa
                         cur_epoch=cur_epoch, device=device, **kwargs)

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super().configure_meters(meters)
        meters.delete_meter("reg_loss")
        return meters

    def _run(self, *args, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        self._model.train()
        return self._run_only_label(*args, **kwargs)

    def _run_only_label(self, *args, **kwargs):
        for self.cur_batch_num, labeled_data in zip(self.indicator, self._labeled_loader):
            (labeled_image, _), labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)

            if self.cur_batch_num < 5:
                logger.trace(f"{self.__class__.__name__}--"
                             f"cur_batch:{self.cur_batch_num}, labeled_filenames: {','.join(labeled_filename)}, ")

            label_logits: Tensor = self.forward_pass(labeled_image=labeled_image)  # noqa
            # supervised part
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)

            total_loss = sup_loss
            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            if self.on_master():
                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                                group_name=label_group)
                report_dict = self.meters.statistics()
                self.indicator.set_postfix_statics(report_dict, cache_time=10)

    def _forward_pass(self, labeled_image, **kwargs):
        label_logits = self._model(labeled_image)
        return label_logits
