import random
from abc import abstractmethod
from contextlib import nullcontext
from typing import Union, Tuple, Callable

import torch
from deepclustering2.augment.tensor_augment import TensorRandomFlip
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats  # noqa
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, UniversalDice, MeterInterface, SurfaceMeter
from deepclustering2.meters2.individual_meters.averagemeter import AverageValueListMeter
from deepclustering2.models import Model
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.schedulers.customized_scheduler import WeightScheduler
from deepclustering2.type import T_loader, T_loss, T_optim
from deepclustering2.utils import class2one_hot, ExceptionIgnorer
from loguru import logger
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader

from contrastyou.utils import get_dataset
from semi_seg.epochers.helper import preprocess_input_with_single_transformation, write_predict, write_img_target, \
    preprocess_input_with_twice_transformation
from semi_seg.utils import _num_class_mixin


# to enable init and _init, in order to insert assertion of params
class Epocher(_num_class_mixin, _Epocher):
    _forward_pass: Callable

    def init(self, *args, **kwargs):
        self._init(*args, **kwargs)
        self._assertion()

    @abstractmethod
    def _init(self, *args, **kwargs):
        pass

    def _assertion(self):
        pass

    def forward_pass(self, *args, **kwargs):
        return self._forward_pass(*args, **kwargs)


# ======== validation epochers =============
class EvalEpocher(Epocher):

    def __init__(self, model: Union[Model, nn.Module], val_loader: T_loader, sup_criterion: T_loss, cur_epoch=0,
                 device="cpu") -> None:
        assert isinstance(val_loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {val_loader.__class__.__name__}."
        super().__init__(model, num_batches=len(val_loader), cur_epoch=cur_epoch, device=device)
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def _init(self, **kwargs):
        pass

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    def _set_model_state(self, model) -> None:
        model.eval()

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        for i, val_data in zip(self._indicator, self._val_loader):
            val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
            val_logits = self._model(val_img)
            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

            val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), group_name=group)
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        report_dict = self.meters.tracking_status(final=True)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class InferenceEpocher(EvalEpocher):

    def _init(self, *, save_dir: str):
        self._save_dir = save_dir  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("hd", SurfaceMeter(C=C, report_axises=report_axis, metername="hausdorff"))
        return meters

    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        assert self._model.training is False, self._model.training
        for i, val_data in zip(self._indicator, self._val_loader):
            val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
            val_logits = self._model(val_img)
            # write image
            write_img_target(val_img, val_target, self._save_dir, file_path)
            write_predict(val_logits, self._save_dir, file_path, )
            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

            val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), group_name=group)
            with ExceptionIgnorer(RuntimeError):
                self.meters["hd"].add(val_logits.max(1)[1], val_target.squeeze(1))
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        report_dict = self.meters.tracking_status(final=True)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]


# ========= base training epocher ===============
class TrainEpocher(Epocher):

    def __init__(self, *, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 unlabeled_loader: T_loader, sup_criterion: T_loss, num_batches: int, cur_epoch=0,
                 device="cpu", train_with_two_stage: bool = False,
                 disable_bn_track_for_unlabeled_data: bool = False, **kwargs) -> None:
        super().__init__(model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)
        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._sup_criterion = sup_criterion
        self._affine_transformer = TensorRandomFlip(axis=[1, 2], threshold=0.8)

        self.train_with_two_stage = train_with_two_stage  # highlight: this is the parameter to use two stage training
        logger.opt(depth=1).trace("{} set to be using {} stage training", self.__class__.__name__,
                                  "two" if self.train_with_two_stage else "single")
        self._disable_bn = disable_bn_track_for_unlabeled_data  # highlight: disable the bn accumulation
        if self._disable_bn:
            logger.debug("{} set to disable bn tracking", self.__class__.__name__)

    def _init(self, *, reg_weight: float, **kwargs):
        self._reg_weight = reg_weight  # noqa

    def _assertion(self):
        labeled_set = get_dataset(self._labeled_loader)
        labeled_transform = labeled_set.transforms
        assert labeled_transform._total_freedom is False  # noqa

        if self._unlabeled_loader is not None:
            unlabeled_set = get_dataset(self._unlabeled_loader)
            unlabeled_transform = unlabeled_set.transforms
            assert unlabeled_transform._total_freedom is False  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("lr", AverageValueListMeter())
        meters.register_meter("reg_weight", AverageValueMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("reg_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    def _run(self, *args, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        assert self._model.training, self._model.training
        return self._run_semi(*args, **kwargs)

    def _set_model_state(self, model) -> None:
        model.train()

    def _run_semi(self, *args, **kwargs) -> EpochResultDict:
        for self.cur_batch_num, labeled_data, unlabeled_data in zip(self._indicator, self._labeled_loader,
                                                                    self._unlabeled_loader):
            seed = random.randint(0, int(1e7))
            (labeled_image, _), labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)
            (unlabeled_image, unlabeled_image_cf), _, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(
                    unlabeled_data, self._device)

            with FixRandomSeed(seed):
                unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image_cf], dim=0)
            assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                (unlabeled_image_tf.shape, unlabeled_image.shape)

            label_logits, unlabel_logits, unlabel_tf_logits = self.forward_pass(
                labeled_image=labeled_image,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            with FixRandomSeed(seed):
                unlabel_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabel_logits], dim=0)

            assert unlabel_logits_tf.shape == unlabel_tf_logits.shape, (
                unlabel_logits_tf.shape, unlabel_tf_logits.shape)
            # supervised part
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
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
                labeled_filename=labeled_filename
            )

            _reg_weight = self._reg_weight
            if isinstance(self._reg_weight, WeightScheduler):
                _reg_weight = self._reg_weight.value
            self.meters["reg_weight"].add(_reg_weight)

            total_loss = sup_loss + _reg_weight * reg_loss
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
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)
        report_dict = self.meters.tracking_status(final=True)
        return report_dict

    def _forward_pass(self, labeled_image, unlabeled_image, unlabeled_image_tf):
        n_l, n_unl = len(labeled_image), len(unlabeled_image)
        if not self.train_with_two_stage:
            # if train with only single stage
            predict_logits = self._model(torch.cat([labeled_image, unlabeled_image, unlabeled_image_tf], dim=0))
            label_logits, unlabel_logits, unlabel_tf_logits = torch.split(predict_logits,
                                                                          [n_l, n_unl, n_unl], dim=0)
        else:
            # train with two stages, while their feature extractions are the same
            label_logits = self._model(labeled_image)
            bn_context = _disable_tracking_bn_stats if self._disable_bn else nullcontext
            with bn_context(self._model):
                unlabel_logits, unlabel_tf_logits = torch.split(
                    self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0)),
                    [n_unl, n_unl],
                    dim=0
                )
        return label_logits, unlabel_logits, unlabel_tf_logits

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return (image, image_ct), target, filename, partition, group

    def regularization(self, **kwargs):
        return self._regularization(**kwargs)

    def _regularization(self, **kwargs):
        return torch.tensor(0, dtype=torch.float, device=self._device)


class FineTuneEpocher(TrainEpocher):

    def __init__(self, *, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 sup_criterion: T_loss, num_batches: int, cur_epoch=0, device="cpu",
                 **kwargs) -> None:
        super().__init__(model=model, optimizer=optimizer, labeled_loader=labeled_loader,
                         unlabeled_loader=None, sup_criterion=sup_criterion, num_batches=num_batches,  # noqa
                         cur_epoch=cur_epoch, device=device, train_with_two_stage=False,
                         disable_bn_track_for_unlabeled_data=False)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(FineTuneEpocher, self)._configure_meters(meters)  # noqa
        meters.delete_meter("reg_loss")
        meters.delete_meter("reg_weight")
        return meters

    def _run(self, *args, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        assert self._model.training, self._model.training
        return self._run_only_label(*args, **kwargs)

    def _run_only_label(self, *args, **kwargs) -> EpochResultDict:
        for self.cur_batch_num, labeled_data in zip(self._indicator, self._labeled_loader):
            (labeled_image, _), labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)
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
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)
        report_dict = self.meters.tracking_status(final=True)
        return report_dict

    def _forward_pass(self, labeled_image, **kwargs):
        label_logits = self._model(labeled_image)
        return label_logits


class DirectTrainEpocher(FineTuneEpocher):
    pass
