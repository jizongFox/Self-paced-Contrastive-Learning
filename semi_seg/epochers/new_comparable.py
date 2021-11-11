import random
from abc import ABC
from functools import lru_cache

import torch
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.type import T_optim, T_loader, T_loss
from deepclustering2.utils import class2one_hot
from loguru import logger
from torch import nn

from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.utils import get_dataset
from . import preprocess_input_with_twice_transformation
from .new_epocher import SemiSupervisedEpocher


class MixUpEpocher(SemiSupervisedEpocher):

    def _assertion(self):
        labeled_set = get_dataset(self._labeled_loader)
        labeled_transform = labeled_set.transforms
        assert labeled_transform._total_freedom is True  # noqa

        if self._unlabeled_loader is not None:
            unlabeled_set = get_dataset(self._unlabeled_loader)
            unlabeled_transform = unlabeled_set.transforms
            assert unlabeled_transform._total_freedom is True  # noqa

    def _run(self, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        self._model.train()
        return self._run_mix_up(**kwargs)

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return (image, image_ct), (target, target_ct), filename, partition, group

    def _run_mix_up(self, **kwargs):
        for self.cur_batch_num, labeled_data, in zip(self.indicator, self._labeled_loader):
            seed = random.randint(0, int(1e7))
            (labeled_image, labeled_image_tf), (labeled_target, labeled_target_tf), labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)

            if self.cur_batch_num < 5:
                logger.trace(f"{self.__class__.__name__}--"
                             f"cur_batch:{self.cur_batch_num}, labeled_filenames: {','.join(labeled_filename)}")

            label_logits = self.forward_pass(
                labeled_image=labeled_image,
                labeled_image_tf=labeled_image_tf
            )

            # supervised part
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
            # regularized part
            reg_loss = self.regularization(
                labeled_image=labeled_image,
                labeled_image_tf=labeled_image_tf,
                labeled_target=labeled_target,
                labeled_target_tf=labeled_target_tf,
                seed=seed
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

    def _forward_pass(self, labeled_image, **kwargs):
        label_logits = self._model(labeled_image)
        return label_logits


class AdversarialEpocher(SemiSupervisedEpocher, ABC):

    def _assertion(self):
        pass

    def __init__(self, *, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 sup_criterion: T_loss, num_batches: int, cur_epoch=0, device="cpu", two_stage: bool = False,
                 disable_bn: bool = False, discriminator=None, discr_optimizer=None, reg_weight=None,
                 dis_consider_image: bool, **kwargs) -> None:
        super().__init__(model=model, optimizer=optimizer, labeled_loader=labeled_loader,
                         unlabeled_loader=unlabeled_loader, sup_criterion=sup_criterion, num_batches=num_batches,
                         cur_epoch=cur_epoch, device=device, two_stage=two_stage, disable_bn=disable_bn, **kwargs)
        self._discriminator = discriminator
        self._discr_optimizer = discr_optimizer
        self._reg_weight = float(reg_weight)
        assert isinstance(discriminator, nn.Module)
        assert isinstance(discr_optimizer, torch.optim.Optimizer)
        self._dis_consider_image = dis_consider_image

    def _run(self, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        self._model.train()
        return self._run_adver(**kwargs)

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(AdversarialEpocher, self).configure_meters(meters)
        meters.delete_meter("reg_loss")
        with self.meters.focus_on("adv_reg"):
            meters.register_meter("dis_loss", AverageValueMeter())
            meters.register_meter("gen_loss", AverageValueMeter())
            meters.register_meter("reg_weight", AverageValueMeter())
        return meters

    def _run_adver(self, **kwargs):
        # following https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        criterion = nn.BCELoss()
        TRUE_LABEL = 1.
        FAKE_LABEL = 0.
        optimizerD = self._discr_optimizer
        optimizerG = self._optimizer
        with self.meters.focus_on("adv_reg"):
            self.meters["reg_weight"].add(self._reg_weight)

        for self.cur_batch_num, labeled_data, in zip(self.indicator, self._labeled_loader):
            (labeled_image, _), labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)
            if self._reg_weight > 0:
                unlabeled_data = next(self.unlabeled_iter)
                (unlabeled_image, _), _, unlabeled_filename, unl_partition, unl_group = \
                    self._unzip_data(unlabeled_data, self._device)
            if self.cur_batch_num < 5:
                if self._reg_weight > 0:
                    logger.trace(f"{self.__class__.__name__}--"
                                 f"cur_batch:{self.cur_batch_num}, labeled_filenames: {','.join(labeled_filename)}, "
                                 f"unlabeled_filenames: {','.join(unlabeled_filename)}")
                else:
                    logger.trace(f"{self.__class__.__name__}--"
                                 f"cur_batch:{self.cur_batch_num}, labeled_filenames: {','.join(labeled_filename)}")

            # update segmentation
            self._optimizer.zero_grad()
            labeled_logits = self._model(labeled_image)
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(labeled_logits.softmax(1), onehot_target)
            generator_err = torch.tensor(0, device=self.device, dtype=torch.float)
            if self._reg_weight > 0:
                unlabeled_logits = self._model(unlabeled_image)
                if self._dis_consider_image:
                    discr_output_unlabeled = self._discriminator(
                        torch.cat([unlabeled_image, unlabeled_logits.softmax(1)], dim=1))
                else:
                    discr_output_unlabeled = self._discriminator(unlabeled_logits.softmax(1))

                generator_err = criterion(discr_output_unlabeled,
                                          torch.zeros_like(discr_output_unlabeled).fill_(TRUE_LABEL))
            generator_loss = sup_loss + self._reg_weight * generator_err
            generator_loss.backward()
            optimizerG.step()
            if self.on_master():
                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["sup_dice"].add(labeled_logits.max(1)[1], labeled_target.squeeze(1),
                                                group_name=label_group)
                    with self.meters.focus_on("adv_reg"):
                        self.meters["gen_loss"].add(generator_err.item())
            disc_loss = torch.tensor(0, device=self.device, dtype=torch.float)
            if self._reg_weight > 0:
                # first # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                self._discriminator.zero_grad()
                if self._dis_consider_image:
                    discr_output_labeled = self._discriminator(
                        torch.cat([labeled_image, labeled_logits.detach().softmax(1)], dim=1))
                else:
                    discr_output_labeled = self._discriminator(labeled_logits.detach().softmax(1))
                discr_err_labeled = criterion(discr_output_labeled,
                                              torch.zeros_like(discr_output_labeled).fill_(TRUE_LABEL))
                if self._dis_consider_image:
                    discr_output_unlabeled = self._discriminator(
                        torch.cat([unlabeled_image, unlabeled_logits.detach().softmax(1)], dim=1))
                else:
                    discr_output_unlabeled = self._discriminator(unlabeled_logits.detach().softmax(1))
                discr_err_unlabeled = criterion(discr_output_unlabeled,
                                                torch.zeros_like(discr_output_unlabeled).fill_(FAKE_LABEL))
                disc_loss = discr_err_labeled + discr_err_unlabeled
                (disc_loss * self._reg_weight).backward()
                optimizerD.step()
            if self.on_master():
                with self.meters.focus_on("adv_reg"):
                    self.meters["dis_loss"].add(disc_loss.item())

                report_dict = self.meters.statistics()
                self.indicator.set_postfix_statics(report_dict, cache_time=10)

    @property
    @lru_cache()
    def unlabeled_iter(self):
        # this is to match the baseline trainer to avoid any perturbation on the baseline
        return iter(self._unlabeled_loader)
