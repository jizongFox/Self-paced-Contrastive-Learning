# ======== base pretrain epocher mixin ================
import random
from typing import Callable

import torch
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.optim import get_lrs_from_optimizer
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from contrastyou.meters import MeterInterface
from contrastyou.mytqdm import tqdm
from contrastyou.utils import get_dataset
from semi_seg.epochers import preprocess_input_with_twice_transformation
from semi_seg.epochers.new_epocher import SemiSupervisedEpocher


class _PretrainEpocherMixin:
    """
    PretrainEpocher makes all images goes to regularization, permitting to use the other classes to create more pretrain
    models
    """
    meters: MeterInterface
    _model: nn.Module
    _optimizer: Optimizer
    indicator: tqdm
    _labeled_loader: DataLoader
    _unlabeled_loader: DataLoader
    _unzip_data: Callable[..., torch.device]
    _device: torch.device
    _affine_transformer: Callable[[Tensor], Tensor]
    on_master: Callable[[], bool]
    regularization: Callable[..., Tensor]
    forward_pass: Callable

    def __init__(self, *, chain_dataloader, inference_until: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._chain_dataloader = chain_dataloader
        self._inference_until = inference_until

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meter = super().configure_meters(meters)  # noqa
        meter.delete_meters(["sup_loss", "sup_dice"])
        return meter

    def _run(self, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        self._model.train()
        return self._run_pretrain(**kwargs)

    def _run_pretrain(self, **kwargs):
        for self.cur_batch_num, data in zip(self.indicator, self._chain_dataloader):
            seed = random.randint(0, int(1e7))
            (unlabeled_image, unlabeled_image_tf), _, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(data, self._device)
            with FixRandomSeed(seed):
                unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image_tf], dim=0)

            unlabeled_logits, unlabeled_tf_logits = self.forward_pass(
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )
            with FixRandomSeed(seed):
                unlabeled_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_logits], dim=0)

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
                affine_transformer=self._affine_transformer
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
                report_dict = self.meters.statistics()
                self.indicator.set_postfix_statics(report_dict, cache_time=20)

    def _forward_pass(self, unlabeled_image, unlabeled_image_tf):
        n_l, n_unl = 0, len(unlabeled_image)
        predict_logits = self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0),
                                     until=self._inference_until)
        unlabeled_logits, unlabeled_tf_logits = torch.split(predict_logits, [n_unl, n_unl], dim=0)
        return unlabeled_logits, unlabeled_tf_logits

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return (image, image_ct), None, filename, partition, group


class PretrainEncoderEpocher(_PretrainEpocherMixin, SemiSupervisedEpocher):
    def _assertion(self):
        labeled_set = get_dataset(self._labeled_loader)
        labeled_transform = labeled_set.transforms
        assert labeled_transform._total_freedom  # noqa

        if self._unlabeled_loader is not None:
            unlabeled_set = get_dataset(self._unlabeled_loader)
            unlabeled_transform = unlabeled_set.transforms
            assert unlabeled_transform._total_freedom  # noqa


class PretrainDecoderEpocher(_PretrainEpocherMixin, SemiSupervisedEpocher):
    def _assertion(self):
        labeled_set = get_dataset(self._labeled_loader)
        labeled_transform = labeled_set.transforms
        assert labeled_transform._total_freedom is False  # noqa

        if self._unlabeled_loader is not None:
            unlabeled_set = get_dataset(self._unlabeled_loader)
            unlabeled_transform = unlabeled_set.transforms
            assert unlabeled_transform._total_freedom is False  # noqa
