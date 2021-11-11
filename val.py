import os
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy as dcopy
from typing import Dict, Any, List

from deepclustering2.loss import KL_div
from torch import nn

from contrastyou import success
from contrastyou.utils import fix_all_seed_within_context
from semi_seg.data.creator import get_data
from semi_seg.trainers.new_trainer import FineTuneTrainer


@contextmanager
def switch_model_device(model: nn.Module, device: str = "cpu"):
    previous_device = next(model.parameters()).device
    model.to(device)
    yield
    model.to(previous_device)


def val(*, model: nn.Module, save_dir: str, base_config: Dict[str, Any], labeled_ratios: List[float],
        seed: int = 10):
    with switch_model_device(model, device="cpu"):
        holding_state_dict = OrderedDict(model.state_dict())

    data_params = base_config["Data"]
    loader_l_params = base_config["LabeledLoader"]
    loader_u_params = base_config["UnlabeledLoader"]
    trainer_params = base_config["Trainer"]
    for ratio in labeled_ratios:
        model.load_state_dict(holding_state_dict)
        with fix_all_seed_within_context(seed):
            """ Inside the seed:
            1. create loader
            2. running the fine-tune trainer
            """
            _val(model=model, data_params=data_params, labeled_loader_params=loader_l_params,
                 unlabeled_loader_params=loader_u_params, main_save_dir=save_dir, trainer_params=trainer_params,
                 global_config=base_config, labeled_data_ratio=ratio)


def _val(*, model: nn.Module, labeled_data_ratio: float, data_params: Dict[str, Any],
         labeled_loader_params: Dict[str, Any], unlabeled_loader_params: Dict[str, Any], main_save_dir: str,
         trainer_params: Dict[str, Any], global_config: Dict[str, Any]):
    data_params, trainer_params, global_config = list(map(dcopy, [data_params, trainer_params, global_config]))

    data_params["labeled_scan_num"] = float(labeled_data_ratio)
    global_config["Data"]["labeled_scan_num"] = float(labeled_data_ratio)

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=data_params, labeled_loader_params=labeled_loader_params,
        unlabeled_loader_params=unlabeled_loader_params, pretrain=False)

    trainer_params["save_dir"] = os.path.join(main_save_dir, "tra",
                                              f"num_labeled_scan_{len(labeled_loader.dataset.get_scan_list())}")

    trainer = FineTuneTrainer(model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                              val_loader=val_loader, test_loader=test_loader,
                              criterion=KL_div(verbose=False), config=global_config, **trainer_params)

    trainer.init()
    trainer.start_training()
    success(save_dir=trainer.save_dir)
