import os
import random
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div
from deepclustering2.utils import gethash
from loguru import logger

from contrastyou import PROJECT_PATH, success
from contrastyou.utils import extract_model_state_dict, fix_all_seed_within_context, set_deterministic
from semi_seg.arch import UNet
from semi_seg.data import get_data_loaders, create_val_loader
from semi_seg.trainers import pre_trainer_zoos, base_trainer_zoos

warnings.filterwarnings("ignore")

cur_githash = gethash(__file__)  # noqa
print(sys.path)

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}


def main():
    config_manager = ConfigManger(base_path=Path(PROJECT_PATH) / "config/base.yaml", strict=False)
    with config_manager(scope="base") as config:
        save_dir = "runs/" + str(config["Trainer"]["save_dir"])
        logger.add(os.path.join(save_dir, "loguru.log"), level="TRACE", diagnose=True, )

        port = random.randint(10000, 60000)
        seed = config.get("RandomSeed", 1)
        with fix_all_seed_within_context(seed):
            main_worker(0, 1, config, port)


@logger.catch(reraise=True)
def main_worker(rank, ngpus_per_node, config, port):  # noqa

    trainer_name = config["Trainer"].get("name")
    is_pretrain: bool = trainer_name in pre_trainer_zoos

    labeled_loader, unlabeled_loader, test_loader = get_data_loaders(
        config["Data"], config["LabeledLoader"], config["UnlabeledLoader"], pretrain=is_pretrain,
        total_freedom=is_pretrain)
    val_loader, test_loader = create_val_loader(test_loader=test_loader)

    labeled_loader.dataset.preload()
    unlabeled_loader.dataset.preload()
    val_loader.dataset.preload()
    test_loader.dataset.preload()

    config_arch = deepcopy(config["Arch"])
    model_checkpoint = config_arch.pop("checkpoint", None)
    model = UNet(**config_arch)
    logger.info(f"Initializing {model.__class__.__name__}")
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

    Trainer = trainer_zoos[trainer_name]

    trainer = Trainer(
        model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
        val_loader=val_loader, test_loader=test_loader, sup_criterion=KL_div(verbose=False),
        configuration={**config, **{"GITHASH": cur_githash}},
        **{k: v for k, v in config["Trainer"].items() if k != "name"}
    )
    trainer.init()
    trainer_checkpoint = config.get("trainer_checkpoint", None)
    if trainer_checkpoint:
        trainer.load_state_dict_from_path(trainer_checkpoint, strict=True)

    if is_pretrain:
        from_, util_ = config["Trainer"]["grad_from"] or "Conv1", \
                       config["Trainer"]["grad_util"] or config["Trainer"]["feature_names"][-1]

        with model.set_grad(False):
            with model.set_grad(True, start=from_, end=util_):
                trainer.start_training()
    else:
        trainer.start_training()
    success(save_dir=trainer._save_dir)  # noqa


if __name__ == '__main__':
    set_deterministic()
    main()
