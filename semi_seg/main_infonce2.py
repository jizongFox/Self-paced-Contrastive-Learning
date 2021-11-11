import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import gethash, write_yaml, fix_all_seed_within_context
from loguru import logger

from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet
from contrastyou.utils import extract_model_state_dict
from semi_seg import ratio_zoo
from semi_seg.helper import semi_train, pretrain  # noqa
from semi_seg.trainers import pre_trainer_zoos, base_trainer_zoos

cur_githash = gethash(__file__)  # noqa

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}


def main():
    config_manager = ConfigManger(base_path=Path(PROJECT_PATH) / "config/base.yaml")
    with config_manager(scope="base") as config:
        port = random.randint(10000, 60000)
        parsed_config = config_manager.parsed_config
        if "Optim" in parsed_config and "lr" in parsed_config["Optim"]:
            raise RuntimeError("`Optim.lr` should not be provided in this interface, "
                               "Provide `Optim.pre_lr` and `Optim.ft_lr` instead.")
        if "max_epoch" in parsed_config.get("Trainer", {}):
            raise RuntimeError("Trainer.max_epoch should not be provided,"
                               "Provide `Trainer.pre_max_epoch` and `Trainer.ft_max_epoch` instead.")

        main_worker(0, 1, config, config_manager, port)


@logger.catch(reraise=True)
def main_worker(rank, ngpus_per_node, config, config_manager, port):  # noqa
    save_dir = str(config["Trainer"]["save_dir"])
    logger.add(os.path.join("runs", save_dir, "loguru.log"), level="TRACE", diagnose=True)
    write_yaml(config, save_dir=os.path.join("runs", save_dir), save_name="config_raw.yaml")

    seed = config.get("RandomSeed", 10)

    config_arch = config["Arch"]
    with fix_all_seed_within_context(seed):
        model_checkpoint = config_arch.pop("checkpoint", None)
        model = UNet(**config_arch)
    logger.info(f"Initializing {model.__class__.__name__}")
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

    pre_lr, ft_lr = config["Optim"].pop("pre_lr", None), config["Optim"].pop("ft_lr", None)
    pre_max_epoch, ft_max_epoch = config["Trainer"].pop("pre_max_epoch", None), \
                                  config["Trainer"].pop("ft_max_epoch", None)

    pretrain_config = deepcopy(config)
    pretrain_trainer_name = pretrain_config["Trainer"]["name"]
    pretrain_config["Optim"]["lr"] = float(pre_lr or pretrain_config["Optim"]["lr"])
    pretrain_config["Trainer"]["max_epoch"] = int(pre_max_epoch or pretrain_config["Trainer"]["max_epoch"])
    is_monitor = pretrain_config["Trainer"].get("monitor", False)
    pre_trainer, model, *_ = pretrain(
        config=pretrain_config, model=model, trainer_name=pretrain_trainer_name, is_monitor=is_monitor,
        save_dir=save_dir, seed=seed
    )

    finetune_config = deepcopy(config)
    base_config = config_manager.base_config
    finetune_config["Optim"] = base_config["Optim"]
    finetune_config["Optim"]["lr"] = float(ft_lr or base_config["Optim"]["lr"])
    finetune_config["Trainer"]["max_epoch"] = int(ft_max_epoch or base_config["Trainer"]["max_epoch"])
    finetune_config["Scheduler"]["multiplier"] = int(base_config["Scheduler"]["multiplier"])

    ratios = ratio_zoo[config["Data"]["name"]]

    semi_train(
        model=model, label_ratios=ratios, seed=seed, save_dir=save_dir, trainer_name="finetune", config=finetune_config
    )


if __name__ == '__main__':
    main()
