import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import gethash, fix_all_seed_within_context
from loguru import logger

from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet
from contrastyou.utils import extract_model_state_dict
from semi_seg import ratio_zoo
from semi_seg.helper import semi_train
from semi_seg.trainers import pre_trainer_zoos, base_trainer_zoos

cur_githash = gethash(__file__)  # noqa
print(sys.path)
trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}


def main():
    config_manager = ConfigManger(base_path=Path(PROJECT_PATH) / "config/base.yaml", strict=False)
    with config_manager(scope="base") as config:
        port = random.randint(10000, 60000)
        main_worker(0, 1, config, config_manager, port)


@logger.catch(reraise=True)
def main_worker(rank, ngpus_per_node, config, config_manager, port):  # noqa
    save_dir = str(config["Trainer"]["save_dir"])
    logger.add(os.path.join("runs", save_dir, "loguru.log"), level="TRACE", diagnose=True, )

    seed = config.get("RandomSeed", 10)
    config_arch = deepcopy(config["Arch"])
    model_checkpoint = config_arch.pop("checkpoint", None)

    with fix_all_seed_within_context(seed):
        model = UNet(**config_arch)
    logger.info(f"Initializing {model.__class__.__name__}")
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=True)

    ratios = ratio_zoo[config["Data"]["name"]]
    trainer_name = config["Trainer"]["name"]
    semi_train(model=model, label_ratios=ratios, config=config, seed=seed, save_dir=save_dir,
               trainer_name=trainer_name)


if __name__ == '__main__':
    main()
