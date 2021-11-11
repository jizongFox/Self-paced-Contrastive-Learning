import os
import random
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div
from deepclustering2.utils import fix_all_seed_within_context
from deepclustering2.utils import gethash
from loguru import logger
import sys
from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet
from contrastyou.utils import extract_model_state_dict
from semi_seg.dsutils import get_dataloaders, create_val_loader
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

    labeled_loader, unlabeled_loader, test_loader = get_dataloaders(config)
    val_loader, test_loader = create_val_loader(test_loader=test_loader)

    config_arch = deepcopy(config["Arch"])
    model_checkpoint = config_arch.pop("checkpoint", None)
    model = UNet(**config_arch)
    logger.info(f"Initializing {model.__class__.__name__}")
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

    trainer_name = config["Trainer"].pop("name")

    Trainer = trainer_zoos[trainer_name]

    is_pretrain: bool = trainer_name in pre_trainer_zoos

    trainer = Trainer(
        model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
        val_loader=val_loader, test_loader=test_loader, sup_criterion=KL_div(verbose=False),
        configuration={**config, **{"GITHASH": cur_githash}},
        **config["Trainer"]
    )
    trainer.init()
    trainer_checkpoint = config.get("trainer_checkpoint", None)
    if trainer_checkpoint:
        trainer.load_state_dict_from_path(trainer_checkpoint, strict=True)

    if is_pretrain:
        from_, util_ = config["Trainer"]["grad_from"] or "Conv1", \
                       config["Trainer"]["grad_util"] or config["Trainer"]["feature_names"][-1]
        trainer.enable_grad(from_=from_, util_=util_)

    trainer.inference()


if __name__ == '__main__':
    main()
