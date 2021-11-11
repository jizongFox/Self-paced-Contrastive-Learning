import os
import random
from copy import deepcopy
from pathlib import Path

import numpy  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div
from deepclustering2.utils import gethash
from loguru import logger

from contrastyou import PROJECT_PATH, success
from contrastyou.utils import extract_model_state_dict, fix_all_seed_within_context, set_deterministic
from semi_seg import ratio_zoo
from semi_seg.arch import UNet
from semi_seg.data import get_data_loaders, create_val_loader
from semi_seg.trainers import pre_trainer_zoos, base_trainer_zoos, FineTuneTrainer

cur_githash = gethash(__file__)  # noqa
trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}


def main():
    config_manager = ConfigManger(base_path=Path(PROJECT_PATH) / "config/base.yaml")
    with config_manager(scope="base") as config:
        port = random.randint(10000, 60000)
        main_worker(0, 1, config, config_manager, port)


@logger.catch(reraise=True)
def main_worker(rank, ngpus_per_node, config, config_manager, port):  # noqa
    base_save_dir = str(config["Trainer"]["save_dir"])
    logger.add(os.path.join("runs", base_save_dir, "loguru.log"), level="TRACE", diagnose=True, )

    seed = config.get("RandomSeed", 1)

    with fix_all_seed_within_context(seed):

        config_arch = deepcopy(config["Arch"])
        model_checkpoint = config_arch.pop("checkpoint", None)
        model = UNet(**config_arch)
        logger.info(f"Initializing {model.__class__.__name__}")
        if model_checkpoint:
            logger.info(f"loading checkpoint from  {model_checkpoint}")
            model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

        trainer_name = config["Trainer"].pop("name")
        assert trainer_name in ("finetune", "directtrain"), trainer_name
        base_model_checkpoint = deepcopy(model.state_dict())

    ratios = ratio_zoo[config["Data"]["name"]]

    def finetune():
        model.load_state_dict(base_model_checkpoint)

        config["Data"]["labeled_data_ratio"] = labeled_ratio
        config["Data"]["unlabeled_data_ratio"] = 1 - labeled_ratio

        labeled_loader, unlabeled_loader, test_loader = get_data_loaders(
            config["Data"], config["LabeledLoader"], config["UnlabeledLoader"],
        )
        val_loader, test_loader = create_val_loader(test_loader=test_loader)

        labeled_loader.dataset.preload()
        unlabeled_loader.dataset.preload()
        val_loader.dataset.preload()
        test_loader.dataset.preload()

        save_dir = os.path.join(base_save_dir, "tra", f"ratio_{str(labeled_ratio)}")

        finetune_trainer = FineTuneTrainer(
            model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
            val_loader=val_loader, test_loader=test_loader, sup_criterion=KL_div(verbose=False),
            configuration={**config, **{"GITHASH": cur_githash}},
            save_dir=save_dir,
            **{k: v for k, v in config["Trainer"].items() if k != "save_dir"}
        )
        finetune_trainer.init()
        finetune_trainer.start_training()
        success(save_dir=finetune_trainer._save_dir)  # noqa

    for labeled_ratio in ratios:
        with fix_all_seed_within_context(seed):
            finetune()


if __name__ == '__main__':
    set_deterministic()
    main()
