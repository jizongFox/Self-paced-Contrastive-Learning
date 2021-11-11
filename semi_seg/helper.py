import os
from copy import deepcopy

import numpy as np  # noqa
from deepclustering2.loss import KL_div
from deepclustering2.utils import gethash, fix_all_seed_within_context

from contrastyou.arch.unet import arch_order
from semi_seg.dsutils import get_dataloaders, create_val_loader
from semi_seg.trainers import pre_trainer_zoos, base_trainer_zoos

cur_githash = gethash(__file__)  # noqa
trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}


def pretrain(*, config, model, trainer_name, save_dir, is_monitor=False, seed: int):
    config = deepcopy(config)
    with fix_all_seed_within_context(seed):
        labeled_loader, unlabeled_loader, test_loader = get_dataloaders(config, pretrain=True)
        val_loader, test_loader = create_val_loader(test_loader=test_loader)

        Trainer = trainer_zoos[trainer_name]

        trainer = Trainer(
            model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
            val_loader=val_loader, test_loader=test_loader, sup_criterion=KL_div(verbose=False),
            configuration={**config, **{"GITHASH": cur_githash}},
            save_dir=os.path.join(save_dir, "pre"),
            **{k: v for k, v in config["Trainer"].items() if k != "save_dir" and k != "name"}  # noqa
        )
        trainer.init()
        trainer_checkpoint = config.get("trainer_checkpoint", None)
        if trainer_checkpoint:
            trainer.load_state_dict_from_path(trainer_checkpoint, strict=True)

        if "FeatureExtractor" not in trainer._config:  # noqa
            raise RuntimeError("FeatureExtractor should be in trainer config")
        from_, util_ = \
            config["Trainer"]["grad_from"] or "Conv1", \
            config["Trainer"]["grad_util"] or \
            sorted(trainer._config["FeatureExtractor"]["feature_names"], key=lambda x: arch_order(x))[-1]  # noqa
        with \
            trainer.enable_grad(from_=from_, util_=util_), \
            trainer.enable_bn(from_=from_, util_=util_):
            trainer.start_training(run_monitor=is_monitor)
    return trainer, model, (from_, util_)


def semi_train(*, config, model, label_ratios, seed, save_dir, trainer_name):
    base_model_checkpoint = model.state_dict()

    for labeled_ratio in label_ratios:
        model.load_state_dict(base_model_checkpoint)
        with fix_all_seed_within_context(seed):
            _semi_train(model, config, labeled_ratio, save_dir, trainer_name)


def _semi_train(model, config, labeled_ratio, save_dir, trainer_name):
    config = deepcopy(config)
    config["Data"]["labeled_data_ratio"] = labeled_ratio
    config["Data"]["unlabeled_data_ratio"] = 1 - labeled_ratio

    labeled_loader, unlabeled_loader, test_loader = get_dataloaders(config)
    val_loader, test_loader = create_val_loader(test_loader=test_loader)

    Trainer = trainer_zoos[trainer_name]

    trainer = Trainer(
        model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
        val_loader=val_loader, test_loader=test_loader, sup_criterion=KL_div(verbose=False),
        configuration={**config, **{"GITHASH": cur_githash}},
        save_dir=os.path.join(save_dir, "tra",
                              f"ratio_{str(labeled_ratio)}"),
        **{k: v for k, v in config["Trainer"].items() if k != "save_dir"}  # noqa
    )
    trainer.init()
    trainer.start_training()
