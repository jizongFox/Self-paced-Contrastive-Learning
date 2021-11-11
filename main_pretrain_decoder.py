import os
from copy import deepcopy as dcopy

import numpy  # noqa
from deepclustering2.loss import KL_div
from loguru import logger

from contrastyou import CONFIG_PATH, success
from contrastyou.configure import ConfigManger
from contrastyou.utils import fix_all_seed_within_context, config_logger, extract_model_state_dict
from hook_creator import create_hook_from_config
from semi_seg import ratio_zoo
from semi_seg.arch import UNet, arch_order
from semi_seg.data.creator import get_data
from semi_seg.hooks import feature_until_from_hooks
from semi_seg.trainers.new_pretrain import PretrainEncoderTrainer, PretrainDecoderTrainer
from utils import separate_pretrain_finetune_configs
from val import val


def main():
    config_manager = ConfigManger(
        base_path=os.path.join(CONFIG_PATH, "base.yaml"),
        optional_paths=os.path.join(CONFIG_PATH, "pretrain.yaml"), strict=False, verbose=False
    )
    # 😁
    pretrain_config, base_config = separate_pretrain_finetune_configs(config_manager=config_manager)

    with config_manager(scope="base") as config:
        seed = config.get("RandomSeed", 10)
        data_name = config["Data"]["name"]
        absolute_save_dir = os.path.abspath(
            os.path.join(PretrainEncoderTrainer.RUN_PATH, str(config["Trainer"]["save_dir"])))
        config_logger(absolute_save_dir)
        with fix_all_seed_within_context(seed):
            model = worker(pretrain_config, absolute_save_dir, seed)

        val(model=model, save_dir=absolute_save_dir, base_config=base_config, seed=seed,
            labeled_ratios=ratio_zoo[data_name])


def worker(config, absolute_save_dir, seed, ):
    config = dcopy(config)
    model_checkpoint = config["Arch"].pop("checkpoint", None)
    with fix_all_seed_within_context(seed):
        model = UNet(**config["Arch"])
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=True)

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=True, total_freedom=False)

    trainer = PretrainDecoderTrainer(model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                                     val_loader=val_loader, test_loader=test_loader,
                                     criterion=KL_div(verbose=False), config=config,
                                     save_dir=os.path.join(absolute_save_dir, "pre"),
                                     **{k: v for k, v in config["Trainer"].items() if k != "save_dir"})

    with fix_all_seed_within_context(seed):
        hooks = create_hook_from_config(model, config, is_pretrain=True)
        assert len(hooks) > 0, "void hooks"

    trainer.register_hooks(*hooks)
    until = feature_until_from_hooks(*hooks)
    assert arch_order(until) > arch_order("Conv5"), until
    trainer.forward_until = until

    with model.set_grad(False):
        with model.set_grad(True, start="Conv5", end=until, include_start=False):
            trainer.init()
            trainer.start_training()

    success(save_dir=trainer.save_dir)
    return model


if __name__ == '__main__':
    # set_deterministic(True)
    main()
