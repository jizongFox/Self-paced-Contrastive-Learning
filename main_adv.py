# the adversarial training use trainer and epochers directly, without using the hook, since it consists of multiple
# gradient steps.
import os

from deepclustering2.loss import KL_div
from loguru import logger

from contrastyou import CONFIG_PATH, success
from contrastyou.configure import ConfigManger
from contrastyou.utils import fix_all_seed_within_context, config_logger, set_deterministic, extract_model_state_dict
from semi_seg.arch import UNet
from semi_seg.data.creator import get_data
from semi_seg.trainers.new_trainer import SemiTrainer, AdversarialTrainer


def main():
    with ConfigManger(
        base_path=os.path.join(CONFIG_PATH, "base.yaml"), strict=True
    )(scope="base") as config:
        seed = config.get("RandomSeed", 10)
        _save_dir = config["Trainer"]["save_dir"]
        absolute_save_dir = os.path.abspath(os.path.join(SemiTrainer.RUN_PATH, _save_dir))
        config_logger(absolute_save_dir)
        with fix_all_seed_within_context(seed):
            worker(config, absolute_save_dir, seed)


def worker(config, absolute_save_dir, seed, ):
    model_checkpoint = config["Arch"].pop("checkpoint", None)
    with fix_all_seed_within_context(seed):
        model = UNet(**config["Arch"])
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=True)

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=False, total_freedom=True)

    checkpoint = config.get("trainer_checkpoint")

    trainer = AdversarialTrainer(model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                                 val_loader=val_loader, test_loader=test_loader,
                                 criterion=KL_div(verbose=False), config=config,
                                 save_dir=absolute_save_dir,
                                 **{k: v for k, v in config["Trainer"].items() if k != "save_dir" and k != "name"})

    trainer.init()
    if checkpoint:
        trainer.resume_from_path(checkpoint)
    trainer.start_training()
    success(save_dir=trainer.save_dir)


if __name__ == '__main__':
    set_deterministic(True)
    main()
