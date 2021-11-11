import os
from itertools import cycle
from pathlib import Path

import numpy as np
from deepclustering2.cchelper import JobSubmiter

from contrastyou import PROJECT_PATH, on_cc
from contrastyou.configure import dictionary_merge_by_hierachy
from script import utils
from script.utils import TEMP_DIR, yaml_load, write_yaml, grid_search, PretrainScriptGenerator, move_dataset
from semi_seg import __accounts, num_batches_zoo, ft_max_epoch_zoo, pre_max_epoch_zoo
from semi_seg.trainers.new_trainer import SemiTrainer

RUN_DIR = SemiTrainer.RUN_PATH

account = cycle(__accounts)
opt_hook_path = {"infonce": "config/hooks/infonce.yaml",
                 "spinfonce": "config/hooks/spinfonce.yaml"}


class PretrainInfoNCEScriptGenerator(PretrainScriptGenerator):

    def __init__(self, *, data_name, num_batches, save_dir, pre_max_epoch, ft_max_epoch) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, save_dir=save_dir,
                         pre_max_epoch=pre_max_epoch, ft_max_epoch=ft_max_epoch)

        self.hook_config = yaml_load(PROJECT_PATH + "/" + opt_hook_path[self.get_hook_name()])

    def get_hook_name(self):
        return "infonce"

    def get_hook_params(self, weight, contrast_on):
        return {"InfonceParams": {"weights": weight,
                                  "contrast_ons": contrast_on}}

    def grid_search_on(self, *, seed, **kwargs):
        jobs = []
        checkpoint_path_list = []
        seeds = []
        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            hook_params = self.get_hook_params(**param)
            sub_save_dir = self._get_hyper_param_string(**param)
            merged_config = dictionary_merge_by_hierachy(self.hook_config, hook_params)
            config_path = write_yaml(merged_config, save_dir=TEMP_DIR, save_name=utils.random_string() + ".yaml")
            true_save_dir = os.path.join(self._save_dir, "Seed_" + str(random_seed), sub_save_dir)
            job = self.generate_single_script(save_dir=true_save_dir,
                                              seed=random_seed, hook_path=config_path)
            jobs.append(job)
            checkpoint_path_list.append(os.path.join(RUN_DIR, true_save_dir, "pre", "last.pth"))
            seeds.append(random_seed)
        return jobs, checkpoint_path_list, seeds

    def generate_single_script(self, save_dir, seed, hook_path):
        from semi_seg import pre_lr_zooms, ft_lr_zooms
        pre_lr = pre_lr_zooms[self._data_name]
        ft_lr = ft_lr_zooms[self._data_name]
        return f"python main_pretrain_encoder.py Trainer.save_dir={save_dir} " \
               f" Optim.pre_lr={pre_lr:.7f} Optim.ft_lr={ft_lr:.7f} RandomSeed={str(seed)} " \
               f" {' '.join(self.conditions)}  " \
               f" --opt-path config/pretrain.yaml {hook_path}"


class PretrainDecoderScriptGenerator(PretrainScriptGenerator):

    def __init__(self, *, data_name, num_batches, save_dir, pre_max_epoch, ft_max_epoch, checkpoint_path: str) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, save_dir=save_dir, pre_max_epoch=pre_max_epoch,
                         ft_max_epoch=ft_max_epoch)
        assert isinstance(checkpoint_path, str), checkpoint_path
        self._checkpoint_path = checkpoint_path
        self.conditions.append(f"Arch.checkpoint={checkpoint_path}")

        self._decoder_hook = yaml_load(PROJECT_PATH + "/" + "config/hooks/infonce_dense.yaml")

    def get_decoder_hook_params(self, decoder_weight, decoder_feature_names, ):
        return {"InfonceParams": {"weights": decoder_weight,
                                  "feature_names": decoder_feature_names}}

    def grid_search_on(self, *, seed, **kwargs):
        jobs = []
        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            decoder_hook_params = self.get_decoder_hook_params(**param)
            sub_save_dir = self._get_hyper_param_string(**param)
            merged_config = dictionary_merge_by_hierachy(self._decoder_hook, decoder_hook_params)
            config_path = write_yaml(merged_config, save_dir=TEMP_DIR, save_name=utils.random_string() + ".yaml")
            true_save_dir = os.path.join(self._save_dir, "decoder", "Seed_" + str(random_seed), sub_save_dir)
            job = self.generate_single_script(save_dir=true_save_dir,
                                              seed=random_seed, hook_path=config_path)
            jobs.append(job)
        return jobs

    def generate_single_script(self, save_dir, seed, hook_path):
        from semi_seg import pre_lr_zooms, ft_lr_zooms
        pre_lr = pre_lr_zooms[self._data_name]
        ft_lr = ft_lr_zooms[self._data_name]
        return f"python main_pretrain_decoder.py Trainer.save_dir={save_dir} " \
               f" Optim.pre_lr={pre_lr:.7f} Optim.ft_lr={ft_lr:.7f} RandomSeed={str(seed)} " \
               f" {' '.join(self.conditions)}  " \
               f" --opt-path config/pretrain.yaml {hook_path}"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--data-name", required=True, choices=["acdc", "prostate", "mmwhsct"])
    parser.add_argument("-s", "--save_dir", required=True, type=str)
    parser.add_argument("stage", choices=["encoder", "decoder"])
    args = parser.parse_args()

    submittor = JobSubmiter(on_local=not on_cc(), project_path="../", time=4)
    submittor.prepare_env([
        "module load python/3.8.2 ",
        f"source ~/venv/bin/activate ",
        'if [ $(which python) == "/usr/bin/python" ]',
        "then",
        "exit 9",
        "fi",

        "export OMP_NUM_THREADS=1",
        "export PYTHONOPTIMIZE=1",
        "export PYTHONWARNINGS=ignore ",
        "export CUBLAS_WORKSPACE_CONFIG=:16:8 ",
        move_dataset()
    ])

    seed = [10, 20, 30, 40]
    data_name = args.data_name
    save_dir = f"{args.save_dir}/{data_name}"
    num_batches = num_batches_zoo[data_name]
    pre_max_epoch = pre_max_epoch_zoo[data_name]
    ft_max_epoch = ft_max_epoch_zoo[data_name]

    if args.stage == "encoder":

        baseline_generator = PretrainInfoNCEScriptGenerator(data_name=data_name, num_batches=num_batches,
                                                            save_dir=f"{save_dir}/baseline",
                                                            pre_max_epoch=0, ft_max_epoch=ft_max_epoch)
        baseline_jobs, *_ = baseline_generator.grid_search_on(weight=1, contrast_on="baseline", seed=seed)

        encoder_generator = PretrainInfoNCEScriptGenerator(data_name=data_name, num_batches=num_batches,
                                                           save_dir=f"{save_dir}/infonce",
                                                           pre_max_epoch=pre_max_epoch, ft_max_epoch=ft_max_epoch)
        encoder_jobs, encoder_checkpoints, encoder_seeds = encoder_generator.grid_search_on(weight=1,
                                                                                            contrast_on=["partition"],
                                                                                            seed=seed)
        np.save(f"encoder_checkpoints_{data_name}.npy", np.asarray([encoder_jobs, encoder_checkpoints, encoder_seeds]))

        for j in [*baseline_jobs, *encoder_jobs]:
            submittor.account = next(account)
            submittor.run(j)
    else:

        [encoder_jobs, encoder_checkpoints, encoder_seeds] = np.load(f"encoder_checkpoints_{data_name}.npy",
                                                                     allow_pickle=True)

        for checkpoint, seed in zip(encoder_checkpoints, encoder_seeds):
            save_dir = Path(checkpoint).parents[1]
            pretrain_generator = PretrainDecoderScriptGenerator(data_name=data_name, num_batches=num_batches,
                                                                save_dir=save_dir,
                                                                pre_max_epoch=pre_max_epoch,
                                                                ft_max_epoch=ft_max_epoch, checkpoint_path=checkpoint)
            jobs = pretrain_generator.grid_search_on(decoder_weight=1, decoder_feature_names=["Up_conv3", "Up_conv2"],
                                                     seed=seed)

            for j in jobs:
                submittor.account = next(account)
                submittor.run(j)
