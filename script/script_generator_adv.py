import argparse
import os
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

from contrastyou import PROJECT_PATH, on_cc, __git_hash__
from contrastyou.configure import dictionary_merge_by_hierachy
from script import utils
from script.utils import TEMP_DIR, yaml_load, write_yaml, grid_search, BaselineGenerator, move_dataset
from semi_seg import __accounts, ratio_zoo, num_batches_zoo, pre_max_epoch_zoo, ft_max_epoch_zoo

account = cycle(__accounts)
opt_hook_path = {
    "adv": "config/hooks/adv.yaml"
}

git_hash = __git_hash__[:6] if __git_hash__ is not None else "none"


class AdversarialScriptGenerator(BaselineGenerator):

    def __init__(self, *, data_name, num_batches, max_epoch, save_dir, model_checkpoint=None) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, max_epoch=max_epoch, save_dir=save_dir,
                         model_checkpoint=model_checkpoint)

        self.hook_config = yaml_load(PROJECT_PATH + "/" + opt_hook_path[self.get_hook_name()])

    def get_hook_name(self):
        return "adv"

    def get_hook_params(self, reg_weight, dis_consider_image):
        return {"Trainer": {"reg_weight": reg_weight, "dis_consider_image": dis_consider_image}}

    def generate_single_script(self, save_dir, labeled_scan_num, seed, hook_path):
        from semi_seg import ft_lr_zooms
        ft_lr = ft_lr_zooms[self._data_name]

        return f"python main_adv.py  Trainer.save_dir={save_dir} " \
               f" Optim.lr={ft_lr:.7f} RandomSeed={str(seed)} Data.labeled_scan_num={int(labeled_scan_num)} " \
               f" {' '.join(self.conditions)} " \
               f" --opt-path {hook_path}"

    def grid_search_on(self, *, seed, **kwargs):
        jobs = []

        labeled_scan_list = ratio_zoo[self._data_name][:-1] if len(ratio_zoo[self._data_name]) > 1 else ratio_zoo[
            self._data_name]

        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            hook_params = self.get_hook_params(**param)
            sub_save_dir = self._get_hyper_param_string(**param)
            merged_config = dictionary_merge_by_hierachy(self.hook_config, hook_params)
            config_path = write_yaml(merged_config, save_dir=TEMP_DIR, save_name=utils.random_string() + ".yaml")
            true_save_dir = os.path.join(self._save_dir, "Seed_" + str(random_seed), sub_save_dir)

            job = " && ".join(
                [self.generate_single_script(save_dir=os.path.join(true_save_dir, "tra", f"labeled_scan_{l:02d}"),
                                             seed=random_seed, hook_path=config_path, labeled_scan_num=l)
                 for l in labeled_scan_list])

            jobs.append(job)
        return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser("adv method")
    parser.add_argument("--data-name", required=True, type=str, help="dataset_name",
                        choices=["acdc", "prostate", "mmwhsct"])
    parser.add_argument("--save_dir", required=True, type=str, help="save_dir")
    args = parser.parse_args()

    submittor = JobSubmiter(on_local=not on_cc(), project_path="../", time=6)
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
        "export LOGURU_LEVEL=TRACE",
        move_dataset()
    ])
    seed = [10, 20, 30]
    data_name = args.data_name
    save_dir = f"{args.save_dir}/adv/hash_{git_hash}/{data_name}"
    num_batches = num_batches_zoo[data_name]
    pre_max_epoch = pre_max_epoch_zoo[data_name]
    ft_max_epoch = ft_max_epoch_zoo[data_name]

    baseline_generator = BaselineGenerator(data_name=data_name, num_batches=num_batches, max_epoch=ft_max_epoch,
                                           save_dir=os.path.join(save_dir, "baseline"))
    b_jobs = baseline_generator.grid_search_on(seed=seed)

    script_generator = AdversarialScriptGenerator(data_name=data_name, save_dir=os.path.join(save_dir, "semi"),
                                                  num_batches=num_batches,
                                                  max_epoch=ft_max_epoch)

    b_jobs2 = script_generator.grid_search_on(reg_weight=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                                              dis_consider_image=[True, False], seed=seed)
    print(b_jobs)
    print(b_jobs2)
    for j in [*b_jobs, *b_jobs2]:
        submittor.account = next(account)
        submittor.run(j)
