import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any

import numpy as np
from deepclustering2.utils import gethash, write_yaml

from contrastyou import on_cc
from semi_seg import __accounts


# CC things


def run_jobs(job_submiter, job_array, args):
    def move_dataset():
        if on_cc():
            from contrastyou import DATA_PATH
            return f" find {DATA_PATH}  " + "-name '*.zip' -exec cp {} $SLURM_TMPDIR \;"
        return ""

    for j in job_array:
        job_submiter.prepare_env(
            [
                "module load python/3.8.2 ",
                f"source ~/venv/bin/activate ",
                'if [ $(which python) == "/usr/bin/python" ]',
                "then",
                "exit 1314520",
                "fi",

                "export OMP_NUM_THREADS=1",
                "export PYTHONOPTIMIZE=1",
                "export PYTHONWARNINGS=ignore ",
                "export CUBLAS_WORKSPACE_CONFIG=:16:8 ",
                move_dataset()
            ]
        )
        job_submiter.account = next(accounts)
        print(j)
        if not args.show_cmd:
            code = job_submiter.run(j)
            if code != 0:
                if job_submiter._on_local:
                    exit(code)
                if not on_cc():
                    exit(code)
                pass


def account_iterable(name_list):
    while True:
        for i in np.random.permutation(name_list):
            yield i


accounts = account_iterable(__accounts)

__git_hash__ = gethash(__file__)


def _assert_equality(feature_name, importance):
    assert len(feature_name) == len(importance), (feature_name, importance)


class _BindOptions:

    def __init__(self) -> None:
        super().__init__()
        self.OptionalScripts = []

    @staticmethod
    def bind(subparser):
        ...

    def parse(self, args):
        ...

    def add(self, string):
        self.OptionalScripts.append(string)

    def get_option_str(self):
        return " ".join(self.OptionalScripts)


class BindPretrainFinetune(_BindOptions):
    @staticmethod
    def bind(subparser):
        subparser.add_argument("--pre_lr", default="null", type=str, help="pretrain learning rate")
        subparser.add_argument("--ft_lr", default="null", type=str, help="finetune learning rate")
        subparser.add_argument("-pe", "--pre_max_epoch", type=str, default="null", help="pretrain max_epoch")
        subparser.add_argument("-fe", "--ft_max_epoch", type=str, default="null", help="finetune max_epoch")

    def parse(self, args):
        pre_lr = args.pre_lr
        self.add(f"Optim.pre_lr={pre_lr}")
        ft_lr = args.ft_lr
        self.add(f"Optim.ft_lr={ft_lr}")
        pre_max_epoch = args.pre_max_epoch
        ft_max_epoch = args.ft_max_epoch
        self.add(f"Trainer.pre_max_epoch={pre_max_epoch}")
        self.add(f"Trainer.ft_max_epoch={ft_max_epoch}")


class BindContrastive(_BindOptions):
    @staticmethod
    def bind(subparser):
        subparser.add_argument("-g", "--group_sample_num", default=6, type=int)
        subparser.add_argument("--global_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                               default=["Conv5"], type=str, help="global_features")
        subparser.add_argument("--global_importance", nargs="+", type=float, default=[1.0, ], help="global importance")

        subparser.add_argument("--contrast_on", "-c", nargs="+", type=str, required=True,
                               choices=["partition", "cycle", "patient", "self"])
        subparser.add_argument("--monitor", default=False, type=str, choices=["true", "false"],
                               help="monitoring the infocne")

    def parse(self, args):
        self.add(f"ContrastiveLoaderParams.scan_sample_num={args.group_sample_num}")
        _assert_equality(args.global_features, args.global_importance)

        self.add(f"ProjectorParams.GlobalParams.feature_names=[{','.join(args.global_features)}]")
        self.add(
            f"ProjectorParams.GlobalParams.feature_importance=[{','.join([str(x) for x in args.global_importance])}]")
        self.add(f"ProjectorParams.LossParams.contrast_on=[{','.join(args.contrast_on)}]")
        self.add(f"Trainer.monitor={args.monitor}")


class BindSelfPaced(_BindOptions):
    @staticmethod
    def bind(subparser):
        subparser.add_argument("--begin_value", default=[1000], type=float, nargs="+",
                               help="ProjectorParams.LossParams.begin_value")
        subparser.add_argument("--end_value", default=[1000], type=float, nargs="+",
                               help="ProjectorParams.LossParams.end_value")
        subparser.add_argument("--method", default="hard", type=str, nargs="+",
                               help="ProjectorParams.LossParams.weight_update")
        subparser.add_argument("--scheduler_type", default=["inversesquare"], type=str,
                               choices=["linear", "square", "inversesquare"], nargs="+",
                               help="ProjectorParams.LossParams.type")

    def parse(self, args):
        self.add(f"ProjectorParams.LossParams.begin_value=[{','.join([str(x) for x in args.begin_value])}]")
        self.add(f"ProjectorParams.LossParams.end_value=[{','.join([str(x) for x in args.end_value])}]")
        self.add(f"ProjectorParams.LossParams.weight_update=[{','.join(args.method)}]")
        self.add(f"ProjectorParams.LossParams.type=[{','.join(args.scheduler_type)}]")


class BindSemiSupervisedLearning(_BindOptions):
    pass


@contextmanager
def dump_config(config: Dict[str, Any]):
    import string
    import random
    import os
    tmp_path = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) + ".yaml"
    Path("./.tmp").mkdir(parents=True, exist_ok=True)
    write_yaml(config, save_dir="./.tmp", save_name=tmp_path, force_overwrite=True)
    tmp_path = os.path.abspath("./.tmp/" + tmp_path)
    yield tmp_path

    def remove():
        os.remove(tmp_path)

    # import atexit
    # atexit.register(remove)
