import argparse
import os
from pathlib import Path
from typing import List, TypeVar

from deepclustering2.utils import path2Path

_path_type = TypeVar("_path_type", str, Path)


def find_experiment_list(root: _path_type) -> List[Path]:
    root_: Path = path2Path(root)
    config_paths = sorted(root_.rglob("*/config.yaml"))
    experiment_paths = [x.parent for x in config_paths]
    return experiment_paths


def is_experiment_sucessed(path: _path_type):
    path_: Path = path2Path(path)
    return (path_ / ".success").exists()


def remove_csv(path: _path_type):
    path_ = path2Path(path)
    csv_files = list(path_.glob("*.csv"))
    for c in csv_files:
        os.remove(c)


def main(root: _path_type):
    root_: Path = path2Path(root)
    assert root_.is_dir() and root_.exists(), root
    exp_list = find_experiment_list(root_)
    print(f"Found {len(exp_list)} experiments.")

    failed_exp_list = [x for x in exp_list if not is_experiment_sucessed(x)]
    print(f"Found {len(failed_exp_list)} failed experiments.")

    for exp in failed_exp_list:
        if not is_experiment_sucessed(exp):
            remove_csv(exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="root path")
    args = parser.parse_args()

    main(args.root)
