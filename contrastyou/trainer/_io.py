import shutil
import tempfile
import warnings
from abc import ABCMeta
from pathlib import Path
from typing import Dict

import torch
from deepclustering2.utils import path2Path, yaml_write

from ._buffer import _BufferMixin
from ..types import typePath


def safe_save(checkpoint_dictionary, save_path):
    try:
        with tempfile.TemporaryFile() as tmp:
            torch.save(checkpoint_dictionary, str(tmp))
            shutil.move(str(tmp), str(save_path))
    finally:
        return


def create_save_dir(self, save_dir: str):
    save_dir = str(save_dir)
    if not Path(save_dir).is_absolute():
        save_dir = str(Path(self.RUN_PATH) / save_dir)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    return save_dir


class _IOMixin(_BufferMixin, metaclass=ABCMeta):
    RUN_PATH: str
    """This mixin provides interface on critical parameter definition,
    saving and loading trainer, and configure saving  """

    def __init__(self, *, save_dir: str, max_epoch: int, num_batches: int, **kwargs) -> None:
        super().__init__()

        self._save_dir = create_save_dir(self, save_dir)

        self._max_epoch = max_epoch
        self._num_batches = num_batches

        self._best_score: float
        self._start_epoch: int
        self._cur_epoch: int

        self._register_buffer("_config", None)
        self._register_buffer("_best_score", 0)
        self._register_buffer("_start_epoch", 0)
        self._register_buffer("_cur_epoch", 0)

    def dump_config(self, config, path: typePath = None, save_name="config.yaml"):
        path_ = self._save_dir
        if path:
            path_ = path2Path(path)
            if not path_.is_absolute():
                path_ = Path(self.RUN_PATH) / path_
        yaml_write(config, str(path_), save_name=save_name)

    def state_dict(self, **kwargs) -> dict:
        buffer_state_dict = super(_IOMixin, self).state_dict()
        local_modules = {k: v for k, v in self.__dict__.items() if k != "_buffers"}

        local_state_dict = {}
        for module_name, module in local_modules.items():
            if hasattr(module, "state_dict") and callable(module.state_dict):
                local_state_dict[module_name] = module.state_dict()
        destination = {**local_state_dict, **{"_buffers": buffer_state_dict}}
        return destination

    def load_state_dict(self, state_dict: dict, strict=True) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :param strict: if raise error
        :return:
        """
        missing_keys = []
        er_msgs = []

        for module_name, module in self.__dict__.items():
            if module_name == "_buffers":
                super(_IOMixin, self).load_state_dict(state_dict["_buffers"])
                continue

            if hasattr(module, "load_state_dict") and callable(
                getattr(module, "load_state_dict", None)
            ):
                try:
                    module.load_state_dict(state_dict[module_name])
                except KeyError:
                    missing_keys.append(module_name)
                except Exception as ex:
                    er_msgs.append(
                        "while copying {} parameters, "
                        "error {} occurs".format(module_name, ex)
                    )
        if len(er_msgs) > 0:
            if strict is True:
                raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(er_msgs)))
            else:
                warnings.warn(RuntimeWarning("Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(er_msgs))))

    def load_state_dict_from_path(self, path: str, name="last.pth", strict=True, ) -> None:
        path_ = path2Path(path)
        assert path_.exists(), path
        if path_.is_file() and path_.suffix in (".pth", ".pt"):
            path_ = path_
        elif path_.is_dir() and (path_ / name).exists():
            path_ = path_ / name
        else:
            raise FileNotFoundError(path_)
        state_dict = torch.load(str(path_), map_location="cpu")
        self.load_state_dict(state_dict, strict)

    def save_to(self, *, save_dir: str = None, save_name: str):
        assert path2Path(save_name).suffix in (".pth", ".pt"), path2Path(save_name).suffix
        if save_dir is None:
            save_dir = self._save_dir

        save_dir_ = path2Path(save_dir)
        save_dir_.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict()
        safe_save(state_dict, str(save_dir_ / save_name))

    def resume_from_checkpoint(self, checkpoint: Dict[str, Dict], strict=True):
        self.load_state_dict(checkpoint, strict=strict)

    def resume_from_path(self, path: str, name="last.pth", strict=True, ):
        return self.load_state_dict_from_path(str(path), name, strict)
