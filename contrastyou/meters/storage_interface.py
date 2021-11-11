import functools
from abc import ABCMeta
from collections import defaultdict
from typing import List, Dict

import pandas as pd
from deepclustering2.utils import path2Path
from termcolor import colored

from .utils import HistoricalContainer, rename_df_columns

__all__ = ["Storage"]

from ..data.dataset.base import typePath


class Storage(metaclass=ABCMeta):
    def __init__(self, save_dir: typePath, csv_name="storage.csv") -> None:
        super().__init__()
        self.__storage = defaultdict(HistoricalContainer)
        self._csv_name = csv_name
        self._save_dir: str = str(save_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.to_csv()

    def put(
        self, name: str, value: Dict[str, float], epoch=None, prefix="", postfix=""
    ):
        self.__storage[prefix + name + postfix].add(value, epoch)

    def put_group(
        self, group_name: str, epoch_result: Dict, epoch=None, sep="/",
    ):
        assert isinstance(group_name, str), group_name
        if epoch_result:
            for k, v in epoch_result.items():
                self.put(group_name + sep + k, v, epoch)

    def add_from_meter_interface(self, *, epoch: int, **kwargs):
        for k, iterator in kwargs.items():
            for g, group_result in iterator.items():
                self.put_group(group_name=k + "/" + g, epoch_result=group_result, epoch=epoch)

    def get(self, name, epoch=None):
        assert name in self.__storage, name
        if epoch is None:
            return self.__storage[name]
        return self.__storage[name][epoch]

    def summary(self) -> pd.DataFrame:
        list_of_summary = [
            rename_df_columns(v.summary(), k, "/") for k, v in self.__storage.items()
        ]
        summary = []
        if len(list_of_summary) > 0:
            summary = functools.reduce(
                lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
                list_of_summary,
            )
        return pd.DataFrame(summary)

    @property
    def meter_names(self) -> List[str]:
        return list(self.__storage.keys())

    @property
    def storage(self):
        return self.__storage

    def state_dict(self):
        return self.__storage

    def load_state_dict(self, state_dict):
        self.__storage = state_dict
        print(colored(self.summary(), "green"))

    def to_csv(self):
        path = path2Path(self._save_dir)
        path.mkdir(exist_ok=True, parents=True)
        self.summary().to_csv(path / self._csv_name)
