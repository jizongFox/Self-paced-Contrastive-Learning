import os
import re
from pathlib import Path
from typing import Tuple, List, Callable

import numpy as np
from torch import Tensor

from contrastyou.augment import SequentialWrapper
from contrastyou.data import ACDCDataset as _acdc, ProstateDataset as _prostate, mmWHSCTDataset as _mmct, \
    mmWHSMRDataset as _mmmr, ProstateMDDataset as _prostate_md
from contrastyou.data.dataset.base import get_stem
from .rearr import ContrastDataset


class ACDCDataset(ContrastDataset, _acdc):
    download_link = "https://drive.google.com/uc?id=1SMAS6R46BOafLKE9T8MDSVGAiavXPV-E"
    zip_name = "ACDC_contrast.zip"
    folder_name = "ACDC_contrast"
    partition_num = 3

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)
        self._acdc_info = np.load(os.path.join(self._root_dir, "acdc_info.npy"),
                                  allow_pickle=True).item()
        assert isinstance(self._acdc_info, dict) and len(self._acdc_info) == 200

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self._acdc_info[self._get_scan_name(filename)]  # noqa
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        if cur_index <= cutting_point - 1:
            return str(0)
        if cur_index <= 2 * cutting_point:
            return str(1)
        return str(2)

    def show_partitions(self) -> List[str]:
        return [self._get_partition(f) for f in next(iter(self.get_memory_dictionary().values()))]

    def show_scan_names(self) -> List[str]:
        return [self._get_scan_name(stem=get_stem(f)) for f in next(iter(self.get_memory_dictionary().values()))]


class ProstateDataset(ContrastDataset, _prostate):
    partition_num = 8

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)
        self._prostate_info = np.load(os.path.join(self._root_dir, "prostate_info.npy"), allow_pickle=True).item()
        assert isinstance(self._prostate_info, dict) and len(self._prostate_info) == 50

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self._prostate_info[self._get_scan_name(filename)]
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))

    def show_partitions(self) -> List[str]:
        return [self._get_partition(f) for f in next(iter(self.get_memory_dictionary().values()))]

    def show_scan_names(self) -> List[str]:
        return [self._get_scan_name(stem=get_stem(f)) for f in next(iter(self.get_memory_dictionary().values()))]


class ProstateMDDataset(ContrastDataset, _prostate_md):
    partition_num = 4

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)
        self._prostate_info = np.load(os.path.join(self._root_dir, "prostate_info.npy"), allow_pickle=True).item()
        assert isinstance(self._prostate_info, dict) and len(self._prostate_info) == 32

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self._prostate_info[self._get_scan_name(filename)]
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))

    def show_partitions(self) -> List[str]:
        return [self._get_partition(f) for f in next(iter(self.get_memory_dictionary().values()))]

    def show_scan_names(self) -> List[str]:
        return [self._get_scan_name(stem=get_stem(f)) for f in next(iter(self.get_memory_dictionary().values()))]


class _mmWHSBase(ContrastDataset):
    partition_num = 8
    _get_scan_name: Callable[[str], str]

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)

        self._meta_info = {"ct": np.load(str(Path(root_dir, "MMWHS", "meta_ct.npy")), allow_pickle=True).tolist(),
                           "mr": np.load(str(Path(root_dir, "MMWHS", "meta_mr.npy")), allow_pickle=True).tolist()}

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)  # noqa
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self.get_meta()[self._get_scan_name(filename)]
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))

    def show_partitions(self) -> List[str]:
        return [self._get_partition(f) for f in next(iter(self.get_memory_dictionary().values()))]

    def show_scan_names(self) -> List[str]:
        return [self._get_scan_name(get_stem(f)) for f in next(iter(self.get_memory_dictionary().values()))]

    @property
    def metainfo_ct(self):
        return self._meta_info["ct"]

    @property
    def metainfo_mr(self):
        return self._meta_info["mr"]

    def get_meta(self):
        ...


class mmWHSMRDataset(_mmWHSBase, _mmmr):

    def get_meta(self):
        return self.metainfo_mr


class mmWHSCTDataset(_mmWHSBase, _mmct):
    def get_meta(self):
        return self.metainfo_ct
