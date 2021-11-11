import os
import re
from collections import OrderedDict
from copy import deepcopy as dcopy
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, TypeVar, OrderedDict as OrderedDictType

from PIL import Image, ImageFile
from deepclustering2.utils import path2Path
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from ...augment import SequentialWrapper
from ...augment.pil_augment import ToTensor, ToLabel

__all__ = ["DatasetBase", "extract_sub_dataset_based_on_scan_names", "get_stem"]

ImageFile.LOAD_TRUNCATED_IMAGES = True

typePath = TypeVar("typePath", str, Path)


def read_image(path, mode):
    with Image.open(path) as image:
        return image.convert(mode)


def allow_extension(path: str, extensions: List[str]) -> bool:
    try:
        return Path(path).suffixes[0] in extensions
    except:  # noqa
        return False


def default_transform() -> SequentialWrapper:
    return SequentialWrapper(
        image_transform=ToTensor(),
        target_transform=ToLabel(),
    )


def get_stem(path: typePath):
    path_ = path2Path(path)
    return path_.stem


def check_folder_types(type_: str):
    assert type_.lower() in ("image", "img", "gt", "label"), type_


def is_image_folder(type_: str):
    if type_.lower() in ("image", "img"):
        return True
    return False


def make_memory_dictionary(root: str, mode: str, folders: List[str], extensions) -> OrderedDictType[str, List[str]]:
    for subfolder in folders:
        assert (Path(root, mode, subfolder).exists() and Path(root, mode, subfolder).is_dir()), \
            os.path.join(root, mode, subfolder)

    items = [os.listdir(Path(os.path.join(root, mode, sub))) for sub in folders]
    cleaned_items = [sorted([x for x in item if allow_extension(x, extensions)]) for item in items]

    memory = OrderedDict()
    for subfolder, item in zip(folders, cleaned_items):
        memory[subfolder] = sorted([os.path.join(root, mode, subfolder, x_path) for x_path in item])

    sub_memory_len_list = [len(x) for x in memory.values()]
    assert len(set(sub_memory_len_list)) == 1, sub_memory_len_list
    return memory


class DatasetBase(Dataset):
    allow_extension = [".jpg", ".png"]

    def __init__(self, *, root_dir: str, mode: str, sub_folders: Union[List[str], str],
                 sub_folder_types: Union[List[str], str], transforms: SequentialWrapper = None,
                 group_re: str = None) -> None:
        """
        :param root_dir: dataset root
        :param mode: train or test mode
        :param sub_folders: the folder list inside train or test folder
        :param transforms: SequentialWrapper transformer
        :param group_re: regex to group scans
        """
        self._name: str = f"{self.__class__.__name__}-{mode}"
        self._mode: str = mode
        self._root_dir: str = root_dir
        Path(self._root_dir).mkdir(parents=True, exist_ok=True)

        self._sub_folders: List[str] = [sub_folders, ] if isinstance(sub_folders, str) else sub_folders
        sub_folder_types = [sub_folder_types, ] if isinstance(sub_folder_types, str) else sub_folder_types
        assert len(self._sub_folders) == len(sub_folder_types)
        for type_ in sub_folder_types:
            check_folder_types(type_)
        self._sub_folder_types = [is_image_folder(type_) for type_ in sub_folder_types]

        self._transforms = transforms if transforms else default_transform()

        logger.opt(depth=1).trace(f"Creating {self.__class__.__name__}")
        self._memory = self.set_memory_dictionary(
            make_memory_dictionary(self._root_dir, self._mode, self._sub_folders, self.allow_extension)
        )
        # pre-load
        self._is_preload = False
        self._preload_storage: OrderedDict = OrderedDict()

        # regex for scan
        self._pattern = group_re
        self._re_pattern = None

        if self._pattern:
            self._re_pattern = re.compile(self._pattern)

    def get_memory_dictionary(self) -> Dict[str, List[str]]:
        return OrderedDict({k: v for k, v in self._memory.items()})

    def set_memory_dictionary(self, new_dictionary: Dict[str, Any], deepcopy=True):
        assert isinstance(new_dictionary, dict)
        self._memory = dcopy(new_dictionary) if deepcopy else new_dictionary
        return self._memory

    @property
    def pattern(self):
        return self._pattern

    @property
    def mode(self) -> str:
        return self._mode

    def __len__(self) -> int:
        return int(len(self._memory[self._sub_folders[0]]))

    def __getitem__(self, index) -> Tuple[List[Tensor], str]:
        image_list, filename_list = self._getitem_index(index)
        filename = Path(filename_list[0]).stem

        images = [x for x, t in zip(image_list, self._sub_folder_types) if t]
        labels = [x for x, t in zip(image_list, self._sub_folder_types) if not t]

        images_, labels_ = self._transforms(images, labels)
        del images, labels
        return [*images_, *labels_], filename

    def _getitem_index(self, index):
        image_list = self._preload_storage[index] if self._is_preload else \
            [read_image(self._memory[subfolder][index], "L") for subfolder in self._sub_folders]

        filename_list = [self._memory[subfolder][index] for subfolder in self._sub_folders]

        stem_set = set([get_stem(x) for x in filename_list])
        assert len(stem_set) == 1, stem_set
        del stem_set

        return image_list.copy(), filename_list.copy()

    def _preload(self):
        logger.opt(depth=1).trace(f"preloading {len(self.get_scan_list())} {self.__class__.__name__} data ...")

        for index in tqdm(range(len(self)), total=len(self), disable=True):
            self._preload_storage[index] = \
                [read_image(self._memory[subfolder][index], "L") for subfolder in self._sub_folders]

    def preload(self):
        self._is_preload = True
        self._preload()

    def deload(self):
        self._is_preload = False
        del self._preload_storage
        self._preload_storage = OrderedDict()

    def is_preloaded(self) -> bool:
        return self._is_preload

    def _get_scan_name(self, stem: str) -> str:
        if self._re_pattern is None:
            raise RuntimeError("Putting group_re first, instead of None")
        try:
            group_name = self._re_pattern.search(stem).group(0)  # type: ignore
        except AttributeError:
            raise AttributeError(f"Cannot match pattern: {self._pattern} for {str(stem)}")
        return group_name

    def get_stem_list(self):
        return [get_stem(x) for x in self._memory[self._sub_folders[0]]]

    def get_scan_list(self):
        return sorted(set([self._get_scan_name(filename) for filename in self.get_stem_list()]))

    @property
    def transforms(self) -> SequentialWrapper:
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: SequentialWrapper):
        assert isinstance(transforms, SequentialWrapper), type(transforms)
        self._transforms = transforms


def extract_sub_dataset_based_on_scan_names(dataset: DatasetBase, group_names: List[str],
                                            transforms: SequentialWrapper = None) -> DatasetBase:
    loaded: bool = dataset.is_preloaded()
    available_group_names = sorted(set(dataset.get_scan_list()))
    for g in group_names:
        assert g in available_group_names, (g, available_group_names)
    memory = dataset.get_memory_dictionary()
    get_scan_name = dataset._get_scan_name  # noqa
    new_memory = OrderedDict()
    for sub_folder, path_list in memory.items():
        new_memory[sub_folder] = [x for x in path_list if get_scan_name(stem=get_stem(x)) in group_names]

    if loaded:
        dataset.deload()

    new_dataset = dcopy(dataset)
    new_dataset.set_memory_dictionary(new_dictionary=new_memory)
    if transforms:
        new_dataset.transforms = transforms
    assert set(new_dataset.get_scan_list()) == set(group_names)

    if loaded:
        new_dataset.preload()
    return new_dataset
