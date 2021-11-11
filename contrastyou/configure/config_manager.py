from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy as dcp
from functools import reduce, partial
from pprint import pprint
from typing import List, Tuple, Dict, Union, Optional

from deepclustering2.utils import path2Path, yaml_load
from loguru import logger

from ._merge_checker import merge_checker as _merge_checker
from .dictionary_utils import dictionary_merge_by_hierachy
from .yaml_parser import yamlArgParser
from ..types import typePath

__all__ = ["ConfigManger", "get_config"]

__config_dictionary__: OrderedDict = OrderedDict()


class ConfigManger:
    def __init__(self, *, base_path: str = None, optional_paths: Union[List[str], str] = None, verbose: bool = True,
                 strict: bool = False, _test_message=None) -> None:
        if isinstance(optional_paths, str):
            optional_paths = [optional_paths, ]

        self._parsed_args, parsed_config_path, parsed_optional_paths, parsed_extra_args_list = \
            yamlArgParser().parse(_test_message)
        self._base_path = parsed_config_path or base_path  # parsed config_path first
        self._optional_paths = parsed_optional_paths or optional_paths  # parsed first

        self._base_config, self._optional_config_list = self.load_yaml(verbose=False)

        self._parsed_args_merge_check = self.merge_check(strict=strict)
        from .dictionary_utils import remove_dictionary_callback
        self._merged_config = reduce(
            partial(dictionary_merge_by_hierachy, deepcopy=True, hook_after_merge=remove_dictionary_callback),
            [self._base_config, *self._optional_config_list, self._parsed_args]
        )
        if verbose:
            self.show_base_dict()
            self.show_opt_dict_list()
            self.show_parsed_dict()
            self.show_merged_dict()

    @staticmethod
    def _load_yaml(config_path: typePath, verbose=False):
        config_path_ = path2Path(config_path)
        assert config_path_.is_file(), config_path
        return yaml_load(config_path_, verbose=verbose)

    def load_yaml(self, verbose=False) -> Tuple[Dict, List[Dict]]:
        base_config = {}
        if self._base_path:
            base_config = self._load_yaml(self._base_path, verbose=verbose)
        optional_config_list = []
        if self._optional_paths:
            optional_config_list = [self._load_yaml(x, verbose=verbose) for x in self._optional_paths]
        return base_config, optional_config_list

    def merge_check(self, strict=True):
        try:
            _merge_checker(
                base_dictionary=reduce(partial(dictionary_merge_by_hierachy, deepcopy=True),
                                       [self._base_config, *self._optional_config_list]),
                coming_dictionary=self._parsed_args
            )
        except RuntimeError as e:
            if strict:
                logger.exception(e)
                raise e

    @contextmanager
    def __call__(self, config=None, scope="base"):
        assert scope not in __config_dictionary__, scope
        config = self.config if config is None else config
        __config_dictionary__[scope] = config
        yield config
        del __config_dictionary__[scope]

    @property
    def base_config(self):
        return dcp(self._base_config)

    @property
    def parsed_config(self):
        return dcp(self._parsed_args)

    @property
    def optional_configs(self):
        return dcp(self._optional_config_list)

    @property
    def merged_config(self):
        return dcp(self._merged_config)

    @property
    def config(self):
        return self.merged_config

    def show_base_dict(self):
        print("default dict from {}".format(self._base_path))
        pprint(self.base_config)

    def show_parsed_dict(self):
        print("parsed dict:")
        pprint(self.parsed_config)

    def show_opt_dict_list(self):
        print("optional dicts:")

        for i, d in enumerate(self._optional_config_list):
            print(f">>>>>>>>>>>{i} start>>>>>>>>>")
            pprint(d)
        print(f">>>>>>>>>> end >>>>>>>>>")

    def show_merged_dict(self):
        print("merged dict:")
        pprint(self.merged_config)

    @property
    def base_path(self) -> str:
        return str(self._base_path)

    @property
    def optional_paths(self) -> Optional[List[str]]:
        if self._optional_paths:
            return [str(x) for x in self._optional_paths]
        return None


def get_config(scope):
    return __config_dictionary__[scope]
