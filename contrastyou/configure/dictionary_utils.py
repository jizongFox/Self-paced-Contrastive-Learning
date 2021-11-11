from collections import Iterable
from copy import deepcopy as dcopy
from typing import Dict, Any

from contrastyou.mytqdm.utils import is_iterable
from contrastyou.types import mapType, is_map


def dictionary_merge_by_hierachy(dictionary1: Dict[str, Any], new_dictionary: Dict[str, Any] = None, deepcopy=True,
                                 hook_after_merge=None):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into``dct``.
    :return: None
    """
    if deepcopy:
        dictionary1, new_dictionary = dcopy(dictionary1), dcopy(new_dictionary)
    if new_dictionary is None:
        return dictionary1
    for k, v in new_dictionary.items():
        if k in dictionary1 and isinstance(dictionary1[k], mapType) and isinstance(new_dictionary[k], mapType):
            dictionary1[k] = dictionary_merge_by_hierachy(dictionary1[k], new_dictionary[k], deepcopy=False)
        else:
            dictionary1[k] = new_dictionary[k]
    if hook_after_merge:
        dictionary1 = hook_after_merge(dictionary1)
    return dictionary1


def remove_dictionary_callback(dictionary, key="remove"):
    new_dictionary = dcopy(dictionary)
    for k, v in dictionary.items():
        if isinstance(v, mapType):
            new_dictionary[k] = remove_dictionary_callback(v, key)
        try:
            if v.lower() == key:
                del new_dictionary[k]
        except AttributeError:
            pass
    return new_dictionary


def extract_dictionary_from_anchor(target_dictionary: Dict, anchor_dictionary: Dict, deepcopy=True, prune_anchor=False):
    result_dict = {}

    if deepcopy:
        anchor_dictionary, target_dictionary = list(map(dcopy, [anchor_dictionary, target_dictionary]))

    for k, v in anchor_dictionary.items():
        if k in target_dictionary:
            if not isinstance(v, mapType):
                result_dict[k] = target_dictionary[k]
            else:
                result_dict[k] = extract_dictionary_from_anchor(target_dictionary[k], anchor_dictionary[k],
                                                                deepcopy=deepcopy, prune_anchor=prune_anchor)
        elif not prune_anchor:
            result_dict[k] = anchor_dictionary[k]

    return result_dict


def flatten_dict(dictionary, parent_key="", sep="_"):
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if is_map(v):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dictionary2string(dictionary, parent_name_list=None, item_list=None):
    def tostring(item):
        if isinstance(item, (float,)):
            return f"{item:.7f}"
        return str(item)

    if parent_name_list is None:
        parent_name_list = []
    if item_list is None:
        item_list = []
    for k, v in dictionary.items():
        if is_map(v):
            dictionary2string(v, parent_name_list=parent_name_list + [k], item_list=item_list)
        elif isinstance(v, Iterable) and (not isinstance(v, str)):
            current_item = ".".join(parent_name_list) + f".{k}=[{','.join([tostring(x) for x in v])}]"
            item_list.append(current_item)
        else:
            current_item = ".".join(parent_name_list) + f".{k}={tostring(v)}"
            item_list.append(current_item)
    return " ".join(item_list)


def extract_params_with_key_prefix(dictionary: Dict[str, Any], prefix: str) -> Dict:
    result_dict = {}
    for k, v in dictionary.items():
        if is_map(v):
            result_dict[k] = extract_params_with_key_prefix(v, prefix=prefix)
        elif is_iterable(v):
            result_dict[k] = [extract_params_with_key_prefix(x, prefix=prefix) for x in v]
        else:
            if k.startswith(prefix):
                result_dict[k.replace(prefix, "")] = v

        # clean items with {}
        for _k, _v in result_dict.copy().items():
            if _v == {}:
                del result_dict[_k]
    return result_dict