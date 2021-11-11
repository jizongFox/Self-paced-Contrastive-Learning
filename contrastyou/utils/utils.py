# dictionary helper functions
import collections
import functools
import os
import random
import warnings
from contextlib import contextmanager
from itertools import repeat
from typing import Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch._six import container_abcs
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter  # noqa


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_dataset(dataloader):
    if isinstance(dataloader, _BaseDataLoaderIter):
        return dataloader._dataset
    elif isinstance(dataloader, DataLoader):
        return dataloader.dataset
    else:
        raise NotImplementedError(type(dataloader))


# make a flatten dictionary to be printablely nice.
def nice_dict(input_dict: Dict[str, Union[int, float]]) -> str:
    """
    this function is to return a nice string to dictionary displace propose.
    :param input_dict: dictionary
    :return: string
    """
    assert isinstance(
        input_dict, dict
    ), f"{input_dict} should be a dict, given {type(input_dict)}."
    is_flat_dict = True
    for k, v in input_dict.items():
        if isinstance(v, dict):
            is_flat_dict = False
            break
    flat_dict = input_dict if is_flat_dict else flatten_dict(input_dict, sep="")
    string_list = [f"{k}:{v:.3f}" for k, v in flat_dict.items()]
    return ", ".join(string_list)


def average_iter(a_list):
    return sum(a_list) / float(len(a_list))


def multiply_iter(iter_a, iter_b):
    return [x * y for x, y in zip(iter_a, iter_b)]


def weighted_average_iter(a_list, weight_list):
    sum_weight = sum(weight_list) + 1e-16
    return sum(multiply_iter(a_list, weight_list)) / sum_weight


def pairwise_distances(x, y=None, recall_func=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
           recall function is a function to manipulate the distance.
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    if recall_func:
        return recall_func(dist)
    return dist


@contextmanager
def plt_interactive():
    plt.ion()
    yield
    plt.ioff()


def extract_model_state_dict(trainer_checkpoint_path: str):
    trainer_state = torch.load(trainer_checkpoint_path, map_location="cpu")

    return trainer_state["_model"]


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


# reproducibility
def set_deterministic(enable=True):
    torch.backends.cudnn.benchmark = not enable  # noqa
    try:
        torch.use_deterministic_algorithms(enable)
    except:
        try:
            torch.set_deterministic(enable)
        finally:
            return


def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def fix_all_seed_for_transforms(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    fix_all_seed(seed)
    yield
    random.setstate(random_state)
    np.random.set_state(np_state)  # noqa
    torch.random.set_rng_state(torch_state)  # noqa


@contextmanager
def fix_all_seed_within_context(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_support = torch.cuda.is_available()
    if cuda_support:
        torch_cuda_state = torch.cuda.get_rng_state()
        torch_cuda_state_all = torch.cuda.get_rng_state_all()
    fix_all_seed(seed)

    yield
    random.setstate(random_state)
    np.random.set_state(np_state)  # noqa
    torch.random.set_rng_state(torch_state)  # noqa
    if cuda_support:
        torch.cuda.set_rng_state(torch_cuda_state)
        torch.cuda.set_rng_state_all(torch_cuda_state_all)


def ntuple(n):
    def parse(x):
        if isinstance(x, str):
            return tuple(repeat(x, n))
        if isinstance(x, container_abcs.Iterable):
            x = list(x)
            if len(x) == 1:
                return tuple(repeat(x[0], n))
            else:
                if len(x) != n:
                    raise RuntimeError(f"inconsistent shape between {x} and {n}")
            return x

        return tuple(repeat(x, n))

    return parse


_single = ntuple(1)
_pair = ntuple(2)
_triple = ntuple(3)
_quadruple = ntuple(4)


def config_logger(save_dir):
    abs_save_dir = os.path.abspath(save_dir)
    from loguru import logger
    logger.add(os.path.join(abs_save_dir, "loguru.log"), level="TRACE", diagnose=True)


def fix_seed(func):
    functools.wraps(func)

    def func_wrapper(*args, **kwargs):
        with fix_all_seed_within_context(1):
            return func(*args, **kwargs)

    return func_wrapper
