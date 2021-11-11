from pathlib import Path
from typing import TypeVar, Union, Tuple, List

import collections
from collections.abc import Mapping, Iterable
from typing import TypeVar, Callable

from numpy import ndarray
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import _BaseDataLoaderIter, DataLoader

mapType = Mapping
T = TypeVar("T")
typePath = TypeVar("typePath", str, Path)
typeNumeric = TypeVar("typeNumeric", int, float, Tensor, ndarray)
genericLoaderType = TypeVar("genericLoaderType", _BaseDataLoaderIter, DataLoader)
dataIterType = TypeVar("dataIterType", _BaseDataLoaderIter, Iterable)
optimizerType = Optimizer
criterionType = Callable[[Tensor, Tensor], Tensor]


def is_map(value):
    return isinstance(value, collections.abc.Mapping)


def is_path(value):
    return isinstance(value, (str, Path))


def is_numeric(value):
    return isinstance(value, (int, float, Tensor, ndarray))


# Create some useful type aliases

# Template for arguments which can be supplied as a tuple, or which can be a scalar which PyTorch will internally
# broadcast to a tuple.
# Comes in several variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d operations.

_tuple_any_t = Tuple[T, ...]
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]
_scalar_or_tuple_4_t = Union[T, Tuple[T, T, T, T]]
_scalar_or_tuple_5_t = Union[T, Tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t = Union[T, Tuple[T, T, T, T, T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]
_size_4_t = _scalar_or_tuple_4_t[int]
_size_5_t = _scalar_or_tuple_5_t[int]
_size_6_t = _scalar_or_tuple_6_t[int]

# For arguments that represent a ratio to adjust each dimension of an input with (eg, upsampling parameters)
_ratio_2_t = _scalar_or_tuple_2_t[float]
_ratio_3_t = _scalar_or_tuple_3_t[float]
_ratio_any_t = _scalar_or_tuple_any_t[float]

_tensor_list_t = _scalar_or_tuple_any_t[Tensor]

# for string
_string_salar_or_tuple = _scalar_or_tuple_any_t[str]
_string_tuple = _tuple_any_t[str]
