from typing import Tuple

from torch import nn, Tensor
from torch.nn import functional as F, Module
from torch.nn.modules.utils import _pair


class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, *_ = features.shape
        return features.view(b, -1)


class SoftmaxWithT(nn.Softmax):

    def __init__(self, dim, T: float = 1.0) -> None:
        super().__init__(dim)
        self._T = T

    def forward(self, input: Tensor) -> Tensor:
        input /= self._T
        return super().forward(input)


class Normalize(nn.Module):

    def __init__(self, dim=1) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self._dim)


class Identical(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return input


def _check_head_type(head_type):
    return head_type in ("mlp", "linear")


def _check_pool_name(pool_name):
    return pool_name in ("adaptive_avg", "adaptive_max", "identical", "none")


def get_pool_component(pool_name, spatial_size: Tuple[int, int]):
    return {
        "adaptive_avg": nn.AdaptiveAvgPool2d(spatial_size),
        "adaptive_max": nn.AdaptiveMaxPool2d(spatial_size),
        None: Identical(),
        "none": Identical(),
        "identical": Identical(),
    }[pool_name]


class _ProjectorHeadBase(nn.Module):

    def __init__(self, *, input_dim: int, output_dim: int, head_type: str, normalize: bool, pool_name="adaptive_avg",
                 spatial_size=(1, 1)):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        assert _check_head_type(head_type=head_type)
        self._head_type = head_type
        self._normalize = normalize
        assert _check_pool_name(pool_name=pool_name)
        self._pool_name = pool_name
        self._spatial_size = _pair(spatial_size)

        self._pooling_module = get_pool_component(self._pool_name, self._spatial_size)

    def _record_message(self):
        return f"Initializing {self.__class__.__name__} with {self._head_type} dense head " \
               f"({self._input_dim}:{self._output_dim}), " \
               f"{' normalization ' if self._normalize else ''}" \
               f"{f'{self._pool_name} with {self._spatial_size}' if 'adaptive' in self._pool_name else ''} "


class ModuleDict(nn.ModuleDict):
    """
    A module dict from pytorch that prevents override parameters
    """

    def __setitem__(self, key: str, module: Module) -> None:
        if key in self.keys():
            import warnings
            warnings.warn(RuntimeWarning(f"force overriding {key}"))
        super().__setitem__(key, module)
