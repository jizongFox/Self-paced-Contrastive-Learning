from abc import ABCMeta
from typing import Union
from warnings import warn

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as _lrschedulerType  # noqa
from torch.optim.optimizer import Optimizer as _optimizerType


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


class _ToMixin(metaclass=ABCMeta):
    _model: nn.Module

    def __init__(self, *, device: str, **kwargs) -> None:
        self._device: str = device
        super().__init__(**kwargs)

    def to(self, device: Union[str, torch.device], strict=True):
        self._device = str(device)

        error_message = []
        for module_name, module in self.__dict__.items():
            if isinstance(module, _optimizerType):
                optimizer_to(module, device)
                continue
            if isinstance(module, _lrschedulerType):
                scheduler_to(module, device)
                continue
            if hasattr(module, "to") and callable(module.to):
                try:
                    module.to(device=device)
                except Exception as e:
                    error_message.append(e)
                    continue
        if len(error_message) > 0:
            if strict is True:
                raise RuntimeError(
                    "Error(s) in to {} for {}:\n\t{}".format(
                        device, self.__class__.__name__, "\n\t".join([str(x) for x in error_message])))
            else:
                warn(RuntimeWarning(
                    "Error(s) in to {} for {}:\n\t{}".format(
                        device, self.__class__.__name__, "\n\t".join([str(x) for x in error_message]))))

    @property
    def device(self):
        return self._device
