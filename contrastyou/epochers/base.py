import weakref
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from typing import Union, Dict, List

import torch
from deepclustering2.ddp.ddp import _DDPMixin
from torch import nn

from ..hooks.base import EpocherHook
from ..meters import MeterInterface
from ..meters.averagemeter import AverageValueListMeter
from ..mytqdm import tqdm


class EpocherBase(_DDPMixin, metaclass=ABCMeta):
    meter_focus = "tra"

    def __init__(self, *, model: nn.Module, num_batches: int, cur_epoch=0, device="cpu", **kwargs) -> None:
        super().__init__()
        self.__bind_trainer_done__ = False

        self._model = model
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._num_batches = num_batches
        self._cur_epoch = cur_epoch

        self.meters = MeterInterface(default_focus=self.meter_focus)
        self.configure_meters(self.meters)

        self.indicator = tqdm(range(self._num_batches), disable=not self.on_master())

        self._trainer = None

        self._hooks = []

    def add_hook(self, hook: EpocherHook):
        assert isinstance(hook, EpocherHook), hook
        self._hooks.append(hook)
        hook.set_epocher(self)

    def close_hooks(self):
        for h in self._hooks:
            h.close()

    def add_hooks(self, hooks: Union[List[EpocherHook], EpocherHook]):
        if isinstance(hooks, EpocherHook):
            hooks = [hooks, ]
        for h in hooks:
            self.add_hook(h)

    @property
    def device(self):
        return self._device

    @contextmanager
    def _register_indicator(self):
        assert isinstance(
            self._num_batches, int
        ), f"self._num_batches must be provided as an integer, given {self._num_batches}."

        self.indicator.set_desc_from_epocher(self)
        yield
        self.indicator.close()
        self.indicator.log_result()

    @contextmanager
    def _register_meters(self):
        meters = self.meters
        meters.reset()
        yield meters
        meters.join()

    @abstractmethod
    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("lr", AverageValueListMeter())
        return meters

    @abstractmethod
    def _run(self, **kwargs):
        raise NotImplementedError()

    def run(self, **kwargs):
        self.to(self.device)  # put all things into the same device
        with self._register_meters(), \
            self._register_indicator():
            run_result = self._run(**kwargs)
        self.close_hooks()
        return run_result

    def get_metric(self) -> Dict[str, Dict[str, float]]:
        return dict(self.meters.statistics())

    def get_score(self):
        raise NotImplementedError()

    def to(self, device: Union[torch.device, str] = torch.device("cpu")):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        for n, m in self.__dict__.items():
            if isinstance(m, nn.Module):
                m.to(device)
        self._device = device

    def set_trainer(self, trainer):
        self._trainer = weakref.proxy(trainer)
        self.__bind_trainer_done__ = True

    @property
    def trainer(self):
        if not self.__bind_trainer_done__:
            raise RuntimeError(f"{self.__class__.__name__} should call `set_trainer` first")
        return self._trainer

