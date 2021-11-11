from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Any

import torch
from deepclustering2 import optim
from deepclustering2.ddp.ddp import _DDPMixin  # noqa
from deepclustering2.schedulers import GradualWarmupScheduler
from loguru import logger
from torch import nn

from contrastyou import PROJECT_PATH
from ._functional import _ToMixin
from ._io import _IOMixin
from ..epochers.base import EpocherBase
from ..hooks.base import TrainerHook
from ..meters import Storage
from ..types import criterionType as _criterion_type, dataIterType as _dataiter_type, genericLoaderType as _loader_type, \
    optimizerType as _optimizer_type
from ..writer import SummaryWriter


class Trainer(_DDPMixin, _ToMixin, _IOMixin, metaclass=ABCMeta):
    RUN_PATH = str(Path(PROJECT_PATH) / "runs2")  # noqa

    def __init__(self, *, model: nn.Module, criterion: _criterion_type, tra_loader: _dataiter_type,
                 val_loader: _loader_type, save_dir: str, max_epoch: int = 100, num_batches: int = 100, device="cpu",
                 config: Dict[str, Any], **kwargs) -> None:
        super().__init__(save_dir=save_dir, max_epoch=max_epoch, num_batches=num_batches, device=device, **kwargs)
        self._model = self._inference_model = model
        self._criterion = criterion
        self._tra_loader: _dataiter_type = tra_loader
        self._val_loader: _loader_type = val_loader
        self.__hooks__ = nn.ModuleList()
        self._storage = Storage(save_dir=self._save_dir)
        self._writer = None
        if self.on_master():
            self._writer = SummaryWriter(log_dir=self._save_dir)
        self._config = config
        if config is not None:
            self.dump_config(self._config)
        self.__initialized__ = False

    def init(self):
        self._optimizer = self._init_optimizer()
        self._scheduler = self._init_scheduler(self._optimizer, scheduler_params=self._config.get("Scheduler", None))
        self.__initialized__ = True

    def register_hook(self, hook: TrainerHook):
        assert isinstance(hook, TrainerHook), hook
        self.__hooks__.append(hook)

    def register_hooks(self, *hook: TrainerHook):
        if self.__initialized__:
            raise RuntimeError("`register_hook must be called before `init()``")
        for h in hook:
            assert isinstance(h, TrainerHook), h
            self.register_hook(h)

    def _init_optimizer(self) -> _optimizer_type:
        optim_params = self._config["Optim"]
        optimizer = optim.__dict__[optim_params["name"]](
            params=filter(lambda p: p.requires_grad, self._model.parameters()),
            **{k: v for k, v in optim_params.items() if k != "name" and k != "pre_lr" and k != "ft_lr"}
        )
        optimizer.add_param_group(
            {"params": self.__hooks__.parameters(), **{k: v for k, v in optim_params.items()
                                                       if k != "name" and k != "pre_lr" and k != "ft_lr"}})
        return optimizer

    def _init_scheduler(self, optimizer, scheduler_params):
        if scheduler_params is None:
            return
        max_epoch = self._max_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epoch - self._config["Scheduler"]["warmup_max"],
            eta_min=1e-7
        )
        scheduler = GradualWarmupScheduler(optimizer, scheduler_params["multiplier"],
                                           total_epoch=scheduler_params["warmup_max"],
                                           after_scheduler=scheduler)
        return scheduler

    def start_training(self, **kwargs):
        if not self.__initialized__:
            raise RuntimeError(f"{self.__class__.__name__} should call `init()` first")
        self.to(self.device)
        if self.on_master():
            with self._writer:
                return self._start_training(**kwargs)
        return self._start_training(**kwargs)

    def _start_training(self, **kwargs):
        start_epoch = max(self._cur_epoch + 1, self._start_epoch)
        self._cur_score: float

        for self._cur_epoch in range(start_epoch, self._max_epoch + 1):
            with self._storage:  # save csv each epoch
                train_metrics = self.run_tra_epoch()
                if self.on_master():
                    inference_model = self._inference_model
                    eval_metrics, cur_score = self.run_eval_epoch(model=inference_model, loader=self._val_loader)
                    test_metrics, _ = self.run_eval_epoch(model=inference_model, loader=self._test_loader)

                best_case_sofa = self._best_score < cur_score
                if best_case_sofa:
                    self._best_score = cur_score
                if self.on_master() and best_case_sofa:
                    self.save_to(save_name="best.pth")

                if self.on_master():
                    self.save_to(save_name="last.pth")

                    self._storage.add_from_meter_interface(tra=train_metrics, val=eval_metrics, test=test_metrics,
                                                           epoch=self._cur_epoch)
                    self._writer.add_scalars_from_meter_interface(tra=train_metrics, val=eval_metrics,
                                                                  test=test_metrics, epoch=self._cur_epoch)

                if hasattr(self, "_scheduler"):
                    self._scheduler.step()

    def run_tra_epoch(self, **kwargs):
        epocher = self._create_tra_epoch(**kwargs)
        return self._run_tra_epoch(epocher)

    @staticmethod
    def _run_tra_epoch(epocher):
        epocher.run()
        return epocher.get_metric()

    @abstractmethod
    def _create_tra_epoch(self, **kwargs) -> EpocherBase:
        ...

    def run_eval_epoch(self, *, model, loader, **kwargs):
        epocher = self._create_eval_epoch(model=model, loader=loader, **kwargs)
        return self._run_eval_epoch(epocher)

    @abstractmethod
    def _create_eval_epoch(self, *, model, loader, **kwargs) -> EpocherBase:
        ...

    @staticmethod
    def _run_eval_epoch(epocher):
        epocher.run()
        return epocher.get_metric(), epocher.get_score()

    def set_model4inference(self, model: nn.Module):
        logger.trace(f"change inference model from {id(self._inference_model)} to {id(model)}")
        self._inference_model = model

    @property
    def save_dir(self):
        return str(self._save_dir)
