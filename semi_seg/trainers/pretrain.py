import os
from typing import Type

from deepclustering2.meters2 import StorageIncomeDict

from ._helper import _PretrainTrainerMixin
from .trainer import InfoNCETrainer, IICTrainer, UDAIICTrainer
from ..epochers import InfoNCEPretrainEpocher, TrainEpocher, MIPretrainEpocher, UDAIICPretrainEpocher


class PretrainInfoNCETrainer(_PretrainTrainerMixin, InfoNCETrainer):
    from semi_seg.epochers.pretrain import InfoNCEPretrainMonitorEpocher

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return InfoNCEPretrainEpocher

    def run_monitor(self, epoch_class=InfoNCEPretrainMonitorEpocher):
        raise RuntimeError("we do not support run monitor mode")
        # previous_class = self.epocher_class
        # self.epocher_class = epoch_class
        # epocher = self._run_init()
        # result = self._run_epoch(epocher, )
        # self.epocher_class = previous_class
        # return result

    def _start_training(self, *, run_monitor=False):
        run_monitor = os.environ.get("MONITOR", 0) == "1" or run_monitor
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            cur_score: float
            train_result = self.run_epoch()
            if run_monitor and self._cur_epoch < 40:
                if self._config["Data"]["name"] == "acdc" and self._cur_epoch % 6 == 1:
                    self.run_monitor()

            # update lr_scheduler
            if hasattr(self, "_scheduler"):
                self._scheduler.step()
            if self.on_master():
                storage_per_epoch = StorageIncomeDict(pretrain=train_result)
                self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
                self._writer.add_scalar_with_StorageDict(storage_per_epoch, self._cur_epoch)
                # save_checkpoint
                self._save_to(self._save_dir, "last.pth")
                # save storage result on csv file.
                self._storage.to_csv(self._save_dir)


class PretrainIICTrainer(_PretrainTrainerMixin, IICTrainer):

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return MIPretrainEpocher


class PretrainUDAIICTrainer(_PretrainTrainerMixin, UDAIICTrainer):

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return UDAIICPretrainEpocher
