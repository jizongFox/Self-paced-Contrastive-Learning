import time
from abc import ABCMeta

from deepclustering2.meters2 import AverageValueMeter


class EpocherTimer(metaclass=ABCMeta):

    def __init__(self) -> None:
        super().__init__()
        self._batch_time = AverageValueMeter()
        self._data_fetch_time = AverageValueMeter()
        self.__batch_end = time.time()
        self.__batch_start = time.time()

    def record_batch_start(self):
        self.__batch_start = time.time()
        self._data_fetch_time.add(self.__batch_start - self.__batch_end)

    def record_batch_end(self):
        previous_batch_end = self.__batch_end

        self.__batch_end = time.time()
        self._batch_time.add(self.__batch_end - previous_batch_end)

    def summary(self):
        return {"batch_time": self._batch_time.summary()["mean"], "fetch_time": self._data_fetch_time.summary()["mean"]}

    def __enter__(self):
        self.record_batch_start()
        return self

    def __exit__(self, *args, **kwargs):
        self.record_batch_end()
