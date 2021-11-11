import atexit
from abc import ABCMeta, abstractmethod
from threading import Thread

from loguru import logger

from .utils import ThreadQueue


class _StopToken:
    pass


class Metric(metaclass=ABCMeta):
    _initialized = False

    def __init__(self, threaded=False, use_deque=False) -> None:
        super().__init__()
        self._initialized = True
        self._threaded = threaded

        if self._threaded:
            logger.trace(f"{self.__class__.__name__} spawn a thread")
            self._queue = ThreadQueue(use_deque=use_deque)
            self._worker = Thread(target=self._worker_func, name="threaded_worker",
                                  args=(self._queue,), )
            self._worker.start()
            atexit.register(self.close)

    @abstractmethod
    def reset(self):
        pass

    def add(self, *args, **kwargs):
        assert self._initialized, f"{self.__class__.__name__} must be initialized by overriding __init__"
        if not self._threaded:
            return self._add(*args, **kwargs)
        return self._add_queue(*args, **kwargs)

    @abstractmethod
    def _add(self, *args, **kwargs):
        pass

    def _add_queue(self, *args, **kwargs):
        self._queue.put((args, kwargs))

    def summary(self):
        return self._summary()

    @abstractmethod
    def _summary(self):
        pass

    def _worker_func(self, input_queue: ThreadQueue):
        while True:
            try:
                args, kwags = input_queue.get()
            except IndexError:
                continue
            if isinstance(args[0], _StopToken):
                break
            self._add(*args, **kwags)

    def join(self):
        if not self._threaded:
            return
        self.close()
        logger.trace(f"{self.__class__.__name__} join the thread")
        self._worker.join()
        logger.trace(f"{self.__class__.__name__} end the thread")

    def close(self):
        self._add_queue(_StopToken())
