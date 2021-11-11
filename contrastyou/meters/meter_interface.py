from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from typing import Dict, List

from . import metric


class MeterInterface:
    """
    meter interface only concerns about the situation in one epoch,
    without considering historical record and save/load state_dict function.
    """

    def __init__(self, default_focus="tra") -> None:
        self._group_bank: Dict[str, Dict[str, metric.Metric]] = defaultdict(OrderedDict)
        self.__group_focus = default_focus

    def register_meter(self, name: str, meter: metric.Metric):
        return self._register_meter(name=name, meter=meter, group_name=self.__group_focus)

    def _register_meter(self, *, name: str, group_name: str, meter: metric.Metric, ) -> None:
        if not isinstance(meter, metric.Metric):
            raise KeyError(meter)
        group_meter = self._group_bank[group_name]
        if name in group_meter:
            raise KeyError(f"{name} exists in {group_name}")
        group_meter[name] = meter

    def _delete_meter(self, *, name: str, group_name: str) -> None:
        meters = self._get_meters_by_group(group_name=group_name)
        if name not in meters:
            raise KeyError(name)
        del self._group_bank[group_name][name]
        if len(self._group_bank[group_name]) == 0:
            del self._group_bank[group_name]

    def delete_meter(self, name: str):
        return self._delete_meter(name=name, group_name=self.__group_focus)

    def delete_meters(self, name_list: List[str]):
        for name in name_list:
            self.delete_meter(name=name)

    def add(self, meter_name, *args, **kwargs):
        meter = self._get_meter(name=meter_name, group_name=self.__group_focus)
        meter.add(*args, **kwargs)

    def reset(self) -> None:
        for g in self.groups():
            for m in self._group_bank[g].values():
                m.reset()

    def join(self):
        for g in self.groups():
            meters = self._get_meters_by_group(g)
            for m in meters.values():
                m.join()

    def _get_meters_by_group(self, group_name: str):
        if group_name not in self.groups():
            raise KeyError(f"{group_name} not in {self.__class__.__name__}: ({', '.join(self.groups())})")
        meters: Dict[str, metric.Metric] = self._group_bank[group_name]
        return meters

    def _get_meter(self, *, name: str, group_name: str):
        meters: Dict[str, metric.Metric] = self._get_meters_by_group(group_name=group_name)
        if name not in meters:
            raise KeyError(f"{name} not in {group_name} group: ({', '.join(meters)})")
        return meters[name]

    def groups(self):
        return list(self._group_bank.keys())

    @property
    def cur_focus(self):
        return self.__group_focus

    @contextmanager
    def focus_on(self, group_name: str):
        prev_focus = self.__group_focus
        self.__group_focus = group_name
        yield
        self.__group_focus = prev_focus

    def _statistics_by_group(self, group_name: str):
        meters = self._get_meters_by_group(group_name)
        return {k: m.summary() for k, m in meters.items()}

    def statistics(self):
        """get statistics from meter_interface. ignoring the group with name starting with `_`"""
        groups = self.groups()
        for g in groups:
            if not g.startswith("_"):
                yield g, self._statistics_by_group(g)

    def __enter__(self):
        self.reset()

    def __exit__(self, *args, **kwargs):
        self.join()

    def __getitem__(self, meter_name: str) -> metric.Metric:
        return self._get_meter(name=meter_name, group_name=self.__group_focus)
