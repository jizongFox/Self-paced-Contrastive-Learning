from torch import nn

from .nn import ModuleDict


class _ProjectorWrapperBase(nn.Module):

    def __init__(self):
        super().__init__()
        self._projectors = ModuleDict()

    def __len__(self):
        return len(self._projectors)

    def __iter__(self):
        for k, v in self._projectors.items():
            yield v

    def __getitem__(self, item):
        if item in self._projectors.keys():
            return self._projectors[item]
        raise IndexError(item)


class CombineWrapperBase(nn.Module):

    def __init__(self):
        super().__init__()
        self._projector_list = nn.ModuleList()

    def __iter__(self):
        for v in self._projector_list:
            yield from v
