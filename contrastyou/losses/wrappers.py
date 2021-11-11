from torch import nn


class LossWrapperBase(nn.Module):
    """
    Base loss wrapper that includes individual losses with different parameters.
    __iter__ should be used
    """

    def __init__(self):
        super().__init__()
        self._LossModuleDict = nn.ModuleDict()

    def __iter__(self):
        for k, v in self._LossModuleDict.items():
            yield v

    def __getitem__(self, item):
        if item in self._LossModuleDict.keys():
            return self._LossModuleDict[item]
        raise IndexError(item)

    def items(self):
        return self._LossModuleDict.items()


