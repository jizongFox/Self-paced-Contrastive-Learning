from torch import nn
from torchvision.models import resnet18


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self._f = resnet18()
        del self._f.avgpool
        del self._f.fc

    def forward(self, *args, **kwargs):
        return self._f.forward(*args, **kwargs)


if __name__ == '__main__':
    net = Net()
    pass
