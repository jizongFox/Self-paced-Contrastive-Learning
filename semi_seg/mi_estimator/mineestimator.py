import torch
from contrastyou.arch.unet import UNet
from torch import nn
from torch.nn import functional as F, Flatten

from .base import SingleEstimator, EstimatorList


class MineEstimator(SingleEstimator):
    __projector_initialized = False
    __criterion_initialized = False

    def __init__(self, *, layer_name):
        super().__init__()
        self._layer_name = layer_name

        input_dim = UNet.dimension_dict[layer_name]

        self._projector = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, 3, 1, 1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(input_dim, input_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, feat1, feat2):
        feat2_prime = torch.cat([feat2[1:], feat2[0:1]], dim=0).contiguous()
        Ej = -F.softplus(self._projector(torch.cat([feat1, feat2], dim=1))).mean()
        Em = F.softplus(self._projector(torch.cat([feat1, feat2_prime], dim=1))).mean()
        return Em - Ej


class MineEstimatorArray(EstimatorList):

    def add(self, name: str, **params):
        single_estimator = MineEstimator(layer_name=name)
        self._estimator_dictionary[name] = single_estimator

    def add_interface(self, feature_names):
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]

        for f in feature_names:
            self.add(name=f)
