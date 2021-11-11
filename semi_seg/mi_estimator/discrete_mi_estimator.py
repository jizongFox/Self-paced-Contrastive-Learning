from typing import Dict, Union, List

import torch
from deepclustering2.configparser._utils import get_config  # noqa

from contrastyou.projectors.heads import DenseClusterHead as _LocalClusterHead, ClusterHead as _EncoderClusterHead
from contrastyou.utils import average_iter
from .base import SingleEstimator, EstimatorList, encoder_names
from ..arch.unet import get_channel_dim
from ..utils import IIDLoss, IIDSegmentationSmallPathLoss, _nlist, \
    _filter_encodernames, _filter_decodernames


class IICEstimator(SingleEstimator):
    """IICEestimator is the estimator for one single layer for the Unet"""
    __projector_initialized = False
    __criterion_initialized = False

    def init_projector(self, *,
                       layer_name: str,
                       head_type: str = "linear",
                       num_subhead: int = 5,
                       num_cluster: int = 10,
                       normalize: bool = False,
                       temperature: float = 1.0):
        super().__init__()
        self._layer_name = layer_name
        self._head_type = head_type
        self._normalize = normalize
        self._num_subhead = num_subhead
        self._num_clusters = num_cluster
        self._t = temperature

        _max_channel = get_config(scope="base")["Arch"].get("max_channel", None)
        input_dim = get_channel_dim(layer_name, max_channel=_max_channel)

        CLUSTERHEAD = _EncoderClusterHead if self._layer_name in encoder_names else _LocalClusterHead

        self._projector = CLUSTERHEAD(input_dim=input_dim, num_clusters=num_cluster,
                                      num_subheads=num_subhead, head_type=head_type, T=temperature,
                                      normalize=normalize)

        self.__projector_initialized = True

    def init_criterion(self, *, padding: int, patch_size: int):
        if self._layer_name in encoder_names:
            self._criterion = IIDLoss()
        else:
            self._criterion = IIDSegmentationSmallPathLoss(padding=padding, patch_size=patch_size)

        self.__criterion_initialized = True

    def forward(self, feat1, feat2):
        if not self.__criterion_initialized and self.__projector_initialized:
            raise RuntimeError("initialize projector and criterion first")

        prob1, prob2 = list(
            zip(*[torch.chunk(x, 2, 0) for x in self._projector(
                torch.cat([feat1, feat2], dim=0)
            )])
        )
        loss = average_iter([self._criterion(x, y) for x, y in zip(prob1, prob2)])
        if torch.isnan(loss):
            raise RuntimeError(f"loss with nan, at {self.__class__.__name__}: {loss}")
        return loss


class IICEstimatorArray(EstimatorList):

    def add(self, *, name: str, projector_params: Dict = None, criterion_params: Dict = None, **kwargs):
        assert name not in self._estimator_dictionary, self._estimator_dictionary.keys()
        single_estimator = IICEstimator()
        single_estimator.init_projector(layer_name=name, **projector_params)
        single_estimator.init_criterion(**criterion_params)

        self._estimator_dictionary[name] = single_estimator

    def add_encoder_interface(self, feature_names: Union[str, List[str]],
                              head_types: Union[str, List[str]] = "linear",
                              num_subheads: Union[int, List[int]] = 5,
                              num_clusters: Union[int, List[int]] = 10,
                              normalize: Union[bool, List[bool]] = False,
                              temperature: Union[float, List[float]] = 1.0,
                              ):
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        feature_names = _filter_encodernames(feature_names)

        self._feature_names = feature_names

        n_pair = _nlist(len(feature_names))

        self._head_types = n_pair(head_types)
        self._num_subheads = n_pair(num_subheads)
        self._num_clusters = n_pair(num_clusters)
        self._normalize = n_pair(normalize)
        self._temperature = n_pair(temperature)

        for f, h, c, s, n, t in zip(self._feature_names, self._head_types, self._num_clusters, self._num_subheads,
                                    self._normalize, self._temperature):
            self.add(name=f,
                     criterion_params={"padding": 0, "patch_size": 0},
                     projector_params={"head_type": h, "num_cluster": c, "num_subhead": s, "normalize": n,
                                       "temperature": t, },
                     )

    def add_decoder_interface(self, feature_names: Union[str, List[str]],
                              head_types: Union[str, List[str]] = "linear",
                              num_subheads: Union[int, List[int]] = 5,
                              num_clusters: Union[int, List[int]] = 10,
                              normalize: Union[bool, List[bool]] = False,
                              temperature: Union[float, List[float]] = 1.0, paddings: int = 0, patch_sizes: int = 1000):

        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        feature_names = _filter_decodernames(feature_names)

        self._feature_names = feature_names

        n_pair = _nlist(len(feature_names))

        self._head_types = n_pair(head_types)
        self._num_subheads = n_pair(num_subheads)
        self._num_clusters = n_pair(num_clusters)
        self._normalize = n_pair(normalize)
        self._temperature = n_pair(temperature)
        self._paddings = n_pair(paddings)
        self._patch_sizes = n_pair(patch_sizes)

        for f, h, c, s, n, t, p, ps in zip(self._feature_names, self._head_types, self._num_clusters,
                                           self._num_subheads,
                                           self._normalize, self._temperature, self._paddings, self._patch_sizes):
            self.add(name=f,
                     criterion_params={"padding": p, "patch_size": ps},
                     projector_params={"head_type": h, "num_cluster": c, "num_subhead": s, "normalize": n,
                                       "temperature": t, },
                     )
