from itertools import repeat
from typing import List, Union

from deepclustering2.configparser._utils import get_config
from torch import nn, Tensor
from torch._six import container_abcs

from contrastyou.losses.iic_loss import IIDLoss as _IIDLoss, IIDSegmentationSmallPathLoss
from contrastyou.losses.pica_loss import PUILoss, PUISegLoss
from contrastyou.losses.wrappers import LossWrapperBase
from contrastyou.projectors.heads import DenseClusterHead as _LocalClusterHead, ClusterHead as _EncoderClusterHead, \
    DenseProjectionHead
from contrastyou.projectors.heads import ProjectionHead
from contrastyou.projectors.wrappers import _ProjectorWrapperBase, CombineWrapperBase
from semi_seg.arch import UNet
from semi_seg.arch.unet import get_channel_dim


def get_model(model):
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        return model.module
    elif isinstance(model, nn.Module):
        return model
    raise TypeError(type(model))


def _filter_encodernames(feature_list):
    encoder_list = UNet.encoder_names
    return list(filter(lambda x: x in encoder_list, feature_list))


def _filter_decodernames(feature_list):
    decoder_list = UNet.decoder_names
    return list(filter(lambda x: x in decoder_list, feature_list))


def _nlist(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            assert len(x) == n, (len(x), n)
            return x
        return list(repeat(x, n))

    return parse


class _num_class_mixin:
    _model: nn.Module

    @property
    def num_classes(self):
        return get_model(self._model).num_classes


class ContrastiveProjectorWrapper(_ProjectorWrapperBase):

    def __init__(self):
        super().__init__()
        self.__index = 0

    def _register_global_projector(self, *, feature_name: str, head_type: str, output_dim: int = 256, normalize=True,
                                   pool_name: str):
        _max_channel = get_config(scope="base")["Arch"].get("max_channel", None)
        input_dim = get_channel_dim(feature_name, max_channel=_max_channel)

        projector = ProjectionHead(input_dim=input_dim, head_type=head_type, normalize=normalize, pool_name=pool_name,
                                   output_dim=output_dim)
        self._projectors[f"{self.__index}|{feature_name}"] = projector
        self.__index += 1

    def _register_dense_projector(self, *, feature_name: str, output_dim: int = 64, head_type: str,
                                  normalize: bool = False, pool_name="adaptive_avg", spatial_size=(16, 16), **kwargs):
        _max_channel = get_config(scope="base")["Arch"].get("max_channel", None)
        input_dim = get_channel_dim(feature_name, max_channel=_max_channel)
        projector = DenseProjectionHead(input_dim=input_dim, output_dim=output_dim, head_type=head_type,
                                        normalize=normalize,
                                        pool_name=pool_name, spatial_size=spatial_size)
        self._projectors[f"{self.__index}|{feature_name}"] = projector
        self.__index += 1

    def register_global_projector(self, *,
                                  feature_names: Union[str, List[str]],
                                  head_type: Union[str, List[str]] = "mlp",
                                  output_dim: Union[int, List[int]] = 256,
                                  normalize: Union[bool, List[bool]] = True,
                                  pool_name: Union[str, List[str]] = "adaptive_avg",
                                  **kwargs):
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._global_feature_names = feature_names
        n = len(self._global_feature_names)
        n_pair = _nlist(n)
        head_type_ = n_pair(head_type)
        normalize_ = n_pair(normalize)
        pool_name_ = n_pair(pool_name)
        output_dim_ = n_pair(output_dim)
        for i, (f, h, n, p, o) in enumerate(
            zip(feature_names, head_type_, normalize_, pool_name_, output_dim_)):
            self._register_global_projector(feature_name=f, head_type=h, output_dim=o, normalize=n, pool_name=p)

    def register_dense_projector(self, *, feature_names: str, output_dim: int = 64, head_type: str,
                                 normalize: bool = False, pool_name="adaptive_avg", spatial_size=(16, 16), **kwargs
                                 ):
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._dense_feature_names = feature_names
        n = len(self._dense_feature_names)
        n_pair = _nlist(n)
        head_type_ = n_pair(head_type)
        normalize_ = n_pair(normalize)
        pool_name_ = n_pair(pool_name)
        output_dim_ = n_pair(output_dim)
        spatial_size_ = n_pair(spatial_size)
        for i, (f, h, n, p, o, s) in enumerate(
            zip(feature_names, head_type_, normalize_, pool_name_, output_dim_, spatial_size_)):
            self._register_dense_projector(feature_name=f, head_type=h, output_dim=o, normalize=n, pool_name=p,
                                           spatial_size=s, )


# decoder IIC projectors
class _LocalClusterWrapper(_ProjectorWrapperBase):
    def __init__(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
        normalize: Union[bool, List[bool]] = False,
        temperature: Union[float, List[float]] = 1.0,
    ) -> None:
        super(_LocalClusterWrapper, self).__init__()
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names

        n_pair = _nlist(len(feature_names))

        self._head_types = n_pair(head_types)
        self._num_subheads = n_pair(num_subheads)
        self._num_clusters = n_pair(num_clusters)
        self._normalize = n_pair(normalize)
        self._temperature = n_pair(temperature)

        for f, h, c, s, n, t in zip(self._feature_names, self._head_types, self._num_clusters, self._num_subheads,
                                    self._normalize, self._temperature):
            self._projectors[f] = self._create_clusterheads(
                input_dim=UNet.dimension_dict[f],
                head_type=h,
                num_clusters=c,
                num_subheads=s,
                normalize=n,
                T=t
            )

    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _LocalClusterHead(*args, **kwargs)


# encoder IIC projectors
class _EncoderClusterWrapper(_LocalClusterWrapper):
    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _EncoderClusterHead(*args, **kwargs)


# encoder and decoder projectors for IIC
class ClusterProjectorWrapper(CombineWrapperBase):

    def __init__(self):
        super().__init__()
        self._encoder_names = []
        self._decoder_names = []

    def init_encoder(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
        normalize: Union[bool, List[bool]] = False,
        temperature: Union[float, List[float]] = 1.0
    ):
        self._encoder_names = _filter_encodernames(feature_names)
        encoder_projectors = _EncoderClusterWrapper(
            self._encoder_names, head_types, num_subheads,
            num_clusters, normalize, temperature)
        self._projector_list.append(encoder_projectors)

    def init_decoder(self,
                     feature_names: Union[str, List[str]],
                     head_types: Union[str, List[str]] = "linear",
                     num_subheads: Union[int, List[int]] = 5,
                     num_clusters: Union[int, List[int]] = 10,
                     normalize: Union[bool, List[bool]] = False,
                     temperature: Union[float, List[float]] = 1.0
                     ):
        self._decoder_names = _filter_decodernames(feature_names)
        decoder_projectors = _LocalClusterWrapper(
            self._decoder_names, head_types, num_subheads,
            num_clusters, normalize, temperature)
        self._projector_list.append(decoder_projectors)

    @property
    def feature_names(self):
        return self._encoder_names + self._decoder_names


# loss function
class IIDLoss(_IIDLoss):

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        return super().forward(x_out, x_tf_out)[0]


# IIC loss for encoder and decoder
class IICLossWrapper(LossWrapperBase):

    def __init__(self,
                 feature_names: Union[str, List[str]],
                 paddings: Union[int, List[int]],
                 patch_sizes: Union[int, List[int]]) -> None:
        super().__init__()
        self._encoder_features = _filter_encodernames(feature_names)
        self._decoder_features = _filter_decodernames(feature_names)
        assert len(feature_names) == len(self._encoder_features) + len(self._decoder_features)

        if len(self._encoder_features) > 0:
            for f in self._encoder_features:
                self._LossModuleDict[f] = IIDLoss()
        if len(self._decoder_features) > 0:
            paddings = _nlist(len(self._decoder_features))(paddings)
            patch_sizes = _nlist(len(self._decoder_features))(patch_sizes)
            for f, p, size in zip(self._decoder_features, paddings, patch_sizes):
                self._LossModuleDict[f] = IIDSegmentationSmallPathLoss(padding=p, patch_size=size)

    @property
    def feature_names(self):
        return self._encoder_features + self._decoder_features


# PICA loss for encoder and decoder
class PICALossWrapper(LossWrapperBase):

    def __init__(self,
                 feature_names: Union[str, List[str]],
                 paddings: Union[int, List[int]]) -> None:
        super().__init__()
        self._encoder_features = _filter_encodernames(feature_names)
        self._decoder_features = _filter_decodernames(feature_names)
        assert len(feature_names) == len(self._encoder_features) + len(self._decoder_features)

        if len(self._encoder_features) > 0:
            for f in self._encoder_features:
                self._LossModuleDict[f] = PUILoss()
        if len(self._decoder_features) > 0:
            paddings = _nlist(len(self._decoder_features))(paddings)
            for f, p in zip(self._decoder_features, paddings):
                self._LossModuleDict[f] = PUISegLoss(padding=p)

    @property
    def feature_names(self):
        return self._encoder_features + self._decoder_features


# decoder IIC projectors
class _LocalClusterWrapper(_ProjectorWrapperBase):
    def __init__(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
        normalize: Union[bool, List[bool]] = False,
        temperature: Union[float, List[float]] = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names

        n_pair = _nlist(len(feature_names))

        self._head_types = n_pair(head_types)
        self._num_subheads = n_pair(num_subheads)
        self._num_clusters = n_pair(num_clusters)
        self._normalize = n_pair(normalize)
        self._temperature = n_pair(temperature)

        for f, h, c, s, n, t in zip(self._feature_names, self._head_types, self._num_clusters, self._num_subheads,
                                    self._normalize, self._temperature):
            self._projectors[f] = self._create_clusterheads(
                input_dim=UNet.dimension_dict[f],
                head_type=h,
                num_clusters=c,
                num_subheads=s,
                normalize=n,
                T=t
            )

    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _LocalClusterHead(*args, **kwargs)
