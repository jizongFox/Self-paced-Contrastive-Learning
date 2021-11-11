from functools import lru_cache

from loguru import logger
from torch import nn

from .nn import _ProjectorHeadBase, Flatten, Normalize, Identical, SoftmaxWithT


def get_contrastive_projector(*, head_type: str, pool_module, input_dim, hidden_dim, output_dim, normalize: bool):
    if head_type == "mlp":
        return nn.Sequential(
            pool_module,
            Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            Normalize() if normalize else Identical()
        )

    return nn.Sequential(
        pool_module,
        Flatten(),
        nn.Linear(input_dim, output_dim),
        Normalize() if normalize else Identical()
    )


def get_contrastive_dense_projector(*, head_type: str, input_dim, hidden_dim, output_dim, ):
    if head_type == "mlp":
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, 1, 0),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden_dim, output_dim, 1, 1, 0),
        )

    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, 1, 1, 0),
    )


def init_sub_header(*, head_type, input_dim, num_clusters, normalize, T):
    if head_type == "linear":
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(input_dim, num_clusters),
            Normalize() if normalize else Identical(),
            SoftmaxWithT(1, T=T)
        )
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(input_dim, 128),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(128, num_clusters),
        Normalize() if normalize else Identical(),
        SoftmaxWithT(1, T=T)
    )


def init_dense_sub_header(head_type, input_dim, hidden_dim, num_clusters, normalize, T):
    if head_type == "linear":
        return nn.Sequential(
            nn.Conv2d(input_dim, num_clusters, 1, 1, 0),
            Normalize() if normalize else Identical(),
            SoftmaxWithT(1, T=T)
        )
    return nn.Sequential(
        nn.Conv2d(input_dim, hidden_dim, 1, 1, 0),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(hidden_dim, num_clusters, 1, 1, 0),
        Normalize() if normalize else Identical(),
        SoftmaxWithT(1, T=T)
    )


# head for contrastive projection
class ProjectionHead(_ProjectorHeadBase):

    def __init__(self, *, input_dim: int, hidden_dim=256, output_dim: int, head_type: str, normalize: bool,
                 pool_name="adaptive_avg", spatial_size=(1, 1)):
        assert pool_name in ("adaptive_avg", "adaptive_max")
        super().__init__(input_dim=input_dim, output_dim=output_dim, head_type=head_type, normalize=normalize,
                         pool_name=pool_name, spatial_size=spatial_size)
        self._header = get_contrastive_projector(head_type=self._head_type, pool_module=self._pooling_module,
                                                 input_dim=self._input_dim, hidden_dim=hidden_dim,
                                                 output_dim=output_dim, normalize=normalize)
        message = self._record_message()
        logger.trace(message)

    def forward(self, features):
        return self._header(features)


# head for contrastive pixel-wise projection
class DenseProjectionHead(_ProjectorHeadBase):

    def __init__(self, *, input_dim: int, hidden_dim=128, output_dim: int, head_type: str, normalize: bool,
                 pool_name="adaptive_avg", spatial_size=(16, 16)):
        super().__init__(input_dim=input_dim, output_dim=output_dim, head_type=head_type, normalize=normalize,
                         pool_name=pool_name, spatial_size=spatial_size)

        self._projector = get_contrastive_dense_projector(head_type=self._head_type,
                                                          input_dim=self._input_dim, hidden_dim=hidden_dim,
                                                          output_dim=output_dim)
        message = self._record_message()
        logger.trace(message)

    def forward(self, features):
        out = self._projector(features)
        # change resolution here
        out = self._pooling_module(out)
        if self._normalize:
            return self._normalize_func(out)
        return out

    @property
    @lru_cache()
    def _normalize_func(self):
        return Normalize()


# head for IIC clustering
class ClusterHead(_ProjectorHeadBase):

    def __init__(self, *, input_dim: int, num_clusters=5, num_subheads=10, head_type="linear", T=1, normalize=False):
        super().__init__(input_dim=input_dim, output_dim=num_clusters, head_type=head_type, normalize=normalize,
                         pool_name="none", spatial_size=(1, 1))
        self._num_clusters = num_clusters
        self._num_subheads = num_subheads
        self._T = T

        headers = [
            init_sub_header(head_type=head_type, input_dim=self._input_dim, num_clusters=self._num_clusters,
                            normalize=self._normalize, T=self._T)
            for _ in range(self._num_subheads)
        ]

        self._headers = nn.ModuleList(headers)
        message = self._record_message()
        logger.debug(message)

    def forward(self, features):
        return [x(features) for x in self._headers]


# head for IIC segmentation clustering
class DenseClusterHead(_ProjectorHeadBase):
    """
    this classification head uses the loss for IIC segmentation, which consists of multiple heads
    """

    def __init__(self, *, input_dim: int, num_clusters=10, hidden_dim=64, num_subheads=10, T=1,
                 head_type: str = "linear", normalize: bool = False):
        super().__init__(input_dim=input_dim, output_dim=num_clusters, head_type=head_type, normalize=normalize,
                         pool_name="none", spatial_size=(1, 1))
        self._T = T

        headers = [
            init_dense_sub_header(head_type=head_type, input_dim=self._input_dim, hidden_dim=hidden_dim,
                                  num_clusters=num_clusters, normalize=self._normalize, T=self._T)
            for _ in range(num_subheads)
        ]
        self._headers = nn.ModuleList(headers)
        message = self._record_message()
        logger.debug(message)

    def forward(self, features):
        return [x(features) for x in self._headers]
