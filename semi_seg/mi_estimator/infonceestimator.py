from contrastyou.arch import UNet
from contrastyou.projectors.heads import ProjectionHead, DenseProjectionHead

from .base import SingleEstimator


# todo: unclear to see what would happen.
class InfoNCEEstimator(SingleEstimator):
    """IICEestimator is the estimator for one single layer for the Unet"""
    __projector_initialized = False
    __criterion_initialized = False

    def _register_global_projector(self, *, feature_name: str, head_type: str, output_dim: int = 256, normalize=True,
                                   pool_name: str):
        input_dim = UNet.dimension_dict[feature_name]

        projector = ProjectionHead(input_dim=input_dim, head_type=head_type, normalize=normalize, pool_name=pool_name,
                                   output_dim=output_dim)
        self._projector = projector
        self.__projector_initialized = True

    def _register_dense_projector(self, *, feature_name: str, output_dim: int = 64, head_type: str,
                                  normalize: bool = False, pool_name="adaptive_avg", spatial_size=(16, 16), **kwargs):
        input_dim = UNet.dimension_dict[feature_name]
        self._projector = DenseProjectionHead(input_dim=input_dim, output_dim=output_dim, head_type=head_type,
                                              normalize=normalize,
                                              pool_name=pool_name, spatial_size=spatial_size)
        self.__projector_initialized = True

    def init_criterion(self, *, name: str, criterion_params=None):

        self.__criterion_initialized = True

    def forward(self, feat1, feat2):
        if not self.__criterion_initialized and self.__projector_initialized:
            raise RuntimeError("initialize projector and criterion first")

        return loss
