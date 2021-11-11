from unittest import TestCase

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from contrastyou.augment import AffineTensorTransform
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation
from semi_seg.tests._helper import create_dataset


class TestAffineTransform(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.label_set, self.unlabel_set, self.val_set = create_dataset(name="acdc", labeled_ratio=0.1)

        self.labeled_loader = DataLoader(self.label_set, batch_size=16)
        self._bilinear_transformer = AffineTensorTransform()
        self._nearest_transformer = AffineTensorTransform(mode="nearest")

    def test_affine_transform(self):
        data = self.labeled_loader.__iter__().__next__()
        (image, target), _, filename, partition, group = \
            preprocess_input_with_twice_transformation(data, "cuda")
        image_tf, affinematrix = self._bilinear_transformer(image, independent=True)
        target_tf, _ = self._nearest_transformer(target.float(), AffineMatrix=affinematrix)

        assert torch.allclose(target_tf.float().unique(), target.float().unique())

    def test_affine_transform_on_differnt_resolution(self):
        data = self.labeled_loader.__iter__().__next__()
        (image, target), _, filename, partition, group = \
            preprocess_input_with_twice_transformation(data, "cuda")
        image_tf, affinematrix = self._bilinear_transformer(image, independent=True)
        target_tf, _ = self._nearest_transformer(target.float(), AffineMatrix=affinematrix)
        target_downsample_tf, _ = self._nearest_transformer(
            F.interpolate(target.float(), scale_factor=0.5, mode="nearest"), AffineMatrix=affinematrix)
        target_tf_downsample = F.interpolate(target_tf, scale_factor=0.5, mode="nearest")

        # from deepclustering2.viewer import multi_slice_viewer_debug
        # multi_slice_viewer_debug([target_downsample_tf.squeeze(), target_tf_downsample.squeeze()], block=True)
        error = (target_downsample_tf - target_tf_downsample).abs().mean()
        assert error <= 0.1
