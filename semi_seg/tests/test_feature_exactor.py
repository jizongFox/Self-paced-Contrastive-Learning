from unittest import TestCase

import torch

from contrastyou.arch import UNet
from semi_seg.utils import FeatureExtractor, _LocalClusterWrapper, ClusterProjectorWrapper, IICLossWrapper, FeatureExtractorWithIndex


class TestFeatureExtractor(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._net = UNet()
        self._image = torch.randn(1, 3, 224, 224)
        self._feature_names = ["Conv1", "Conv5", "Up_conv5", "Up_conv4", "DeConv_1x1"]

    def test_feature_extractor(self):
        with FeatureExtractor(self._net, self._feature_names) as feature_extractor:
            for i in range(3):
                segment, (e5, e4, e3, e2, e1), (d5, d4, d3, d2) = self._net(self._image, return_features=True)
                assert id(feature_extractor["Conv1"]) == id(e1)
                assert id(feature_extractor["Conv5"]) == id(e5)
                assert id(feature_extractor["Up_conv5"]) == id(d5)
                assert id(feature_extractor["DeConv_1x1"]) == id(segment)


class TestFeatureExtractorWithIndex(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._net = UNet()
        self._image1 = torch.randn(1, 3, 224, 224)
        self._image2 = torch.randn(10, 3, 224, 224)
        self._feature_names = ["Conv1", "Conv5", "Up_conv5", "Up_conv4", "DeConv_1x1"]

    def test_feature_extractor(self):
        with FeatureExtractorWithIndex(self._net, self._feature_names) as fextractor:
            _ = self._net(self._image1)
            _ = self._net(self._image2)

            for f in fextractor:
                print(f.shape)




class TestLocalClusterWrapper(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._net = UNet()
        self._image = torch.randn(1, 3, 224, 224)
        self._feature_names = ["Up_conv4", "Up_conv3", "Up_conv2"]
        self._num_subheads = [10, 20, 30]
        self._num_clusters = [11, 12, 13]

    def test_clusterwrapper(self):
        self._wrapper = _LocalClusterWrapper(self._feature_names, num_subheads=self._num_subheads,
                                             num_clusters=self._num_clusters)
        with FeatureExtractor(self._net, self._feature_names) as feature_extractor:
            segment = self._net(self._image)
            for i, (feature, projector) in enumerate(zip(feature_extractor, self._wrapper)):
                projectsimplex = projector(feature)
                assert len(projectsimplex) == self._num_subheads[i]
                assert projectsimplex[0].shape[1] == self._num_clusters[i]


class TestProjectorWrapper(TestCase):
    def test_init(self):
        feature_names = ["Conv3", "Conv5", "Up_conv5", "Up_conv3"]
        projectors = ClusterProjectorWrapper()
        projectors.init_encoder(feature_names=feature_names,
                                )
        projectors.init_decoder(feature_names=feature_names)

        for projector in projectors:
            print(projector)

    def test_init_2(self):
        feature_names = ["Up_conv5", "Up_conv3"]
        projectors = ClusterProjectorWrapper()
        projectors.init_encoder(feature_names=feature_names
                                )
        projectors.init_decoder(feature_names=feature_names)

        for projector in projectors:
            print(projector)

    def test_init_3(self):
        feature_names = ["Up_conv5", "Up_conv3"]
        header_type = ["linear", "mlp"]
        projectors = ClusterProjectorWrapper()
        projectors.init_encoder(feature_names=feature_names
                                )
        projectors.init_decoder(feature_names=feature_names, head_types=header_type)

        for projector in projectors:
            print(projector)

    def test_init_failed(self):
        feature_names = []
        projectors = ClusterProjectorWrapper()
        projectors.init_encoder(feature_names=feature_names,
                                )
        projectors.init_decoder(feature_names=feature_names)
        for i, projector in enumerate(projectors):
            print(projector)

    def test_get(self):
        feature_names = ["Up_conv5", "Up_conv3"]
        projectors = ClusterProjectorWrapper()
        projectors.init_encoder(feature_names=feature_names,
                                )
        projectors.init_decoder(feature_names=feature_names)

        _ = projectors["Up_conv5"]
        _ = projectors["Up_conv3"]
        with self.assertRaises(IndexError):
            _ = projectors["Up_conv4"]


class TestIIDLossWrapper(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._feature_names = ["Conv3", "Conv5", "Up_conv5", "Up_conv3"]

    def test_losswrapper(self):
        wrapper = IICLossWrapper(self._feature_names, paddings=[1, 2], patch_sizes=[32, 64])

        for k, criterion in wrapper.items():
            print(k, ":", criterion)

    def test_losswrapper2(self):
        wrapper = IICLossWrapper(self._feature_names, paddings=1, patch_sizes=128)
        for k, criterion in wrapper.items():
            print(k, ":", criterion)
