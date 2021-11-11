from unittest import TestCase

import torch

from contrastyou.arch import UNet, UNetFeatureExtractor
from deepclustering2.loss import Entropy


def _if_grad_disabled(grad):
    """check if a given grad is set to be None or Zerolike"""
    if grad is None:
        return True
    if torch.allclose(grad, torch.zeros_like(grad)):
        return True
    return False


class TestUnet(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._img = torch.randn(10, 1, 224, 224)

    def test_unet(self):
        net = UNet(input_dim=1, num_classes=4)
        prediction = net(self._img, )
        prediction_representation = net(self._img, return_features=True)
        self.assertTrue(torch.allclose(prediction, prediction_representation[0]))
        self.assertFalse(id(prediction) == id(prediction_representation[0]))

    def test_encoder_grad(self):
        net = UNet(input_dim=1, num_classes=4)
        net.disable_grad_encoder()
        predict = net(self._img, return_features=True)[0]
        loss = Entropy()(predict.softmax(1))
        loss.backward()
        assert _if_grad_disabled(net.Conv2.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Up_conv2.parameters().__next__().grad)

        net.zero_grad()
        net.enable_grad_encoder()
        net.disable_grad_decoder()
        predict = net(self._img, return_features=True)[0]
        loss = Entropy()(predict.softmax(1))
        loss.backward()
        assert not _if_grad_disabled(net.Conv2.parameters().__next__().grad)
        assert _if_grad_disabled(net.Up_conv2.parameters().__next__().grad)


class TestFeatureExtratorAndGradientManipulation(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._img = torch.randn(10, 1, 224, 224)

    def test_extract_d2(self):
        feature_extractor = UNetFeatureExtractor("Up_conv2")
        net = UNet(input_dim=1, num_classes=4)
        net.disable_grad_all()
        net.enable_grad_util("Up_conv2")
        prediction, *features = net(self._img, return_features=True)
        e1 = feature_extractor(features)[0]
        loss = e1.mean()
        loss.backward()
        assert not _if_grad_disabled(net.Conv1.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Up_conv2.parameters().__next__().grad)
        assert _if_grad_disabled(net.DeConv_1x1.parameters().__next__().grad)

    def test_extract_e4(self):
        feature_extractor = UNetFeatureExtractor("Conv4")
        net = UNet(input_dim=1, num_classes=4)
        net.disable_grad_all()
        net.enable_grad_util("Conv4")
        prediction, *features = net(self._img, return_features=True)
        e1 = feature_extractor(features)[0]
        loss = e1.mean()
        loss.backward()
        assert not _if_grad_disabled(net.Conv1.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Conv2.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Conv3.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Conv4.parameters().__next__().grad)
        assert _if_grad_disabled(net.Conv5.parameters().__next__().grad)
        assert _if_grad_disabled(net.Up_conv2.parameters().__next__().grad)
        assert _if_grad_disabled(net.DeConv_1x1.parameters().__next__().grad)

    def test_freeze_somedecoder(self):
        feature_extractor = UNetFeatureExtractor("Up_conv3")
        net = UNet(input_dim=1, num_classes=4)
        net.disable_grad_all()
        net.enable_grad(from_="Conv2", util="Up_conv3")
        prediction, *features = net(self._img, return_features=True)
        e1 = feature_extractor(features)[0]
        loss = e1.mean()
        loss.backward()
        assert _if_grad_disabled(net.Conv1.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Conv2.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Conv3.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Conv4.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Conv5.parameters().__next__().grad)

        assert not _if_grad_disabled(net.Up_conv5.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Up_conv4.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Up_conv3.parameters().__next__().grad)

        assert not _if_grad_disabled(net.Up5.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Up4.parameters().__next__().grad)
        assert not _if_grad_disabled(net.Up3.parameters().__next__().grad)

        assert _if_grad_disabled(net.Up2.parameters().__next__().grad)
        assert _if_grad_disabled(net.Up_conv2.parameters().__next__().grad)
        assert _if_grad_disabled(net.DeConv_1x1.parameters().__next__().grad)
