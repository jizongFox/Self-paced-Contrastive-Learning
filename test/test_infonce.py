from unittest import TestCase

from torch.optim import Adam

from contrastyou.arch import UNet
from contrastyou.losses.contrast_loss import SupConLoss2
from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div
from semi_seg.utils import ContrastiveProjectorWrapper  # noqa
from semi_seg.dsutils import get_dataloaders
from semi_seg.epochers.comparable import InfoNCEEpocher


class TestInfoNCECase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        network = UNet(input_dim=1, num_classes=4)
        optimizer = Adam(network.parameters(), )
        self._model = network
        self._optimizer = optimizer
        self._feature_names = ["Conv5", "Conv5", "Up_conv2"]
        self._feature_importance = [1.0, 1.0, 1.0]
        with ConfigManger(
            base_path="../config/base.yaml",
            optional_paths="../config/specific/infonce.yaml"
        )(scope="base") as self.config:
            self.labeled_loader, self.unlabeled_loader, self.val_loader = get_dataloaders(self.config)
            self._projector = ContrastiveProjectorWrapper()

            self.__contrast_name = self.config["InfoNCEParameters"]["EncoderParams"].pop("method_name")

            self._projector.init_encoder(
                feature_names=self._feature_names,
                **self.config["InfoNCEParameters"]["EncoderParams"])
            self._projector.init_decoder(
                feature_names=self._feature_names,
                **self.config["InfoNCEParameters"]["DecoderParams"]
            )
            self._infonce_criterion = SupConLoss2()

    def override_encoder_params(self, **kwargs):
        self.config["InfoNCEParameters"]["EncoderParams"].update(kwargs)

    def override_decoder_params(self, **kwargs):
        self.config["InfoNCEParameters"]["DecoderParams"].update(kwargs)

    def test_run_epocher(self):
        epocher = InfoNCEEpocher(
            model=self._model, optimizer=self._optimizer,
            labeled_loader=iter(self.labeled_loader),
            unlabeled_loader=iter(self.unlabeled_loader),
            sup_criterion=KL_div(),
            num_batches=10,
            feature_position=self._feature_names,
            feature_importance=self._feature_importance,
            device="cuda"
        )
        epocher.init(reg_weight=0.1, projectors_wrapper=self._projector, infoNCE_criterion=self._infonce_criterion)
        epocher.run()
