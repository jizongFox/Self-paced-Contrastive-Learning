import itertools
from copy import deepcopy
from unittest import TestCase

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from contrastyou import DATA_PATH
from contrastyou.augment import SequentialWrapperTwice
from contrastyou.datasets._seg_datset import ContrastBatchSampler
from contrastyou.datasets.acdc_dataset import ACDCDataset
from contrastyou.epocher.contrast_epocher import PretrainEncoderEpoch, PretrainDecoderEpoch
from contrastyou.losses.contrast_loss import SupConLoss
from deepclustering2.arch import get_arch
from deepclustering2.augment import pil_augment

transform = SequentialWrapperTwice(
    img_transform=pil_augment.Compose([
        pil_augment.CenterCrop(256),
        torchvision.transforms.ColorJitter(brightness=[0.8, 1], contrast=[0.8, 1]),
        pil_augment.ToTensor()
    ]),
    target_transform=pil_augment.Compose([
        pil_augment.CenterCrop(256),
        pil_augment.ToLabel()
    ]),
    if_is_target=[False, True]
)
global arch_dict
arch_dict = {"name": "ContrastUnet",
             "num_classes": 4,
             "input_dim": 1
             }
optim_dict = {
    "name": "Adam",
    "lr": 1e-6,
    "weight_decay": 1e-5
}
scheduler_dict = {
    "name": "CosineAnnealingLR",
    "T_max": 90,
    "warmup":
        {
            "multiplier": 300,
            "total_epoch": 10
        }

}


class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, c, *_ = features.shape
        return features.view(b, -1)


class TestPretrainEpocher(TestCase):

    def setUp(self) -> None:
        super().setUp()
        global arch_dict

        pretrain_datsaet = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=transform)
        self._pretrain_loader = iter(DataLoader(pretrain_datsaet,
                                                batch_sampler=ContrastBatchSampler(pretrain_datsaet, group_sample_num=4,
                                                                                   partition_sample_num=1)))
        arch_dict = deepcopy(arch_dict)
        self._model = get_arch(arch_dict.pop("name"), arch_dict)
        self._projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 128),
        )
        self._optimizer = torch.optim.Adam(itertools.chain(self._model.parameters(), self._projector.parameters()),
                                           lr=1e-6, weight_decay=1e-5)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, 10, 0)
        self._model.cuda()
        self._projector.cuda()

    def test_run_epoch(self):
        epocher = PretrainEncoderEpoch(self._model, self._projector, self._optimizer, cur_epoch=0, device="cuda",
                                       pretrain_encoder_loader=self._pretrain_loader,
                                       contrastive_criterion=SupConLoss(), num_batches=100)
        print(epocher.run())


class TestPretrainDecoder(TestCase):
    def setUp(self) -> None:
        super().setUp()
        global arch_dict

        arch_dict = deepcopy(arch_dict)
        pretrain_datsaet = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=transform)
        self._pretrain_loader = iter(DataLoader(pretrain_datsaet,
                                                batch_sampler=ContrastBatchSampler(pretrain_datsaet, group_sample_num=4,
                                                                                   partition_sample_num=1)))
        self._model = get_arch(arch_dict.pop("name"), arch_dict)
        self._projector = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        self._optimizer = torch.optim.Adam(itertools.chain(self._model.parameters(), self._projector.parameters()),
                                           lr=1e-6, weight_decay=1e-5)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, 10, 0)
        self._model.cuda()
        self._projector.cuda()

    def test_decoder(self):
        epocher = PretrainDecoderEpoch(self._model, self._projector, self._optimizer, cur_epoch=0, device="cuda",
                                       pretrain_decoder_loader=self._pretrain_loader,
                                       contrastive_criterion=SupConLoss(), num_batches=100)
        print(epocher.run())


class TestFineTuneEpocher(TestCase):
    def setUp(self) -> None:
        global arch_dict

        arch_dict = deepcopy(arch_dict)
        super().setUp()
        pretrain_datsaet = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=transform)
        self._pretrain_loader = iter(DataLoader(pretrain_datsaet))
        self._model = get_arch(arch_dict.pop("name"), arch_dict)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-6, weight_decay=1e-5)
        self._model.cuda()
        self._projector.cuda()


class TestIICEpocher(TestCase):
    pass
