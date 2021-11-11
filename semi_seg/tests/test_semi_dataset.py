from unittest import TestCase

import torch
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation
from deepclustering2.dataset import PatientSampler
from semi_seg.tests._helper import create_dataset
from torch.utils.data import DataLoader


class TestACDCDataset(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.label_set, self.unlabel_set, self.val_set = create_dataset(name="acdc", labeled_ratio=0.1)

    def test_10_split(self):
        assert len(self.label_set.get_group_list()) == 174 // 10
        assert len(self.unlabel_set.get_group_list()) == (174 - 174 // 10)

    def test_100_split(self):
        label_set, unlabel_set, val_set = create_dataset(name="acdc", labeled_ratio=1.0)
        assert len(label_set.get_group_list()) == 174
        assert len(unlabel_set.get_group_list()) == 174

    def test_unfold_data(self):
        loader = DataLoader(self.label_set, batch_size=16)
        data = loader.__iter__().__next__()

        (image1, target1), (image2, target2), filename, partition, patient_group = \
            preprocess_input_with_twice_transformation(data, "cuda")
        assert image1.shape == torch.Size([16, 1, 224, 224])
        assert image1.shape == image2.shape


class TestMMWHMDataset(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.label_set, self.unlabel_set, self.val_set = create_dataset(name="mmwhm", labeled_ratio=0.1, modality="mr")

    def test_10_split(self):
        assert len(self.label_set.get_group_list()) == 15 // 10
        assert len(self.unlabel_set.get_group_list()) == (15 - 15 // 10)

    def test_100_split(self):
        label_set, unlabel_set, val_set = create_dataset(name="mmwhm", labeled_ratio=1.0, modality="ct")
        assert len(label_set.get_group_list()) == 15
        assert len(unlabel_set.get_group_list()) == 15

    def test_unfold_data(self):
        loader = DataLoader(self.label_set,
                            batch_sampler=PatientSampler(self.label_set, grp_regex="\d{4}"))
        data = loader.__iter__().__next__()

        (image1, target1), (image2, target2), filename, partition, patient_group = \
            preprocess_input_with_twice_transformation(data, "cuda")
        assert image1.shape == torch.Size([16, 1, 224, 224])
        assert image1.shape == image2.shape
