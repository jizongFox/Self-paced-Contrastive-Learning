from unittest import TestCase

from torch.utils.data import DataLoader

from contrastyou import DATA_PATH
from contrastyou.augment import SequentialWrapperTwice, SequentialWrapper
from contrastyou.datasets._seg_datset import ContrastBatchSampler  # noqa
from contrastyou.datasets.acdc_dataset import ACDCDataset
from contrastyou.datasets.spleen_dataset import SpleenDataset
from contrastyou.epocher._utils import preprocess_input_with_single_transformation, \
    preprocess_input_with_twice_transformation  # noqa
from deepclustering2.augment import pil_augment
from deepclustering2.dataset import PatientSampler
from deepclustering2.tqdm import tqdm
from deepclustering2.type import to_float

single_transform = SequentialWrapper(
    comm_transform=pil_augment.Compose([
        pil_augment.RandomCrop(224),
        pil_augment.RandomRotation(40),
    ]),
)
twice_transform = SequentialWrapperTwice(
    comm_transform=pil_augment.Compose([
        pil_augment.RandomCrop(224),
        pil_augment.RandomRotation(40),
    ]),
)


class TestACDCDataset(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._root_dir = DATA_PATH
        self._twice_trans = twice_transform
        self._single_trans = single_transform

    def test_init_acdc(self):
        dataset_tra = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=single_transform)
        batchsampler = PatientSampler(dataset_tra, grp_regex=dataset_tra.dataset_pattern, shuffle=False)
        dataloader_tra = DataLoader(dataset_tra, batch_sampler=batchsampler, num_workers=0)

        dataset_val = ACDCDataset(root_dir=DATA_PATH, mode="val", transforms=single_transform)
        batchsampler = PatientSampler(dataset_val, grp_regex=dataset_val.dataset_pattern, shuffle=False)
        dataloader_val = DataLoader(dataset_val, batch_sampler=batchsampler, num_workers=0)
        from itertools import chain
        info_df = {}
        for data in chain(dataloader_tra, dataloader_val):
            _, _, filenames, partition_list, group_list = preprocess_input_with_single_transformation(data, "cpu")
            info_df[group_list[0]] = len(filenames)
            assert set(to_float(list(partition_list))) == {0, 1, 2}
        self.assertEqual(len(info_df), 200)

    def test_contrastive_sampler(self):
        data_tra = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=single_transform)
        batchsampler = ContrastBatchSampler(data_tra, group_sample_num=4, partition_sample_num=1)
        dataloader = DataLoader(data_tra, batch_sampler=batchsampler, num_workers=8)
        for i, data in zip(range(100), tqdm(dataloader)):
            _, _, filenames, partition_list, group_list = preprocess_input_with_single_transformation(data, "cpu")
            assert len(set(group_list)) == 4
            assert set(to_float(list(partition_list))) == {0, 1, 2}

    def test_contrastive_sampler_twice_transform(self):
        data_tra = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=twice_transform)
        batchsampler = ContrastBatchSampler(data_tra, group_sample_num=4, partition_sample_num=1)
        dataloader = DataLoader(data_tra, batch_sampler=batchsampler, num_workers=12)
        for i, data in zip(range(1000), tqdm(dataloader)):
            _, _, filenames, partition_list, group_list = preprocess_input_with_twice_transformation(data, "cpu")
            assert len(set(group_list)) == 4
            assert set(to_float(list(partition_list))) == {0, 1, 2}


class TestSpleenDataset(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._root_dir = DATA_PATH
        self._twice_trans = SpleenStrongTransforms.pretrain
        self._single_trans = SpleenStrongTransforms.val

    def test_train_test_dataset(self):
        train_dataset = SpleenDataset(self._root_dir, mode="train", transforms=self._single_trans)
        data = train_dataset[0]
