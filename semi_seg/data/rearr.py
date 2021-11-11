import random
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from copy import deepcopy as dcopy
from typing import Union, List, Callable, Dict

from torch.utils.data.sampler import Sampler

__all__ = ["ContrastDataset", "ContrastBatchSampler"]


class ContrastDataset(metaclass=ABCMeta):
    """
    each patient has 2 code, the first code is the group_name, which is the patient id
    the second code is the partition code, indicating the position of the image slice.
    All patients should have the same partition numbers so that they can be aligned.
    For ACDC dataset, the ED and ES ventricular volume should be considered further
    """
    get_memory_dictionary: Callable[[], Dict[str, List[str]]]

    @abstractmethod
    def _get_partition(self, *args) -> Union[str, int]:
        """get the partition of a 2D slice given its index or filename"""
        pass

    @abstractmethod
    def show_partitions(self) -> List[Union[str, int]]:
        """show all groups of 2D slices in the dataset"""
        pass

    @abstractmethod
    def show_scan_names(self) -> List[Union[str, int]]:
        """show all groups of 2D slices in the dataset"""
        pass


class ContrastBatchSampler(Sampler):
    """
    This class is going to realize the sampling for different patients and from the same patients
    `we form batches by first randomly sampling m < M volumes. Then, for each sampled volume, we sample one image per
    partition resulting in S images per volume. Next, we apply a pair of random transformations on each sampled image and
    add them to the batch
    """

    class _SamplerIterator:

        def __init__(self, group2index, partion2index, group_sample_num=4, partition_sample_num=1,
                     shuffle=False) -> None:
            self._group2index, self._partition2index = dcopy(group2index), dcopy(partion2index)

            assert 1 <= group_sample_num <= len(self._group2index.keys()), group_sample_num
            self._group_sample_num = group_sample_num
            self._partition_sample_num = partition_sample_num
            self._shuffle = shuffle

        def __iter__(self):
            return self

        def __next__(self):
            batch_index = []
            cur_gsamples = random.sample(self._group2index.keys(), self._group_sample_num)
            assert isinstance(cur_gsamples, list), cur_gsamples
            # for each gsample, sample at most partition_sample_num slices per partion
            for cur_gsample in cur_gsamples:
                gavailableslices = self._group2index[cur_gsample]
                for savailbleslices in self._partition2index.values():
                    try:
                        sampled_slices = random.sample(sorted(set(gavailableslices) & set(savailbleslices)),
                                                       self._partition_sample_num)
                        batch_index.extend(sampled_slices)
                    except ValueError:
                        continue
            if self._shuffle:
                random.shuffle(batch_index)
            return batch_index

    def __init__(self, dataset: ContrastDataset, scan_sample_num=4, partition_sample_num=1, shuffle=False) -> None:
        self._dataset = dataset
        filenames = dcopy(next(iter(dataset.get_memory_dictionary().values())))
        scan2index = defaultdict(lambda: [])
        partiton2index = defaultdict(lambda: [])
        for i, filename in enumerate(filenames):
            group = dataset._get_scan_name(filename)  # noqa
            scan2index[group].append(i)
            partition = dataset._get_partition(filename)  # noqa
            partiton2index[partition].append(i)
        self._scan2index = scan2index
        self._partition2index = partiton2index
        self._scan_sample_num = scan_sample_num
        self._partition_sample_num = partition_sample_num
        self._shuffle = shuffle

    def __iter__(self):
        return self._SamplerIterator(self._scan2index, self._partition2index, self._scan_sample_num,
                                     self._partition_sample_num, shuffle=self._shuffle)

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore
