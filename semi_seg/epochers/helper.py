import os
import warnings
from typing import List, Union

import numpy as np
import torch
from deepclustering2.type import to_device
from skimage.io import imsave
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from semi_seg.arch.hook import FeatureExtractor


class unl_extractor:
    def __init__(self, features: FeatureExtractor, n_uls: int) -> None:
        super().__init__()
        self._features = features
        self._n_uls = n_uls

    def __iter__(self):
        for feature in self._features:
            assert len(feature) >= self._n_uls, (len(feature), self._n_uls)
            yield feature[len(feature) - self._n_uls:]


def preprocess_input_with_twice_transformation(data, device, non_blocking=True):
    if len(data[0]) == 4:
        (image, image_tf, target, target_tf), filename, (partition_list, group_list) = \
            to_device(data[0], device, non_blocking), data[1], data[2]
        return (image, target), (image_tf, target_tf), filename, partition_list, group_list
    else:
        (t2, dw, t2_tf, dw_tf, target, target_tf), filename, (partition_list, group_list) = \
            to_device(data[0], device, non_blocking), data[1], data[2]
        return (torch.cat([t2, dw], dim=1), target), (torch.cat([t2_tf, dw_tf], dim=1), target_tf), \
               filename, partition_list, group_list


def preprocess_input_with_single_transformation(data, device, non_blocking=True):
    if len(data[0]) == 2:
        return data[0][0].to(device, non_blocking=non_blocking), data[0][1].to(device, non_blocking=non_blocking), \
               data[1], *data[2]
    (t2, dw, target), filename, (partition_list, group_list) = \
        to_device(data[0], device, non_blocking), data[1], data[2]
    return torch.cat([t2, dw], dim=1), target, filename, partition_list, group_list


class PartitionLabelGenerator:
    def __call__(self, partition_list: List[str], **kwargs):
        return LabelEncoder().fit(partition_list).transform(partition_list).tolist()


class PatientLabelGenerator:
    def __call__(self, patient_list: List[str], **kwargs):
        return LabelEncoder().fit(patient_list).transform(patient_list).tolist()


class ACDCCycleGenerator:
    def __call__(self, experiment_list: List[str], **kwargs):
        return [0 if e == "00" else 1 for e in experiment_list]


class SIMCLRGenerator:
    def __call__(self, partition_list: List[str], **kwargs):
        return list(range(len(partition_list)))


def _write_single_png(mask: Tensor, save_dir: str, filename: str):
    assert mask.shape.__len__() == 2, mask.shape
    mask = mask.cpu().detach().numpy()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        imsave(os.path.join(save_dir, (filename + ".png")), mask.astype(np.uint8))


def write_predict(predict_logit: Tensor, save_dir: str, filenames: Union[str, List[str]]):
    assert len(predict_logit.shape) == 4, predict_logit.shape
    if isinstance(filenames, str):
        filenames = [filenames, ]
    assert len(filenames) == len(predict_logit)
    predict_mask = predict_logit.max(1)[1]
    for m, f in zip(predict_mask, filenames):
        _write_single_png(m, os.path.join(save_dir, "pred"), f)


def write_img_target(image: Tensor, target: Tensor, save_dir: str, filenames: Union[str, List[str]]):
    if isinstance(filenames, str):
        filenames = [filenames, ]
    image = image.squeeze()
    target = target.squeeze()
    assert image.shape == target.shape
    for img, f in zip(image, filenames):
        _write_single_png(img * 255, os.path.join(save_dir, "img"), f)
    for targ, f in zip(target, filenames):
        _write_single_png(targ, os.path.join(save_dir, "gt"), f)
