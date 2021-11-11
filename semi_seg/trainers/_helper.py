from copy import deepcopy as dcopy
from functools import partial
from pathlib import Path
from typing import List, Dict, Any, Callable

from deepclustering2.meters2 import StorageIncomeDict, Storage, EpochResultDict
from deepclustering2.tqdm import item2str
from deepclustering2.writer import SummaryWriter
from loguru import logger
from torch import nn
from torch.utils.data.dataloader import _BaseDataLoaderIter as BaseDataLoaderIter, DataLoader  # noqa

from contrastyou import get_cc_data_path
from contrastyou.data import InfiniteRandomSampler, ScanBatchSampler, DatasetBase
from contrastyou.utils import get_dataset, fix_seed
from semi_seg.data import ACDCDataset, ProstateDataset, mmWHSCTDataset, ProstateMDDataset
from semi_seg.data.creator import augment_zoo, data_zoo
from semi_seg.data.rearr import ContrastBatchSampler


def get_partition_num(name: str):
    partition_num_zoo = {"acdc": ACDCDataset.partition_num,
                         "prostate": ProstateDataset.partition_num,
                         "mmwhsct": mmWHSCTDataset.partition_num,
                         "mmwhsmr": mmWHSCTDataset.partition_num,
                         "prostate_md": ProstateMDDataset.partition_num}
    return partition_num_zoo[name]


@fix_seed
def _get_contrastive_dataloader(partial_loader, contrastive_params):
    contrastive_params = dcopy(contrastive_params)
    # going to get all dataset with contrastive sampler
    unlabeled_dataset: DatasetBase = get_dataset(partial_loader)
    data_name = {class_.__name__: name for name, class_ in data_zoo.items()}[unlabeled_dataset._name.split("-")[0]]
    is_preload = unlabeled_dataset._is_preload  # noqa
    dataset_type = unlabeled_dataset.__class__
    dataset = dataset_type(root_dir=get_cc_data_path(), mode="train", transforms=unlabeled_dataset.transforms)

    logger.opt(depth=2).debug(f"creating {dataset.__class__.__name__} contrastive dataset with "
                              f"{len(dataset.get_scan_list())} scans")

    if is_preload:
        dataset.preload()

    num_workers = contrastive_params.pop("num_workers")

    augment = augment_zoo[data_name]

    batch_sampler = None

    batch_size = contrastive_params["scan_sample_num"] * get_partition_num(data_name)
    sampler = InfiniteRandomSampler(dataset, shuffle=True)

    logger.opt(depth=2).trace(f" with batch_size = {batch_size}, num_workers = {num_workers} ")
    if data_name == "acdc":
        # only group the acdc dataset
        batch_sampler = ContrastBatchSampler(dataset=dataset, **contrastive_params)
        logger.opt(depth=2).trace(f"creating contrastive batch sampler")
        batch_size = 1
        sampler = None

    contrastive_loader = DataLoader(
        dataset, batch_sampler=batch_sampler, sampler=sampler,
        num_workers=num_workers, batch_size=batch_size,
        pin_memory=True, shuffle=False,
    )

    demo_dataset = dataset_type(root_dir=str(Path(unlabeled_dataset._root_dir).parent),  # noqa
                                mode="train", transforms=augment.trainval)

    demo_loader = DataLoader(demo_dataset, batch_size=1, batch_sampler=ScanBatchSampler(dataset))
    logger.opt(depth=2).trace(f"creating {dataset.__class__.__name__} demo dataset with {len(dataset)} scans")
    return contrastive_loader, demo_loader


# mixin for feature extractor
class _FeatureExtractor:
    feature_positions: List[str]
    _config: Dict[str, Any]
    set_feature_positions: Callable

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__feature_extractor_initialized = False

    def _init(self):
        if "FeatureExtractor" not in self._config:
            raise RuntimeError("FeatureExtractor Should be in the config.")
        feature_config = self._config["FeatureExtractor"]
        self.set_feature_positions(feature_config["feature_names"])
        feature_importance = feature_config["feature_importance"]
        assert isinstance(feature_importance, list), type(feature_importance)
        feature_importance = [float(x) for x in feature_importance]
        self._feature_importance = feature_importance

        assert len(self._feature_importance) == len(self.feature_positions), \
            (self._feature_importance, self.feature_positions)

        logger.info("{} feature importance: {}", self.__class__.__name__,
                    item2str({f"{c}|{i}": v for i, (c, v) in
                              enumerate(zip(self.feature_positions, self._feature_importance))}))
        self.__feature_extractor_initialized = True
        super(_FeatureExtractor, self)._init()  # noqa

    def start_training(self, *args, **kwargs):
        if not self.__feature_extractor_initialized:
            raise RuntimeError("_FeatureExtractor should be initialized by calling `_int()` first.")
        return super(_FeatureExtractor, self).start_training(*args, **kwargs)  # noqa


class _PretrainTrainerMixin:
    _model: nn.Module
    _unlabeled_loader: iter
    _config: Dict[str, Any]
    _start_epoch: int
    _max_epoch: int
    _save_dir: str
    init: Callable[..., None]
    on_master: Callable[[], bool]
    run_epoch: Callable[[], EpochResultDict]
    _save_to: Callable[[str, str], None]
    _contrastive_loader = BaseDataLoaderIter
    _storage: Storage
    _writer: SummaryWriter

    def __init__(self, *args, **kwargs):
        super(_PretrainTrainerMixin, self).__init__(*args, **kwargs)
        self.__initialized_grad = False

    def _init(self, *args, **kwargs):
        super(_PretrainTrainerMixin, self)._init(*args, **kwargs)  # noqa
        # here you have conventional training objects
        self._contrastive_loader, self._monitor_loader = _get_contrastive_dataloader(
            self._unlabeled_loader, self._config["ContrastiveLoaderParams"]
        )

    def _run_epoch(self, epocher, *args, **kwargs) -> EpochResultDict:
        epocher.init = partial(epocher.init, chain_dataloader=self._contrastive_loader,
                               monitor_dataloader=self._monitor_loader)
        return super(_PretrainTrainerMixin, self)._run_epoch(epocher, *args, **kwargs)  # noqa

    def _start_training(self, **kwargs):
        assert self.__initialized_grad, "`enable_grad` must be called first"
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            # update lr_scheduler
            if hasattr(self, "_scheduler"):
                self._scheduler.step()
            if self.on_master():
                storage_per_epoch = StorageIncomeDict(pretrain=train_result)
                self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
                self._writer.add_scalar_with_StorageDict(storage_per_epoch, self._cur_epoch)
                # save_checkpoint
                self._save_to(self._save_dir, "last.pth")
                # save storage result on csv file.
                self._storage.to_csv(self._save_dir)
