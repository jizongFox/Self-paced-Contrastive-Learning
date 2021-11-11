import os
from pathlib import Path

import torch
from deepclustering2.configparser._utils import get_config  # noqa
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats as disable_bn  # noqa
from torch import Tensor

from semi_seg.epochers import unl_extractor, ProjectionHead
from ._mixins import _PretrainEpocherMixin, _PretrainMonitorEpocherMxin
from .comparable import InfoNCEEpocher
from .miepocher import ConsistencyMIEpocher
from .miepocher import MITrainEpocher


# override batch loop in order to ignore the supervised loss.
class InfoNCEPretrainEpocher(_PretrainEpocherMixin, InfoNCEEpocher):
    pass


class InfoNCEPretrainMonitorEpocher(_PretrainMonitorEpocherMxin, InfoNCEEpocher):

    def generate_infonce(self, *, feature_name, features, projector, seed, partition_group, label_group) -> Tensor:
        proj_tf_feature, proj_feature_tf, unlabeled_tf_feature, unlabeled_feature_tf = \
            self.unlabeled_projection(unl_features=features, projector=projector, seed=seed,
                                      return_unlabeled_features=True)
        config = get_config(scope="base")
        save_dir = os.path.join("runs", config["Trainer"]["save_dir"])
        tag = f"features/{self.trainer._cur_epoch}"  # noqa
        tag = Path(save_dir) / tag
        tag.mkdir(parents=True, exist_ok=True)
        torch.save(unlabeled_tf_feature.cpu().detach(), str(tag / label_group[0]))

        tag = f"projections/{self.trainer._cur_epoch}"  # noqa
        tag = Path(save_dir) / tag
        tag.mkdir(parents=True, exist_ok=True)
        torch.save(proj_tf_feature.cpu().detach(), str(tag / label_group[0]))

        if isinstance(projector, ProjectionHead):
            # it goes to **global** representation here.
            return self._global_infonce(
                feature_name=feature_name,
                proj_tf_feature=proj_tf_feature,
                proj_feature_tf=proj_feature_tf,
                partition_group=partition_group,
                label_group=label_group
            )
        # it goes to a **dense** representation on pixels
        return self._dense_based_infonce(
            feature_name=feature_name,
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            partition_group=partition_group,
            label_group=label_group
        )

    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group):
        """methods go for global vectors"""
        pass

    def _regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                        partition_group, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        _ = [
            self.generate_infonce(
                feature_name=n, features=f, projector=p, seed=seed, partition_group=partition_group,
                label_group=label_group) for n, f, p in
            zip(self._fextractor.feature_names, unl_extractor(self._fextractor, n_uls=n_uls), self._projectors_wrapper)
        ]

        return torch.tensor(0, device=self.device)


class MIPretrainEpocher(_PretrainEpocherMixin, MITrainEpocher):
    pass


class UDAIICPretrainEpocher(_PretrainEpocherMixin, ConsistencyMIEpocher):
    pass
