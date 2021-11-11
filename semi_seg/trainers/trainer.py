from copy import deepcopy
from typing import Tuple, Type

from deepclustering2.loss import KL_div
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.models import ema_updater
from deepclustering2.schedulers.customized_scheduler import RampScheduler
from torch import nn

from contrastyou.losses.contrast_loss2 import SelfPacedSupConLoss as SupConLoss
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss
from semi_seg.epochers import MITrainEpocher, ConsistencyMIEpocher, InfoNCEMeanTeacherEpocher
from semi_seg.epochers import TrainEpocher, EvalEpocher, ConsistencyTrainEpocher, EntropyMinEpocher, MeanTeacherEpocher, \
    MIMeanTeacherEpocher, MIDLPaperEpocher, InfoNCEEpocher, UCMeanTeacherEpocher
from semi_seg.mi_estimator.discrete_mi_estimator import IICEstimatorArray
from semi_seg.trainers._helper import _FeatureExtractor
from semi_seg.trainers.base import SemiTrainer
from semi_seg.utils import ContrastiveProjectorWrapper, _nlist


class UDATrainer(SemiTrainer):

    def _init(self):
        super(UDATrainer, self)._init()
        config = deepcopy(self._config["UDARegCriterion"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[config["name"]]
        self._reg_weight = float(config["weight"])

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return ConsistencyTrainEpocher

    def _run_epoch(self, epocher: ConsistencyTrainEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, reg_criterion=self._reg_criterion)
        result = epocher.run()
        return result


class MIDLTrainer(UDATrainer):
    def _init(self):
        super(MIDLTrainer, self)._init()
        self._uda_weight = deepcopy(self._reg_weight)
        self._reg_weight = 1.0
        config = deepcopy(self._config["MIDLPaperParameters"])
        self._iic_segcriterion = IIDSegmentationSmallPathLoss(
            padding=int(config["padding"]),
            patch_size=int(config["patch_size"])
        )
        self._iic_weight = float(config["iic_weight"])

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return MIDLPaperEpocher

    def _run_epoch(self, epocher: MIDLPaperEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(mi_weight=self._iic_weight, consistency_weight=self._uda_weight,
                     iic_segcriterion=self._iic_segcriterion,
                     reg_criterion=self._reg_criterion)  # noqa
        result = epocher.run()
        return result


# MI trainer
class IICTrainer(_FeatureExtractor, SemiTrainer):
    def _init(self):
        super(IICTrainer, self)._init()
        config = deepcopy(self._config["IICRegParameters"])
        self._mi_estimator_array = IICEstimatorArray()
        self._mi_estimator_array.add_encoder_interface(feature_names=self.feature_positions,
                                                       **config["EncoderParams"])
        self._mi_estimator_array.add_decoder_interface(feature_names=self.feature_positions,
                                                       **config["DecoderParams"], **config["LossParams"])

        self._reg_weight = float(config["weight"])
        self._enforce_matching = config["enforce_matching"]

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return MITrainEpocher

    def _run_epoch(self, epocher: MITrainEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, mi_estimator_array=self._mi_estimator_array,
                     enforce_matching=self._enforce_matching)
        result = epocher.run()
        return result

    def _init_optimizer(self):
        super(IICTrainer, self)._init_optimizer()
        optim_config = self._config["Optim"]
        self._optimizer.add_param_group(
            {"params": self._mi_estimator_array.parameters(),
             "lr": optim_config["lr"],
             "weight_decay": optim_config["weight_decay"]}
        )


# using MINE as estimation
class MineTrainer(IICTrainer):
    def _init(self):
        super(IICTrainer, self)._init()
        config = deepcopy(self._config["IICRegParameters"])
        from semi_seg.mi_estimator.mineestimator import MineEstimatorArray
        self._mi_estimator_array = MineEstimatorArray()
        self._mi_estimator_array.add_interface(self.feature_positions)

        self._reg_weight = float(config["weight"])
        self._enforce_matching = config["enforce_matching"]


# using infonce as estimation
class InfoNCETrainer(_FeatureExtractor, SemiTrainer):

    def _init(self):
        _projector_config = deepcopy(self._config["ProjectorParams"])

        global_features = _projector_config["GlobalParams"]["feature_names"] or []
        global_importance = _projector_config["GlobalParams"].pop("feature_importance") or []

        if isinstance(global_features, str):
            global_features = [global_features, ]
            global_importance = [global_importance, ]

        global_importance = global_importance[:len(global_features)]

        dense_features = _projector_config.get("DenseParams", {}).get("feature_names") or []
        dense_importance = _projector_config.get("DenseParams", {}).pop("feature_importance", None) or []

        if isinstance(dense_features, str):
            dense_features = [dense_features, ]
            dense_importance = [dense_importance, ]

        dense_importance = dense_importance[:len(dense_features)]

        # override FeatureExtractor by using the settings from Global and Dense parameters.
        if "FeatureExtractor" not in self._config:
            self._config["FeatureExtractor"] = {}
        self._config["FeatureExtractor"]["feature_names"] = global_features + dense_features
        self._config["FeatureExtractor"]["feature_importance"] = global_importance + dense_importance

        super(InfoNCETrainer, self)._init()

        _loss_config = _projector_config["LossParams"]
        self.__encoder_method__ = _loss_config["contrast_on"]
        if isinstance(self.__encoder_method__, str):
            self.__encoder_method__ = [self.__encoder_method__, ] * len(global_features)
        assert len(self.__encoder_method__) == len(global_features), self.__encoder_method__

        self._projector = ContrastiveProjectorWrapper()
        if len(global_features) > 0:
            self._projector.register_global_projector(
                **_projector_config["GlobalParams"]
            )
        if len(dense_features) > 0:
            self._projector.register_dense_projector(
                **_projector_config["DenseParams"]
            )
        self._encoder_criterion_list = self._define_encoder_criterion_list(
            criterion_params=_loss_config,
            n=len(global_features)
        )

        self._reg_weight = float(_projector_config["Weight"])

    def _define_encoder_criterion_list(self, *, criterion_params: dict, n: int):
        encoder_criterion = []
        n_pair = _nlist(n)
        for k, v in criterion_params.items():
            criterion_params[k] = iter(n_pair(v))

        for _ in range(n):
            params = {k: next(v) for k, v in criterion_params.items()}
            encoder_criterion.append(SupConLoss(**params))
        return encoder_criterion

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return InfoNCEEpocher

    def _run_epoch(self, epocher: InfoNCEEpocher, *args, **kwargs) -> EpochResultDict:
        for c in self._encoder_criterion_list:
            c.epoch_start()
        epocher.init(reg_weight=self._reg_weight, projectors_wrapper=self._projector,
                     infoNCE_criterion=self._encoder_criterion_list)
        epocher.set_global_contrast_method(contrast_on_list=self.__encoder_method__)
        result = epocher.run()
        for i, c in enumerate(self._encoder_criterion_list):
            criterion_result = c.epoch_end()
            result.update({k + f"{i}": v for k, v in criterion_result.items()})
        return result

    def _init_optimizer(self):
        # initialize the optimizer for encoder and decoder using same / different learning rates
        super(InfoNCETrainer, self)._init_optimizer()

        optim_config = self._config["Optim"]
        lr = optim_config["lr"]
        wd = optim_config["weight_decay"]

        self._optimizer.add_param_group({
            "params": self._projector.parameters(),
            "lr": lr,
            "weight_decay": wd
        })


class UDAIICTrainer(IICTrainer):

    def _init(self):
        super(UDAIICTrainer, self)._init()
        self._iic_weight = deepcopy(self._reg_weight)
        UDA_config = deepcopy(self._config["UDARegCriterion"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[UDA_config["name"]]
        self._uda_weight = float(UDA_config["weight"])

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return ConsistencyMIEpocher

    def _run_epoch(self, epocher: ConsistencyMIEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(mi_weight=self._iic_weight, consistency_weight=self._uda_weight,
                     mi_estimator_array=self._mi_estimator_array,
                     reg_criterion=self._reg_criterion, enforce_matching=self._enforce_matching)
        result = epocher.run()
        return result


class EntropyMinTrainer(SemiTrainer):

    def _init(self):
        super(EntropyMinTrainer, self)._init()
        config = deepcopy(self._config["EntropyMinParameters"])
        self._reg_weight = float(config["weight"])

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return EntropyMinEpocher

    def _run_epoch(self, epocher: EntropyMinEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight)
        result = epocher.run()
        return result


class MeanTeacherTrainer(SemiTrainer):

    def _init(self):
        super(MeanTeacherTrainer, self)._init()
        self._teacher_model = deepcopy(self._model)
        for param in self._teacher_model.parameters():
            param.detach_()
        self._teacher_model.train()
        config = deepcopy(self._config["MeanTeacherParameters"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[config["name"]]
        self._ema_updater = ema_updater(alpha=float(config["alpha"]), weight_decay=float(config["weight_decay"]))
        self._reg_weight = float(config["weight"])

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return MeanTeacherEpocher

    def _run_epoch(self, epocher: MeanTeacherEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, teacher_model=self._teacher_model, reg_criterion=self._reg_criterion,
                     ema_updater=self._ema_updater)
        result = epocher.run()
        return result

    def _eval_epoch(self, *, loader, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(model=self._teacher_model, val_loader=loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score


class UCMeanTeacherTrainer(MeanTeacherTrainer):

    def _init(self):
        super()._init()
        self._threshold = RampScheduler(begin_epoch=0, max_epoch=int(self._config["Trainer"]["max_epoch"]) // 3 * 2,
                                        max_value=1, min_value=0.75)

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return UCMeanTeacherEpocher

    def _run_epoch(self, epocher: UCMeanTeacherEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, teacher_model=self._teacher_model, reg_criterion=self._reg_criterion,
                     ema_updater=self._ema_updater, threshold=self._threshold)
        result = epocher.run()
        self._threshold.step()
        return result


class IICMeanTeacherTrainer(IICTrainer):

    def _init(self):
        super()._init()
        self._iic_weight = deepcopy(self._reg_weight)
        self._teacher_model = deepcopy(self._model)
        for param in self._teacher_model.parameters():
            param.detach_()
        self._teacher_model.train()
        config = deepcopy(self._config["MeanTeacherParameters"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[config["name"]]
        self._ema_updater = ema_updater(alpha=float(config["alpha"]), weight_decay=float(config["weight_decay"]))
        self._mt_weight = float(config["weight"])

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return MIMeanTeacherEpocher

    def _run_epoch(self, epocher: MIMeanTeacherEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(mi_estimator_array=self._mi_estimator_array, reg_criterion=self._reg_criterion,
                     teacher_model=self._teacher_model, ema_updater=self._ema_updater, mt_weight=self._mt_weight,
                     iic_weight=self._iic_weight, enforce_matching=self._enforce_matching)
        result = epocher.run()
        return result

    def _eval_epoch(self, *, loader, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(self._teacher_model, val_loader=loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score


class InfoNCEMeanTeacherTrainer(InfoNCETrainer):

    def _init(self):
        super()._init()  # initialize infonce
        # initialize mt
        if "MeanTeacherParameters" not in self._config:
            raise RuntimeError("`MeanTeacherParameters` should be in self._config")
        self._teacher_model = deepcopy(self._model)
        for param in self._teacher_model.parameters():
            param.detach_()
        self._teacher_model.train()
        config = deepcopy(self._config["MeanTeacherParameters"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[config["name"]]
        self._ema_updater = ema_updater(alpha=float(config["alpha"]), weight_decay=float(config["weight_decay"]))
        self._mt_weight = float(config["weight"])

    @property
    def epocher_class(self) -> Type[TrainEpocher]:
        return InfoNCEMeanTeacherEpocher

    def _run_epoch(self, epocher: InfoNCEMeanTeacherEpocher, *args, **kwargs) -> EpochResultDict:
        for c in self._encoder_criterion_list:
            c.epoch_start()
        epocher.init(teacher_model=self._teacher_model, reg_criterion=self._reg_criterion,
                     ema_updater=self._ema_updater, projectors_wrapper=self._projector,
                     infoNCE_criterion=self._encoder_criterion_list, infonce_weight=self._reg_weight,
                     mt_weight=self._mt_weight)
        epocher.set_global_contrast_method(contrast_on_list=self.__encoder_method__)
        result = epocher.run()
        for i, c in enumerate(self._encoder_criterion_list):
            criterion_result = c.epoch_end()
            result.update({k + f"{i}": v for k, v in criterion_result.items()})
        return result

    def _eval_epoch(self, *, loader, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(self._teacher_model, val_loader=loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score
