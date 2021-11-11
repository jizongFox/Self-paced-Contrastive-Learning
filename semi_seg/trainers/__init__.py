from .base import *
from .pretrain import *
from .trainer import *

base_trainer_zoos = {
    "finetune": FineTuneTrainer,
    "uda": UDATrainer,
    "iic": IICTrainer,
    "mine": MineTrainer,
    "udaiic": UDAIICTrainer,
    "entropy": EntropyMinTrainer,
    "meanteacher": MeanTeacherTrainer,
    "ucmeanteacher": UCMeanTeacherTrainer,
    "iicmeanteacher": IICMeanTeacherTrainer,
    "midl": MIDLTrainer,
    "infonce": InfoNCETrainer,
    "infoncemt": InfoNCEMeanTeacherTrainer
}
pre_trainer_zoos = {
    "infoncepretrain": PretrainInfoNCETrainer,
}

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}
