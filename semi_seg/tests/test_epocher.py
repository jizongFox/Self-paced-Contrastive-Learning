from unittest import TestCase

import torch
from torch import nn
from torch.utils.data import DataLoader

from contrastyou.arch import UNet
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.loss import KL_div
from deepclustering2.utils import set_benchmark
from semi_seg.utils import ClusterProjectorWrapper, IICLossWrapper, ContrastiveProjectorWrapper
from semi_seg.epochers import TrainEpocher, EvalEpocher, ConsistencyTrainEpocher, MITrainEpocher, ConsistencyMIEpocher
from semi_seg.tests._helper import create_dataset


class TestPartialEpocher(TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.label_set, self.unlabel_set, self.val_set = create_dataset("acdc", 0.1)

        self.labeled_loader = DataLoader(
            self.label_set, batch_size=2,
            sampler=InfiniteRandomSampler(self.label_set, shuffle=True),
        )
        self.unlabeled_loader = DataLoader(
            self.unlabel_set, batch_size=3,
            sampler=InfiniteRandomSampler(self.unlabel_set, shuffle=True)
        )
        self.val_loader = DataLoader(self.val_set, )
        self.net = UNet(input_dim=1, num_classes=4)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self._num_batches = 10
        self._feature_position = ["Conv5", "Up_conv3", "Up_conv2"]
        self._feature_importance = [1, 1, 1]

        set_benchmark(1)

    def test_partial_epocher(self):
        partial_trainer = TrainEpocher(self.net, self.optimizer, self.labeled_loader, self.unlabeled_loader,
                                       sup_criterion=KL_div(), reg_weight=0.0, num_batches=self._num_batches,
                                       cur_epoch=0, device="cuda", feature_position=self._feature_position,
                                       feature_importance=self._feature_importance)
        train_result = partial_trainer.run()
        print(train_result)

    def test_val_epocher(self):
        val_trainer = EvalEpocher(self.net, sup_criterion=KL_div(), val_loader=self.val_loader, cur_epoch=0,
                                  device="cuda")
        val_result, cur_score = val_trainer.run()
        print(val_result)

    def test_uda_epocher(self):
        uda_trainer = ConsistencyTrainEpocher(self.net, self.optimizer, self.labeled_loader, self.unlabeled_loader,
                                              sup_criterion=KL_div(), reg_criterion=nn.MSELoss(), reg_weight=0.1,
                                              num_batches=self._num_batches, cur_epoch=0, device="cuda",
                                              feature_position=self._feature_position,
                                              feature_importance=self._feature_importance)
        uda_result = uda_trainer.run()
        print(uda_result)

    def test_iic_epocher(self):
        iic_segment_criterion = IICLossWrapper(self._feature_position, paddings=[1, 1], patch_sizes=2048)

        projector_wrapper = ClusterProjectorWrapper()
        projector_wrapper.init_encoder(self._feature_position, )
        projector_wrapper.init_decoder(self._feature_position, )

        iic_epocher = MITrainEpocher(model=self.net, optimizer=self.optimizer, labeled_loader=self.labeled_loader,
                                     unlabeled_loader=self.unlabeled_loader, sup_criterion=KL_div(),
                                     num_batches=self._num_batches, cur_epoch=0, device="cuda",
                                     feature_position=self._feature_position,
                                     feature_importance=self._feature_importance)
        iic_epocher.init(reg_weight=1.0, projectors_wrapper=projector_wrapper,
                         IIDSegCriterionWrapper=iic_segment_criterion)
        result_dict = iic_epocher.run()
        print(result_dict)

    def test_udaiic_epocher(self):
        iic_segment_criterion = IIDSegmentationSmallPathLoss(padding=1, patch_size=64)
        projectors_wrapper = _LocalClusterWrappaer(self._feature_position, num_subheads=10, num_clusters=10).to("cuda")
        udaiic_epocher = ConsistencyMIEpocher(
            self.net, optimizer=self.optimizer, labeled_loader=self.labeled_loader,
            unlabeled_loader=self.unlabeled_loader, sup_criterion=KL_div(), iic_weight=0.1,
            cons_weight=0.2,
            num_batches=self._num_batches, cur_epoch=0, device="cuda",
            feature_position=self._feature_position,
            feature_importance=self._feature_importance,
            reg_criterion=nn.MSELoss(),
            IIDSegCriterion=iic_segment_criterion, projectors_wrapper=projectors_wrapper)
        result_dict = udaiic_epocher.run()
        print(result_dict)


class TestPreTrainEpocher(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.label_set, self.unlabel_set, self.val_set = create_dataset("acdc", 0.1)

        self.labeled_loader = DataLoader(
            self.label_set, batch_size=2,
            sampler=InfiniteRandomSampler(self.label_set, shuffle=True),
        )
        self.unlabeled_loader = DataLoader(
            self.unlabel_set, batch_size=3,
            sampler=InfiniteRandomSampler(self.unlabel_set, shuffle=True)
        )
        self.val_loader = DataLoader(self.val_set, )
        self.net = UNet(input_dim=1, num_classes=4)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self._num_batches = 10
        self._feature_position = ["Conv5", "Up_conv3", "Up_conv2"]
        self._feature_importance = [1, 1, 1]

    def test_infoncepretrainepocher(self):
        from semi_seg.epochers.pretrain import InfoNCEPretrainEpocher
        from contrastyou.losses.contrast_loss import SupConLoss
        epocher = InfoNCEPretrainEpocher(model=self.net, optimizer=self.optimizer, labeled_loader=self.labeled_loader,
                                         unlabeled_loader=self.unlabeled_loader, sup_criterion=nn.MSELoss(),
                                         num_batches=self._num_batches,
                                         cur_epoch=0, device="cuda", feature_position=self._feature_position,
                                         feature_importance=self._feature_importance)

        reg_weight = 0.1
        projectors_wrapper: ContrastiveProjectorWrapper = ContrastiveProjectorWrapper()
        projectors_wrapper.init_encoder(self._feature_position)
        projectors_wrapper.init_decoder(self._feature_position)
        infoNCE_criterion: SupConLoss = SupConLoss()

        epocher.init(chain_dataloader=iter(self.unlabeled_loader), reg_weight=reg_weight,
                     projectors_wrapper=projectors_wrapper, infoNCE_criterion=infoNCE_criterion)
        epocher.run()

    def test_uda_iictrainepocher(self):
        from semi_seg.epochers.pretrain import UDAIICPretrainEpocher
        epocher = UDAIICPretrainEpocher(model=self.net, optimizer=self.optimizer, labeled_loader=self.labeled_loader,
                                        unlabeled_loader=self.unlabeled_loader, sup_criterion=nn.MSELoss(),
                                        num_batches=self._num_batches,
                                        cur_epoch=0, device="cuda", feature_position=self._feature_position,
                                        feature_importance=self._feature_importance)
        iic_weight = 1
        uda_weight = 1
        projectors_wrapper: ClusterProjectorWrapper = ClusterProjectorWrapper()  # noqa
        projectors_wrapper.init_encoder(self._feature_position, )
        projectors_wrapper.init_decoder(self._feature_position)
        IIDSegCriterionWrapper: IICLossWrapper = IICLossWrapper(self._feature_position, paddings=0, patch_sizes=1024)
        reg_criterion = nn.MSELoss()

        epocher.init(chain_dataloader=iter(self.unlabeled_loader), iic_weight=iic_weight, uda_weight=uda_weight,
                     projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=IIDSegCriterionWrapper, reg_criterion=reg_criterion)
        epocher.run()

    def test_iictrainerepocher(self):
        from semi_seg.epochers.pretrain import MIPretrainEpocher
        epocher = MIPretrainEpocher(model=self.net, optimizer=self.optimizer, labeled_loader=self.labeled_loader,
                                    unlabeled_loader=self.unlabeled_loader, sup_criterion=nn.MSELoss(),
                                    num_batches=self._num_batches, cur_epoch=0, device="cuda",
                                    feature_position=self._feature_position,
                                    feature_importance=self._feature_importance)

        reg_weight: float = 1
        projectors_wrapper: ClusterProjectorWrapper = ClusterProjectorWrapper()  # noqa
        projectors_wrapper.init_encoder(self._feature_position)
        projectors_wrapper.init_decoder(self._feature_position)

        IIDSegCriterionWrapper: IICLossWrapper = IICLossWrapper(self._feature_position, paddings=0, patch_sizes=1024)

        epocher.init(chain_dataloader=iter(self.unlabeled_loader), reg_weight=reg_weight,
                     projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=IIDSegCriterionWrapper)
        epocher.run()
