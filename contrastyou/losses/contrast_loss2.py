import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from deepclustering2.configparser._utils import get_config  # noqa
from deepclustering2.meters2 import AverageValueMeter
from deepclustering2.schedulers.customized_scheduler import LinearScheduler, ExpScheduler, InverseExpScheduler
from deepclustering2.writer import SummaryWriter
from loguru import logger
from torch import Tensor, nn


@contextmanager
def switch_plt_backend(env="agg"):
    prev = matplotlib.get_backend()
    matplotlib.use(env, force=True)
    yield
    matplotlib.use(prev, force=True)


def is_normalized(feature: Tensor, dim=1):
    norms = feature.norm(dim=dim)
    return torch.allclose(norms, torch.ones_like(norms))


def exp_sim_temperature(proj_feat1: Tensor, proj_feat2: Tensor, t: float) -> Tuple[Tensor, Tensor]:
    projections = torch.cat([proj_feat1, proj_feat2], dim=0)
    sim_logits = torch.mm(projections, projections.t().contiguous()) / t
    max_value = sim_logits.max().detach()
    sim_logits -= max_value
    sim_exp = torch.exp(sim_logits)
    return sim_exp, sim_logits


class SupConLoss1(nn.Module):
    def __init__(self, temperature=0.07, exclude_other_pos=False):
        super().__init__()
        self._t = temperature
        self._exclude_pos = exclude_other_pos
        logger.info(f"initializing {self.__class__.__name__} with t: {self._t}, exclude_pos: {self._exclude_pos}")

    def forward(self, proj_feat1, proj_feat2, target=None, mask: Tensor = None, **kwargs):
        batch_size = proj_feat1.size(0)
        if mask is not None:
            assert mask.shape == torch.Size([batch_size, batch_size])
            pos_mask = mask == 1
            neg_mask = mask == 0

        elif target is not None:
            if isinstance(target, list):
                target = torch.Tensor(target).to(device=proj_feat2.device)
            mask = torch.eq(target[..., None], target[None, ...])

            pos_mask = mask == True
            neg_mask = mask == False
        else:
            # only postive masks are diagnal of the sim_matrix
            pos_mask = torch.eye(batch_size, dtype=torch.float, device=proj_feat2.device)  # SIMCLR
            neg_mask = 1 - pos_mask
        return self._forward(proj_feat1, proj_feat2, pos_mask.float(), neg_mask.float(), **kwargs)

    def _forward(self, proj_feat1, proj_feat2, pos_mask, neg_mask, **kwargs):
        """
        Here the proj_feat1 and proj_feat2 should share the same mask within and cross proj_feat1 and proj_feat2
        :param proj_feat1:
        :param proj_feat2:
        :return:
        """
        assert is_normalized(proj_feat1) and is_normalized(proj_feat2), f"features need to be normalized first"
        assert proj_feat1.shape == proj_feat2.shape, (proj_feat1.shape, proj_feat2.shape)

        batch_size = len(proj_feat1)
        unselect_diganal_mask = 1 - torch.eye(
            batch_size * 2, batch_size * 2, dtype=torch.float, device=proj_feat2.device
        )

        # upscale
        pos_mask = pos_mask.repeat(2, 2)
        neg_mask = neg_mask.repeat(2, 2)

        pos_mask *= unselect_diganal_mask
        neg_mask *= unselect_diganal_mask

        # 2n X 2n
        sim_exp, sim_logits = exp_sim_temperature(proj_feat1, proj_feat2, self._t)
        assert pos_mask.shape == sim_exp.shape == neg_mask.shape, (pos_mask.shape, sim_exp.shape, neg_mask.shape)

        # =============================================
        # in order to have a hook for further processing
        self.sim_exp = sim_exp
        self.sim_logits = sim_logits
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask
        # ================= end =======================
        pos_count, neg_count = pos_mask.sum(1), neg_mask.sum(1)
        pos_sum = (sim_exp * pos_mask).sum(1, keepdim=True).repeat(1, batch_size * 2)
        neg_sum = (sim_exp * neg_mask).sum(1, keepdim=True).repeat(1, batch_size * 2)
        if self._exclude_pos:
            neg_ratio = neg_count.float() / (pos_count + neg_count).float()
            log_pos_div_sum_pos_neg = sim_logits - torch.log(
                sim_exp + neg_sum / (neg_ratio + 1e-4)[..., None].repeat(1, batch_size * 2) + 1e-16)
        else:
            log_pos_div_sum_pos_neg = sim_logits - torch.log(pos_sum + neg_sum + 1e-16)

        # over positive mask
        loss = (log_pos_div_sum_pos_neg * pos_mask).sum(1) / pos_count
        loss = -loss.mean()

        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss

    @contextmanager
    def register_writer(self, writer: SummaryWriter, epoch=0, extra_tag=None):
        yield
        sim_exp = self.sim_exp.detach().cpu().numpy()
        sim_logits = self.sim_logits.detach().cpu().numpy()
        pos_mask = self.pos_mask.detach().cpu().numpy()
        neg_mask = self.neg_mask.detach().cpu().numpy()
        with switch_plt_backend("agg"):
            fig1 = plt.figure()
            plt.imshow(sim_exp, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "sim_exp"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)
            fig1 = plt.figure()
            plt.imshow(sim_logits, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "sim_logits"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)
            fig1 = plt.figure()
            plt.imshow(pos_mask, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "pos_mask"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)
            fig1 = plt.figure()
            plt.imshow(neg_mask, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "neg_mask"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)

    def epoch_start(self):
        pass

    def epoch_end(self):
        return {}


class SelfPacedSupConLoss(nn.Module):
    def __repr__(self):
        message = f"{self.__class__.__name__} with T: {self._t}, method: {self._weight_update} gamma: {self.__gamma}"
        return message

    def __init__(self, temperature=0.07, weight_update="hard", type="linear", begin_value=1e6, end_value=1e6,
                 **kwargs):
        super().__init__()
        self._t = temperature
        self._weight_update = weight_update
        self.__gamma = 1e6
        self._scheduler_type = type
        logger.info(f"initializing {self.__class__.__name__} with t: {self._t} ")
        self._chosen_percentage_meter = AverageValueMeter()
        self._real_chosen_percentage_meter = AverageValueMeter()
        config = deepcopy(get_config(scope="base"))
        scheduler_class = {"linear": LinearScheduler, "square": ExpScheduler, "inversesquare": InverseExpScheduler}[
            type]
        logger.debug(f"creating {scheduler_class} scheduler for {self.__class__.__name__}")
        self._scheduler = scheduler_class(max_epoch=config["Trainer"]["max_epoch"], begin_value=begin_value,
                                          end_value=end_value)

        self._scheduler_tracker = AverageValueMeter()

    def forward(self, proj_feat1, proj_feat2, target=None, mask: Tensor = None, **kwargs):
        batch_size = proj_feat1.size(0)
        if mask is not None:
            assert mask.shape == torch.Size([batch_size, batch_size])
            pos_mask = mask == 1
            neg_mask = mask == 0

        elif target is not None:
            if isinstance(target, list):
                target = torch.Tensor(target).to(device=proj_feat2.device)
            mask = torch.eq(target[..., None], target[None, ...])

            pos_mask = mask == True
            neg_mask = mask == False
        else:
            # only postive masks are diagnal of the sim_matrix
            pos_mask = torch.eye(batch_size, dtype=torch.float, device=proj_feat2.device)  # SIMCLR
            neg_mask = 1 - pos_mask
        gamma = self.__gamma
        return self._forward(proj_feat1, proj_feat2, pos_mask.float(), neg_mask.float(), gamma=gamma, **kwargs)

    def _forward(self, proj_feat1, proj_feat2, pos_mask, neg_mask, gamma=1e6, **kwargs):
        """
        Here the proj_feat1 and proj_feat2 should share the same mask within and cross proj_feat1 and proj_feat2
        :param proj_feat1:
        :param proj_feat2:
        :return:
        """
        assert is_normalized(proj_feat1) and is_normalized(proj_feat2), f"features need to be normalized first"
        assert proj_feat1.shape == proj_feat2.shape, (proj_feat1.shape, proj_feat2.shape)

        batch_size = len(proj_feat1)
        unselect_diganal_mask = 1 - torch.eye(
            batch_size * 2, batch_size * 2, dtype=torch.float, device=proj_feat2.device
        )

        # upscale
        pos_mask = pos_mask.repeat(2, 2)
        neg_mask = neg_mask.repeat(2, 2)

        pos_mask *= unselect_diganal_mask
        neg_mask *= unselect_diganal_mask

        # 2n X 2n
        sim_exp, sim_logits = exp_sim_temperature(proj_feat1, proj_feat2, self._t)
        assert pos_mask.shape == sim_exp.shape == neg_mask.shape, (pos_mask.shape, sim_exp.shape, neg_mask.shape)

        # =============================================
        # in order to have a hook for further processing
        self.sim_exp = sim_exp
        self.sim_logits = sim_logits
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask
        # ================= end =======================
        pos_count, neg_count = pos_mask.sum(1), neg_mask.sum(1)
        pos_sum = (sim_exp * pos_mask).sum(1, keepdim=True).repeat(1, batch_size * 2)
        neg_sum = (sim_exp * neg_mask).sum(1, keepdim=True).repeat(1, batch_size * 2)

        log_pos_div_sum_pos_neg = sim_logits - torch.log(pos_sum + neg_sum + 1e-16)
        assert log_pos_div_sum_pos_neg.shape == torch.Size([batch_size * 2, batch_size * 2])

        log_pos_div_sum_pos_neg = self._selfpaced_weighted_loss(log_pos_div_sum_pos_neg, gamma, pos_mask=pos_mask)

        # over positive mask
        loss = (log_pos_div_sum_pos_neg * pos_mask).sum(1) / pos_count
        loss = -loss.mean()

        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss

    def _selfpaced_weighted_loss(self, loglikelihoodmatrix, gamma, *, pos_mask):
        l_i_j = -loglikelihoodmatrix.detach()
        weight: Tensor
        if self._weight_update == "hard":
            weight = (l_i_j <= gamma).float()
        else:
            weight = torch.max(1 - 1 / gamma * l_i_j, torch.zeros_like(l_i_j))
        self.weight = weight  # * pos_mask
        assert torch.logical_and(weight >= 0, weight <= 1).any()
        self._chosen_percentage_meter.add(weight.mean().item())

        self._real_chosen_percentage_meter.add(torch.masked_select(weight, pos_mask.bool()).mean().item())

        return loglikelihoodmatrix * weight

    @contextmanager
    def register_writer(self, writer: SummaryWriter, epoch=0, extra_tag=None):
        yield
        sim_exp = self.sim_exp.detach().cpu().numpy()
        sim_logits = self.sim_logits.detach().cpu().numpy()
        pos_mask = self.pos_mask.detach().cpu().numpy()
        neg_mask = self.neg_mask.detach().cpu().numpy()
        weight_mask = self.weight.detach().cpu().numpy()
        with switch_plt_backend("agg"):
            fig1 = plt.figure()
            plt.imshow(sim_exp, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "sim_exp"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)
            fig1 = plt.figure()
            plt.imshow(sim_logits, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "sim_logits"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)
            fig1 = plt.figure()
            plt.imshow(pos_mask, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "pos_mask"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)
            fig1 = plt.figure()
            plt.imshow(neg_mask, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "neg_mask"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)
            fig1 = plt.figure()
            plt.imshow(weight_mask, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "self-paced"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)

    def epoch_start(self):
        gamma = self._scheduler.value
        self._scheduler_tracker.reset()
        self._scheduler_tracker.add(gamma)
        self.set_gamma(gamma)
        self._chosen_percentage_meter.reset()
        self._real_chosen_percentage_meter.reset()

    def epoch_end(self):
        self._scheduler.step()
        return {"chosen_percentage": self._chosen_percentage_meter.summary(),
                "real_percentage": self._real_chosen_percentage_meter.summary(),
                "gamma": self._scheduler_tracker.summary()}

    def set_gamma(self, gamma):
        logger.debug(f"{self.__class__.__name__} set gamma as {gamma}")
        self.__gamma = gamma


if __name__ == '__main__':
    """ verify the SupContrastLoss1 
    from torch.nn.functional import normalize

    feature1 = normalize(torch.randn(10, 256, device="cuda"), dim=1)
    feature2 = normalize(torch.randn(10, 256, device="cuda"), dim=1)
    target = [random.choice([0, 1, 2]) for i in range(10)]
    criterion1 = SupConLoss(temperature=0.07, base_temperature=0.07)
    criterion2 = SupConLoss1(temperature=0.07, exclude_other_pos=True)
    loss1 = criterion1(torch.stack([feature1, feature2], dim=1), labels=target)
    loss2 = criterion2(feature1, feature2, target=target)
    assert loss1.allclose(loss2), (loss1, loss2)
    """
    """ verify the Self-pacedSupcontrast loss"""
    from torch.nn.functional import normalize

    anchor1 = torch.randn(1, 256, device="cuda")
    anchor2 = torch.randn(1, 256, device="cuda")
    anchor3 = torch.randn(1, 256, device="cuda")

    feature1 = torch.cat([anchor1 * (1 - alpha) + anchor2 * alpha for alpha in torch.linspace(0, 1, steps=100)], dim=0)
    feature2 = torch.cat([anchor1 * (1 - alpha) + anchor3 * alpha for alpha in torch.linspace(0, 1, steps=100)], dim=0)
    feature1, feature2 = normalize(feature1, ), normalize(feature2)

    target = [random.choice([0, 1, 2]) for i in range(100)]
    self_paced_criterion = SelfPacedSupConLoss(temperature=0.07, weight_update="hard")
    loss1 = self_paced_criterion(feature1, feature2, target=target, gamma=1e6)
    criterion2 = SupConLoss1(temperature=0.07)
    loss2 = criterion2(feature1, feature2, target=target)
    assert torch.allclose(loss2, loss1)
