from contextlib import contextmanager
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from deepclustering2.writer import SummaryWriter


@contextmanager
def _switch_plt_backend(env="agg"):
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


class SupConLoss2(nn.Module):
    def __init__(self, temperature=0.07, out_mode=True):
        super().__init__()
        self._t = temperature
        self._out_mode = out_mode

    def forward(self, proj_feat1, proj_feat2, target=None, mask: Tensor = None):
        """
        Here the proj_feat1 and proj_feat2 should share the same mask within and cross proj_feat1 and proj_feat2
        :param proj_feat1:
        :param proj_feat2:
        :param target:
        :param mask:
        :return:
        """
        assert is_normalized(proj_feat1) and is_normalized(proj_feat2), f"features need to be normalized first"
        assert proj_feat1.shape == proj_feat2.shape, (proj_feat1.shape, proj_feat2.shape)

        if (target is not None) and (mask is not None):
            raise RuntimeError(f"`target` and `mask` should not be provided in the same time")

        batch_size = len(proj_feat1)
        sim_exp, sim_logits = exp_sim_temperature(proj_feat1, proj_feat2, self._t)

        unselect_diganal_mask = 1 - torch.eye(
            batch_size * 2, batch_size * 2, dtype=torch.float, device=proj_feat2.device)

        # build negative examples
        if mask is not None:
            assert mask.shape == torch.Size([batch_size, batch_size])
            mask = mask.repeat(2, 2)
            pos_mask = mask == 1
            neg_mask = mask == 0

        elif target is not None:
            if isinstance(target, list):
                target = torch.Tensor(target).to(device=proj_feat2.device)
            mask = torch.eq(target[..., None], target[None, ...])
            mask = mask.repeat(2, 2)

            pos_mask = mask == True
            neg_mask = mask == False
        else:
            # only postive masks are diagnal of the sim_matrix
            pos_mask = torch.eye(batch_size, dtype=torch.float, device=proj_feat2.device)  # SIMCLR
            pos_mask = pos_mask.repeat(2, 2)
            neg_mask = 1 - pos_mask

        # in order to have a hook for further processing
        self.sim_exp = sim_exp
        self.sim_logits = sim_logits
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask
        # ================= end =======================

        pos_mask = pos_mask * unselect_diganal_mask
        neg_mask = neg_mask * unselect_diganal_mask
        pos = sim_exp * pos_mask
        negs = sim_exp * neg_mask
        pos_count = pos_mask.sum(1)
        if not self._out_mode:
            # this is the in mode
            loss = (- torch.log(pos.sum(1) / (pos.sum(1) + negs.sum(1))) / pos_count).mean()
        # this is the out mode
        else:
            log_pos_div_sum_pos_neg = (sim_logits - torch.log((pos + negs).sum(1, keepdim=True))) * pos_mask
            log_pos_div_sum_pos_neg = log_pos_div_sum_pos_neg.sum(1) / pos_count
            loss = -log_pos_div_sum_pos_neg.mean()
        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss

    @contextmanager
    def register_writer(self, writer: SummaryWriter, epoch=0, extra_tag=None):
        yield
        sim_exp = self.sim_exp.detach().cpu().numpy()
        sim_logits = self.sim_logits.detach().cpu().numpy()
        pos_mask = self.pos_mask.detach().cpu().numpy()
        with _switch_plt_backend("agg"):
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
            dest = "/".join([x for x in [extra_tag, "pos_weight"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)


class SupConLoss3(SupConLoss2):
    """
    Soften supervised contrastive loss
    """

    def forward(self, proj_feat1, proj_feat2, pos_weight: Tensor = None, **kwargs):
        """
        w_{ip} log \frac{exp(z_i * z_p/t)}/{\sum_{a\in A(i)}exp(z_i,z_a/t)}
        :param proj_feat1:
        :param proj_feat2:
        :param pos_weight:
        :param kwargs:
        :return:
        """
        assert is_normalized(proj_feat1) and is_normalized(proj_feat2), f"features need to be normalized first"
        assert proj_feat1.shape == proj_feat2.shape, (proj_feat1.shape, proj_feat2.shape)
        pos_weight: Tensor
        assert pos_weight is not None
        batch_size = len(proj_feat1)

        assert pos_weight.shape == torch.Size([batch_size, batch_size])
        # assert pos_weight.max() <= 1 and pos_weight.min() >= 0, (pos_weight.min(), pos_weight.max())
        [pos_weight, ] = list(map(lambda x: x.repeat(2, 2), [pos_weight, ]))
        unselect_diganal_mask = 1 - torch.eye(
            batch_size * 2, batch_size * 2, dtype=torch.float, device=proj_feat2.device)

        sim_exp, sim_logits = exp_sim_temperature(proj_feat1, proj_feat2, self._t)

        # in order to have a hook for further processing
        self.sim_exp = sim_exp
        self.sim_logits = sim_logits
        self.pos_weight = pos_weight
        # ================= end =======================

        # todo: do you want to weight something here for the denominator?
        denominator = (sim_exp * unselect_diganal_mask).sum(1, keepdim=True)
        exp_div_sum_exp = sim_exp / denominator

        pos_weight = pos_weight * unselect_diganal_mask

        if not self._out_mode:
            # this is the in mode
            loss = torch.log((exp_div_sum_exp * pos_weight).sum(1)) / pos_weight.sum(1)
            loss = -loss.mean()
        else:
            # this is the out mode
            log_pos_div_sum_pos_neg = torch.log(exp_div_sum_exp) * pos_weight
            log_pos_div_sum_pos_neg = log_pos_div_sum_pos_neg.sum(1) / pos_weight.sum(1)
            loss = -log_pos_div_sum_pos_neg.mean()
        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss

    @contextmanager
    def register_writer(self, writer: SummaryWriter, epoch=0, extra_tag=None):
        yield
        sim_exp = self.sim_exp.detach().cpu().numpy()
        sim_logits = self.sim_logits.detach().cpu().numpy()
        pos_weight = self.pos_weight.detach().cpu().numpy()
        with _switch_plt_backend("agg"):
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
            plt.imshow(pos_weight, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "pos_weight"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)


class SupConLoss4(SupConLoss2):

    def forward(self, *, proj_feat1, proj_feat2, one2one_weight: Tensor = None, two2two_weight: Tensor,  # noqa
                one2two_weight=None, **kwargs):  # noqa
        """
        w_{ip} log \frac{exp(z_i * z_p/t)}/{\sum_{a\in A(i)}exp(z_i,z_a/t)}
        :param proj_feat1: normalized feature1
        :param proj_feat2: normalized feature2
        :param one2one_weight: weighted matrix within proj_feat1
        :param two2two_weight: weighted matrix within proj_feat2
        :param one2two_weight: weighted matrix from proj_feat1 to proj_feat2
        :return:
        """
        assert is_normalized(proj_feat1) and is_normalized(proj_feat2), f"features need to be normalized first"
        assert proj_feat1.shape == proj_feat2.shape, (proj_feat1.shape, proj_feat2.shape)
        assert one2one_weight is not None or one2two_weight is not None or two2two_weight is not None
        batch_size = len(proj_feat1)

        pos_weight = torch.zeros(batch_size * 2, batch_size * 2, device=proj_feat2.device, dtype=torch.float)
        enable_mask = torch.zeros_like(pos_weight)
        if one2two_weight is not None:
            pos_weight[:batch_size, :batch_size] = one2one_weight
            enable_mask[:batch_size, :batch_size] = 1
        if two2two_weight is not None:
            pos_weight[batch_size:, batch_size:] = two2two_weight
            enable_mask[batch_size:, batch_size:] = 1
        if one2two_weight is not None:
            pos_weight[:batch_size, batch_size:] = one2two_weight
            pos_weight[batch_size:, :batch_size] = one2two_weight
            enable_mask[:batch_size, batch_size:] = 1
            enable_mask[batch_size:, :batch_size] = 1
        pos_weight = pos_weight.contiguous()
        enable_mask = enable_mask.contiguous()

        unselect_diganal_mask = 1 - torch.eye(
            batch_size * 2, batch_size * 2, dtype=torch.float, device=proj_feat2.device)

        sim_exp, sim_logits = exp_sim_temperature(proj_feat1, proj_feat2, self._t)

        # in order to have a hook for further processing
        self.sim_exp = sim_exp
        self.sim_logits = sim_logits
        self.pos_weight = pos_weight
        self.enable_mask = enable_mask
        # ================= end =======================

        # todo: do you want to weight something here for the denominator?
        denominator = (sim_exp * unselect_diganal_mask * enable_mask).sum(1, keepdim=True)
        exp_div_sum_exp = sim_exp / denominator

        pos_weight = pos_weight * unselect_diganal_mask

        if not self._out_mode:
            # this is the in mode
            loss = torch.log((exp_div_sum_exp * pos_weight).sum(1)) / pos_weight.sum(1)
            loss = -loss.mean()
        else:
            # this is the out mode
            log_pos_div_sum_pos_neg = torch.log(exp_div_sum_exp) * pos_weight
            log_pos_div_sum_pos_neg = log_pos_div_sum_pos_neg.sum(1) / pos_weight.sum(1)
            loss = -log_pos_div_sum_pos_neg.mean()
        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss

    @contextmanager
    def register_writer(self, writer: SummaryWriter, epoch=0, extra_tag=None):
        yield
        sim_exp = self.sim_exp.detach().cpu().numpy()
        sim_logits = self.sim_logits.detach().cpu().numpy()
        pos_weight = self.pos_weight.detach().cpu().numpy()
        enable_mask = self.enable_mask.detach().cpu().numpy()
        with _switch_plt_backend("agg"):
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
            plt.imshow(pos_weight, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "pos_weight"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)
            fig1 = plt.figure()
            plt.imshow(enable_mask, cmap="gray")
            plt.colorbar()
            dest = "/".join([x for x in [extra_tag, "enable_mask"] if x is not None])
            writer.add_figure(tag=dest, figure=fig1, global_step=epoch)


if __name__ == '__main__':
    feature1 = torch.randn(100, 256, device="cuda")
    feature2 = torch.randn(100, 256, device="cuda")
    criterion1 = SupConLoss(temperature=0.07, base_temperature=0.07)
    criterion2 = SupConLoss2(temperature=0.07, out_mode=False)
    criterion3 = SupConLoss2(temperature=0.07, out_mode=True)
    criterion4 = SupConLoss3(temperature=0.07, out_mode=True)
    criterion5 = SupConLoss3(temperature=0.07, out_mode=False)
    mask = torch.randint(0, 2, [100, 100], device="cuda")
    loss2 = criterion2(
        nn.functional.normalize(feature1, dim=1),
        nn.functional.normalize(feature2, dim=1),
        mask=mask
    )
    loss3 = criterion3(
        nn.functional.normalize(feature1, dim=1),
        nn.functional.normalize(feature2, dim=1),
        mask=mask
    )
    loss4 = criterion4(
        nn.functional.normalize(feature1, dim=1),
        nn.functional.normalize(feature2, dim=1),
        pos_weight=mask
    )
    loss5 = criterion5(
        nn.functional.normalize(feature1, dim=1),
        nn.functional.normalize(feature2, dim=1),
        pos_weight=mask
    )
    assert torch.isclose(loss3, loss4), (loss3, loss4)
    assert torch.isclose(loss2, loss5), (loss2, loss5)

    # target = [random.randint(0, 5) for i in range(100)]
    # from torch.cuda import Event
    #
    # start = Event(enable_timing=True, blocking=True)
    # end = Event(enable_timing=True, blocking=True)
    # start.record()
    # loss1 = criterion1(torch.stack(
    #     [nn.functional.normalize(feature1, dim=1),
    #      nn.functional.normalize(feature2, dim=1), ], dim=1
    # ), labels=target)
    # end.record()
    # print(start.elapsed_time(end))
    #
    # start = Event(enable_timing=True, blocking=True)
    # end = Event(enable_timing=True, blocking=True)
    # start.record()
    # loss2 = criterion2(
    #     nn.functional.normalize(feature1, dim=1),
    #     nn.functional.normalize(feature2, dim=1),
    #     target=target
    # )
    # end.record()
    # print(start.elapsed_time(end))
    #
    # start = Event(enable_timing=True, blocking=True)
    # end = Event(enable_timing=True, blocking=True)
    # start.record()
    # loss3 = criterion3(
    #     nn.functional.normalize(feature1, dim=1),
    #     nn.functional.normalize(feature2, dim=1),
    #     target=target
    # )
    # end.record()
    # print(start.elapsed_time(end))
    #
    # assert torch.allclose(loss1, loss2) and torch.allclose(loss3, loss1), (loss1, loss2, loss3)
