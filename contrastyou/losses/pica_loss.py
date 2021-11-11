import math

import torch
from deepclustering2.utils import simplex
from torch import nn
from torch.nn import functional as F


class PUILoss(nn.Module):
    """classification loss"""

    def __init__(self, lamda=2.0):
        super(PUILoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.lamda = lamda

    def forward(self, x, y):
        """Partition Uncertainty Index

        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        assert x.shape == y.shape, ('Inputs are required to have same shape')

        # partition uncertainty index
        assert simplex(x) and simplex(y)
        pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        loss_ce = self.xentropy(pui, torch.arange(pui.size(0), device=x.device))

        # balance regularisation
        p = x.mean(0).view(-1)
        loss_ne = math.log(p.size(0)) + (p * p.log()).sum()  # this is the entropy loss.

        return loss_ce + self.lamda * loss_ne


class PUISegLoss(nn.Module):
    def __init__(self, lamda=2.0, padding=3):
        super(PUISegLoss, self).__init__()
        self.lamda = lamda
        self.padding = padding

    def forward(self, x_out, x_tf_out):
        """Partition Uncertainty Index

        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        assert x_out.shape == x_tf_out.shape, ('Inputs are required to have same shape')

        x_out = x_out.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        x_tf_out = x_tf_out.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
        p_i_j = F.conv2d(x_out, weight=x_tf_out, padding=(self.padding, self.padding))
        p_i_j = p_i_j - p_i_j.min().detach() + 1e-16

        # T x T x k x k
        p_i_j = p_i_j.permute(2, 3, 0, 1)
        p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)  # norm

        # symmetrise, transpose the k x k part
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
        p_i_j = p_i_j.mean(dim=[0, 1])

        loss_ce = self.kl(p_i_j)

        # balance regularisation
        p = x_out.mean(0).view(-1)
        loss_ne = math.log(p.size(0)) + (p * p.log()).sum()  # this is the entropy loss.

        return loss_ce + self.lamda * loss_ne

    def kl(self, joint_p):
        diagnal = torch.eye(joint_p.size(0), device=joint_p.device, dtype=torch.float)
        loss = (-diagnal * torch.log(joint_p + 1e-16)).mean()
        return loss
