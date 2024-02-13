import torch
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.bti_loss import BTI_Loss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from nnunetv2.training.loss.tvmf_dice_loss import Adaptive_tvMF_DiceLoss

class DC_and_CE_and_BTI_Loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, ti_kwargs, weight_ce=1, weight_dice=1, weight_ti=1e-6, ignore_label=None,
                 dice_class=Adaptive_tvMF_DiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_BTI_Loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_ti = weight_ti
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = Adaptive_tvMF_DiceLoss(n_classes=14)
        self.ti = BTI_Loss(**ti_kwargs)

    ## kappa - tvMF DICE
    def forward(self, net_output: torch.Tensor, target: torch.Tensor, kappa: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        ## KAPPA PENTRU TVMF DICE, SCOATE-L PT ORICE ALTCEVA
        dc_loss = self.dc(net_output, target_dice, kappa) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        bti_loss = self.ti(net_output, target) if self.weight_ti != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_ti * bti_loss
        return result

