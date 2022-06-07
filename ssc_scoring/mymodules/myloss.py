# -*- coding: utf-8 -*-
# @Time    : 7/5/21 7:50 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import torch
import torch.nn as nn


class MSEHigher(nn.Module):
    """This loss aims to penalty more when pred is greater than ground truth. But it would make the training not
    stable and results is not better from my own experiments.

    .. warning::
        I am not sure if the implementation is correct. Should if-else condition be in `forward` function.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        if torch.sum(y_pred) > torch.sum(y_true):
            loss = self.mse(y_pred, y_true)
            print('normal loss')
        else:
            loss = self.mse(y_pred, y_true) * 5  # 5 could be changed
            print("higher loss")

        return loss


class MsePlusMae(nn.Module):
    """MSE + MAE"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, y_pred, y_true):
        mse = self.mse(y_pred, y_true)
        mae = self.mae(y_pred, y_true)
        print(f"mse loss: {mse}, mae loss: {mae}")
        return mse + mae


def get_loss(loss: str) -> nn.Module:
    """Return loss from its name

    Args:
        loss: The name of loss function

    Examples:

    >>> loss = get_loss('mse')

    """

    if loss == 'mae':
        loss_fun = nn.L1Loss()
    elif loss == 'smooth_mae':
        loss_fun = nn.SmoothL1Loss()
    elif loss == 'mse':
        loss_fun = nn.MSELoss()
    elif loss == 'mse+mae':
        loss_fun = nn.MSELoss() + nn.L1Loss()  # for regression task
    elif loss == 'msehigher':
        loss_fun = MSEHigher()
    else:
        raise Exception("loss function is not correct " + loss)
    return loss_fun
