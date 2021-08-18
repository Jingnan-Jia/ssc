# -*- coding: utf-8 -*-
# @Time    : 7/26/21 6:59 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import torch.nn as nn
from ssc_scoring.mymodules.networks.kd import get_enc_t


class EncPlusConv(nn.Module):
    def __init__(self, enc_s, conv):
        super().__init__()
        self.enc_s = enc_s
        self.conv = conv

    def __call__(self, x):
        out = self.enc_s(x)
        out = self.conv(out)
        return out




class GetEncSConv:
    """
    Using class method to make sure the network only be created once and reusable many times.
    """
    enc_t = None
    enc_s = None
    enc_plus_conv = None
    def get(cls, enc_t: nn.Module, enc_s: nn.Module, dims: int = 3):
        if cls.enc_t is None and cls.enc_s is None:
            cls.enc_t = enc_t
            cls.enc_s = enc_s
            if dims == 3:
                input_tmp = torch.ones((2, 1, 128, 128, 128))
                out_t = cls.enc_t(input_tmp)
                out_s = cls.enc_s(input_tmp)
                batch_size, chn_t, dim1_t, dim2_t, dim3_t = out_t.shape
                batch_size, chn_s, dim1_s, dim2_s, dim3_s = out_s.shape
                # o = (n - f + 2 * p) / s + 1
                # f = n - ((o - 1) * s - 2 * p)
                conv_sz1 = dim1_s - ((dim1_t - 1) * 1 - 2 * 1)  # stride = 1, padding = 1
                conv_sz2 = dim2_s - ((dim2_t - 1) * 1 - 2 * 1)  # stride = 1, padding = 1
                conv_sz3 = dim3_s - ((dim3_t - 1) * 1 - 2 * 1)  # stride = 1, padding = 1
                assert conv_sz1 == conv_sz2 == conv_sz3
                conv = nn.Conv3d(chn_s, chn_t, kernel_size=conv_sz1, stride=1, padding=1)
            else:
                input_tmp = torch.ones((2, 1, 128, 128))
                out_t = cls.enc_t(input_tmp)
                out_s = cls.enc_s(input_tmp)
                batch_size, chn_t, dim1_t, dim2_t = out_t.shape
                batch_size, chn_s, dim1_s, dim2_s = out_s.shape
                # o = (n - f + 2 * p) / s + 1
                # f = n - ((o - 1) * s - 2 * p)
                conv_sz1 = dim1_s - ((dim1_t - 1) * 1 - 2 * 1)  # stride = 1, padding = 1
                conv_sz2 = dim2_s - ((dim2_t - 1) * 1 - 2 * 1)  # stride = 1, padding = 1
                assert conv_sz1 == conv_sz2
                conv = nn.Conv2d(chn_s, chn_t, kernel_size=conv_sz1, stride=1, padding=1)

            cls.enc_plus_conv = EncPlusConv(cls.enc_s, conv)
        return cls.enc_plus_conv

def kd_loss(batch_x: torch.Tensor,
            enc_s: nn.Module,
            net_t_name: str ='resnet3d'):
    if len(batch_x.shape) == 5:
        dims: int = 3
    elif len(batch_x.shape) == 4:
        dims = 2
    else:
        raise Exception(f'shape of batch_x: {batch_x.shape} is not correct')

    enc_t = get_enc_t(net_t_name)
    enc_s = GetEncSConv().get(enc_t, enc_s, dims)
    with torch.no_grad():
        out_t = enc_t(batch_x)
    out_s = enc_s(batch_x)
    loss = nn.MSELoss()
    kd_loss = loss(out_t, out_s)
    return kd_loss

