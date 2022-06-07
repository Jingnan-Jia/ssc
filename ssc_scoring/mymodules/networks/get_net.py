# -*- coding: utf-8 -*-
# @Time    : 7/5/21 9:27 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

from .cnn_fc3d import Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d
from .cnn_fc3d_enc import Cnn3fc1Enc, Cnn3fc2Enc, Cnn4fc2Enc, Cnn5fc2Enc, Cnn6fc2Enc, Vgg11_3dEnc
import torch
import torchvision

def get_net_pos(name: str, nb_cls: int, fc1_nodes=1024, fc2_nodes=1024, level_node = 0, pretrained=True):
    if name == 'cnn3fc1':
        net = Cnn3fc1(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn3fc2':
        net = Cnn3fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn4fc2':
        net = Cnn4fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn5fc2':
        net = Cnn5fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn6fc2':
        net = Cnn6fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == "vgg11_3d":
        net = Vgg11_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == "r3d_18":
        net = torchvision.models.video.r3d_18(pretrained = pretrained, progress  = True)

        class BasicStem(torch.nn.Sequential):
            """The default conv-batchnorm-relu stem"""

            def __init__(self) -> None:
                super().__init__(
                    torch.nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
                    torch.nn.BatchNorm3d(64),
                    torch.nn.ReLU(inplace=True),
                )
        net.stem = BasicStem()
        net.fc = torch.nn.Linear(512 * 1, nb_cls)
    elif name == "slow_r50":
        net = torch.hub.load('facebookresearch/pytorchvideo', name , pretrained=pretrained)
    elif name == "slowfast_r50":
        net = torch.hub.load('facebookresearch/pytorchvideo', name, pretrained=pretrained)
    elif "x3d" in name:
        if name in ("x3d_xs", "x3d_s", "x3d_m", "x3d_l"):
            net = torch.hub.load('facebookresearch/pytorchvideo', name, pretrained=pretrained)
        else:
            raise Exception(f"wrong net name : {name}")
    else:
        raise Exception('wrong net name', name)

    return net


def get_net_pos_enc(name: str, nb_cls: int, fc1_nodes=1024, fc2_nodes=1024, level_node = 0):
    if name == 'cnn3fc1':
        net = Cnn3fc1Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn3fc2':
        net = Cnn3fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn4fc2':
        net = Cnn4fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn5fc2':
        net = Cnn5fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == 'cnn6fc2':
        net = Cnn6fc2Enc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    elif name == "vgg11_3d":
        net = Vgg11_3dEnc(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls, level_node=level_node)
    else:
        raise Exception('wrong net name', name)

    return net