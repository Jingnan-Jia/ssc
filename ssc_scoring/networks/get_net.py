# -*- coding: utf-8 -*-
# @Time    : 7/5/21 9:27 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

from .cnn_fc3d import Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d

def get_net_pos(name: str, nb_cls: int, fc1_nodes=1024, fc2_nodes=1024, level_node = 0):
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
    else:
        raise Exception('wrong net name', name)

    return net