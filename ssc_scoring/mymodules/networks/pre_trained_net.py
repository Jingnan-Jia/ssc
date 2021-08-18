# -*- coding: utf-8 -*-
# @Time    : 7/26/21 7:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
from torch import nn
from ssc_scoring.mymodules.networks import med3d_resnet as resnet

def generate_model(opt):
    """

    :param opt: require following values: model, model_depth, input_W, input_H, input_D,
    resnet_shortcut, no_cuda, n_seg_classes, phase, pretrain_path
    :return:
    """
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)

    # if not opt.no_cuda:
    #     if len(opt.gpu_id) > 1:
    #         model = model.cuda()
    #         model = nn.DataParallel(model, device_ids=opt.gpu_id)
    #         net_dict = model.state_dict()
    #     else:
    #         import os
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
    #         model = model.cuda()
    #         model = nn.DataParallel(model, device_ids=None)
    #         net_dict = model.state_dict()
    # else:
    #     net_dict = model.state_dict()

    # # load pretrain
    # if opt.pretrain_path:
    #     print('loading pretrained model {}'.format(opt.pretrain_path))
    #     pretrain = torch.load(opt.pretrain_path)
    #     pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    #
    #     net_dict.update(pretrain_dict)
    #     model.load_state_dict(net_dict)


    return model
