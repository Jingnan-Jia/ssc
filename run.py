# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import copy
import csv
import datetime
import glob
import itertools
import os
import random
import shutil
import threading
import time
from collections import OrderedDict
from typing import (List, Tuple, Optional, Union, Dict, Sequence)

import jjnutils.util as futil
import numpy as np
import nvidia_smi
import pandas as pd
import torch
import torch.nn as nn
# import streamlit as st
import torchvision.models as models
from filelock import FileLock
from monai.transforms import ScaleIntensityRange, RandGaussianNoise, MapTransform
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomAffine
from tqdm import tqdm

import confusion
from set_args import args

LogType = Optional[Union[int, float, str]]  # int includes bool
log_dict: Dict[str, LogType] = {}  # a global dict to store variables saved to log files


def summary(model: nn.Module, input_size: Tuple[int, ...], batch_size=-1, device="cpu"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params: Union[torch.Tensor, int] = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return summary


class ReconNet(nn.Module):
    def __init__(self, reg_net, input_size=512):
        super().__init__()
        self.reg_net = reg_net
        self.features = copy.deepcopy(reg_net.features)  # encoder
        self.enc_dec = self._build_dec_from_enc()

    def _last_channels(self):
        last_chn = None
        for layer in self.features[::-1]:
            if isinstance(layer, torch.nn.Conv2d):
                last_chn = layer.out_channels
                break
        if last_chn is None:
            raise Exception("No convolution layers at all in regression network")
        return last_chn

    def _build_dec_from_enc(self):
        decoder_ls = []
        in_channels = None  # statement for convtransposed
        last_chns = self._last_channels()

        layer_shapes = summary(self.features, (1, 512, 512))  # ordered dict saving shapes of each layer
        transit_chn = 0
        for layer, (layer_name, layer_shape) in zip(self.features[::-1], reversed(layer_shapes.items())):
            if transit_chn == 0:
                transit_chn = layer_shape['input_shape'][1]

            if isinstance(layer, torch.nn.Conv2d):

                enc_in_channels = layer_shape['input_shape'][1]
                # enc_out_channels = layer_shape['output_shape'][1]

                enc_kernel_size: int = layer.kernel_size[0]  # square kernel, get one of the sizes
                enc_stride: int = layer.stride[0]

                if enc_stride > 1:  # shape is reduced
                    decoder_ls.append(nn.Upsample(scale_factor=enc_stride, mode='bilinear'))
                    decoder_ls.append(nn.Conv2d(transit_chn, enc_in_channels, enc_kernel_size,
                                                padding=enc_kernel_size - enc_kernel_size // 2 - 1))
                else:
                    decoder_ls.append(nn.Conv2d(transit_chn, enc_in_channels, enc_kernel_size,
                                                padding=enc_kernel_size - enc_kernel_size // 2 - 1))

                decoder_ls.extend([nn.BatchNorm2d(enc_in_channels),
                                   nn.ReLU(inplace=True)])

                transit_chn = enc_in_channels  # new value

            elif isinstance(layer, torch.nn.MaxPool2d):
                decoder_ls.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                decoder_ls.append(nn.Conv2d(transit_chn, transit_chn, 3, padding=1))
                decoder_ls.extend([nn.BatchNorm2d(transit_chn),
                                   nn.ReLU(inplace=True)])

            else:
                pass
        # correct the shape of the final output
        while (isinstance(decoder_ls[-1], nn.ReLU)) or (isinstance(decoder_ls[-1], nn.BatchNorm2d)):
            decoder_ls.pop()
        decoder = nn.Sequential(*decoder_ls)

        # class EncDoc(nn.Module):
        #     def __init__(self, enc, dec):
        #         super().__init__()
        #         self.enc = enc
        #         self.dec = dec
        #
        #     def forward(self, x):
        #         out = self.enc(x)
        #         out = self.dec(out)
        #         return out
        #
        # enc_dec = EncDoc(self.features, decoder)
        enc_dec = nn.Sequential(self.features, decoder)
        enc_dec_layer_shapes = summary(enc_dec, (1, 512, 512), device='cpu')
        input_sz = list(iter(enc_dec_layer_shapes.items()))[0][-1]['input_shape'][-1]
        output_sz = list(iter(enc_dec_layer_shapes.items()))[-1][-1]['input_shape'][-1]
        dif: int = input_sz - output_sz
        if dif > 0:  # the last output of decoder is less than the first output of encoder, need pad
            enc_dec = nn.Sequential(enc_dec,
                                    nn.Upsample(size=input_sz, mode="bilinear"),
                                    nn.Conv2d(1, 1, 3, padding=1))

        # decoder_dict = OrderedDict([(key, value) for key, value in zip(range(len(decoder_ls)), decoder_ls)])
        tmp = summary(enc_dec, (1, 512, 512), device='cpu')
        return enc_dec

    def forward(self, x):
        out = self.enc_dec(x)

        return out


class Cnn3fc1(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn2fc1_old(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn2fc1(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)

        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, args.fc_m1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc_m1, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_net(name: str, nb_cls: int):
    if 'vgg' in name:
        if name == 'vgg11_bn':
            net = models.vgg11_bn(pretrained=args.pretrained, progress=True)
        elif name == 'vgg16':
            net = models.vgg16(pretrained=args.pretrained, progress=True)
        elif name == 'vgg19':
            net = models.vgg19(pretrained=args.pretrained, progress=True)
        else:
            raise Exception("Wrong vgg net name specified ", name)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # change in_features to 1
        net.classifier[0] = torch.nn.Linear(in_features=512 * 7 * 7, out_features=args.fc1_nodes)
        net.classifier[3] = torch.nn.Linear(in_features=args.fc1_nodes, out_features=args.fc2_nodes)
        net.classifier[6] = torch.nn.Linear(in_features=args.fc2_nodes, out_features=3)
    elif name == 'alex':
        net = models.alexnet(pretrained=args.pretrained, progress=True)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        net.classifier[1] = torch.nn.Linear(in_features=256 * 6 * 6, out_features=args.fc1_nodes)
        net.classifier[4] = torch.nn.Linear(in_features=args.fc1_nodes, out_features=args.fc2_nodes)
        net.classifier[6] = torch.nn.Linear(in_features=args.fc2_nodes, out_features=3)
    elif name == 'cnn3fc1':
        net = Cnn3fc1(num_classes=3)
    elif name == 'cnn2fc1':
        net = Cnn2fc1(num_classes=3)
    elif name == 'squeezenet':
        net = models.squeezenet1_0()
    elif name == 'densenet161':
        net = models.densenet161()
    elif name == 'inception_v3':
        net = models.inception_v3()
    elif name == 'mnasnet1_0':
        net = models.mnasnet1_0()
    elif name == 'shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0()
    elif 'res' in name:
        if name == 'resnext50_32x4d':
            net = models.resnext50_32x4d(pretrained=args.pretrained, progress=True)
            net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, nb_cls)
        elif name == 'resnet18':
            net = models.resnet18(pretrained=args.pretrained, progress=True)
            net.fc = nn.Linear(512, nb_cls)
        elif name == 'wide_resnet50_2':
            net = models.wide_resnet50_2()
        elif name == 'resnext101_32x8d':
            net = models.resnext101_32x8d(pretrained=args.pretrained, progress=True)
            net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, nb_cls)
        else:
            raise Exception('Net name is not correct')
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise Exception("net name is wrong")
    print(net)
    net_parameters = futil.count_parameters(net)
    log_dict['net_parameters'] = net_parameters

    return net


def load_level_data_old(data_dir: str, label_file: str, level: int) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param label_file:
    :param data_dir:
    :param level:
    :return: 
    """
    file_prefix = "Level" + str(level)
    # 3 neighboring slices for one level
    x_up = sorted(glob.glob(os.path.join(data_dir, "*", file_prefix + "_up*")))
    x_middle = sorted(glob.glob(os.path.join(data_dir, "*", file_prefix + "_middle*")))
    x_down = sorted(glob.glob(os.path.join(data_dir, "*", file_prefix + "_down*")))
    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    y_disext = pd.DataFrame(label_excel, columns=['L' + str(level) + '_disext']).values
    y_gg = pd.DataFrame(label_excel, columns=['L' + str(level) + '_gg']).values
    y_retp = pd.DataFrame(label_excel, columns=['L' + str(level) + '_retp']).values

    y_disext = np.array(y_disext).reshape((-1,))
    y_gg = np.array(y_gg).reshape((-1,))
    y_retp = np.array(y_retp).reshape((-1,))

    x = sorted([*x_up, *x_middle, *x_down])

    # repeat each element of y 3 times
    y_disext = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_disext))
    y_gg = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_gg))
    y_retp = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_retp))

    # y_disext = list(itertools.chain.from_iterable((itertools.repeat(y, 3) for y in y_disext)))
    # y_gg = list(itertools.chain.from_iterable((itertools.repeat(y, 3) for y in y_gg)))
    # y_retp = list(itertools.chain.from_iterable((itertools.repeat(y, 3) for y in y_retp)))

    y = [np.array([a_, b_, c_]) for a_, b_, c_ in zip(y_disext, y_gg, y_retp)]

    assert os.path.dirname(x[0]) == os.path.dirname(x[1]) == os.path.dirname(x[2])
    assert len(x) == len(y)
    log_dict['patients_per_level'] = len(x)

    return x, y


def load_level_names_from_dir(pat_dir: str, label_file: str, levels: Union[int, List[int]]) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param pat_dir:
    :param label_file:
    :param levels:
    :return:
    """
    label_excel = pd.read_excel(label_file, engine='openpyxl')
    if type(levels) is int:
        levels = [levels]
    x_list, y_list = [], []
    for level in levels:
        file_prefix = "Level" + str(level)
        x_middle = sorted(glob.glob(os.path.join(pat_dir, file_prefix + "_middle*")))

        # 3 labels for one level
        y_disext = pd.DataFrame(label_excel, columns=['L' + str(level) + '_disext']).values
        y_gg = pd.DataFrame(label_excel, columns=['L' + str(level) + '_gg']).values
        y_retp = pd.DataFrame(label_excel, columns=['L' + str(level) + '_retp']).values

        y_disext = np.array(y_disext).reshape((-1,))
        y_gg = np.array(y_gg).reshape((-1,))
        y_retp = np.array(y_retp).reshape((-1,))

        y = [np.array([a_, b_, c_]) for a_, b_, c_ in zip(y_disext, y_gg, y_retp)]

        x_list.extend(x_middle)
        y_list.extend(y)

    assert len(x_list) == len(y_list)

    return x_list, y_list


def load_data_of_pats(dir_pats: Sequence, label_file: str):
    df_excel = pd.read_excel(label_file, engine='openpyxl')
    df_excel = df_excel.set_index('PatID')
    x, y = [], []
    for dir_pat in dir_pats:
        # print(dir_pat)
        x_pat, y_pat = load_data_of_5_levels(dir_pat, df_excel)
        x.extend(x_pat)
        y.extend(y_pat)
    return x, y


def load_data_of_levels(level_names: List, y):
    level_middle = level_names
    level_all_names, level_all_y = [], []
    for level_m, y_ in zip(level_middle, y):
        level_idx = level_m.split('Level')[-1].split("_")[
            0]  # /data/jjia/ssc_scoring/dataset/SSc_DeepLearning/Pat_010/Level1_up.mha

        level_u = sorted(glob.glob(os.path.join(os.path.dirname(level_m), "Level" + level_idx + "_up*")))[0]
        # level_m = sorted(glob.glob(os.path.join(os.path.dirname(level_m), "Level" + level_idx + "_up*")))
        level_d = sorted(glob.glob(os.path.join(os.path.dirname(level_m), "Level" + level_idx + "_down*")))[0]

        level_all_names.extend([level_u, level_m, level_d])
        level_all_y.extend([y_, y_, y_])

    return level_all_names, level_all_y


def load_data_of_5_levels(dir_pat: str, df_excel: pd.DataFrame) -> Tuple[List, List]:
    x, y = [], []
    for level in [1, 2, 3, 4, 5]:
        x_level, y_level = load_data_of_a_level(dir_pat, df_excel, level)
        x.extend(x_level)
        y.extend(y_level)

    return x, y


def load_data_of_a_level_old(dir_pat: str, label_file: str, level: int) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param dir_pat:
    :param label_file:
    :param level:
    :return:
    """
    file_prefix = "Level" + str(level)
    # 3 neighboring slices for one level
    x_up = sorted(glob.glob(os.path.join(dir_pat, file_prefix + "_up*")))
    x_middle = sorted(glob.glob(os.path.join(dir_pat, file_prefix + "_middle*")))
    x_down = sorted(glob.glob(os.path.join(dir_pat, file_prefix + "_down*")))
    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    y_disext = pd.DataFrame(label_excel, columns=['L' + str(level) + '_disext']).values
    y_gg = pd.DataFrame(label_excel, columns=['L' + str(level) + '_gg']).values
    y_retp = pd.DataFrame(label_excel, columns=['L' + str(level) + '_retp']).values

    y_disext = np.array(y_disext).reshape((-1,))
    y_gg = np.array(y_gg).reshape((-1,))
    y_retp = np.array(y_retp).reshape((-1,))

    x = sorted([*x_up, *x_middle, *x_down])

    # repeat each element of y 3 times
    y_disext = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_disext))
    y_gg = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_gg))
    y_retp = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_retp))

    y = [np.array([a_, b_, c_]) for a_, b_, c_ in zip(y_disext, y_gg, y_retp)]

    assert os.path.dirname(x[0]) == os.path.dirname(x[1]) == os.path.dirname(x[2])
    assert len(x) == len(y)
    log_dict['patients_per_level'] = len(x)

    return x, y


def load_data_of_a_level(dir_pat: str, df_excel: pd.DataFrame, level: int) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param df_excel:
    :param dir_pat:
    :param level:
    :return:
    """
    file_prefix = "Level" + str(level)
    # 3 neighboring slices for one level
    if args.masked_by_lung:
        x_up = glob.glob(os.path.join(dir_pat, file_prefix + "_up_MaskedByLung.mha"))[0]
        x_middle = glob.glob(os.path.join(dir_pat, file_prefix + "_middle_MaskedByLung.mha"))[0]
        x_down = glob.glob(os.path.join(dir_pat, file_prefix + "_down_MaskedByLung.mha"))[0]
    else:
        x_up = glob.glob(os.path.join(dir_pat, file_prefix + "_up.mha"))[0]
        x_middle = glob.glob(os.path.join(dir_pat, file_prefix + "_middle.mha"))[0]
        x_down = glob.glob(os.path.join(dir_pat, file_prefix + "_down.mha"))[0]
    x = [x_up, x_middle, x_down]

    excel = df_excel
    idx = int(dir_pat.split('/')[-1].split('Pat_')[-1])

    y_disext = excel.at[idx, 'L' + str(level) + '_disext']
    y_gg = excel.at[idx, 'L' + str(level) + '_gg']
    y_retp = excel.at[idx, 'L' + str(level) + '_retp']

    y_disext = [y_disext, y_disext, y_disext]
    y_gg = [y_gg, y_gg, y_gg]
    y_retp = [y_retp, y_retp, y_retp]

    y = [np.array([a_, b_, c_]) for a_, b_, c_ in zip(y_disext, y_gg, y_retp)]

    assert os.path.dirname(x[0]) == os.path.dirname(x[1]) == os.path.dirname(x[2])
    assert len(x) == len(y)

    return x, y


def normalize(image):
    # normalize the image
    mean, std = np.mean(image), np.std(image)
    image = image - mean
    image = image / std
    return image


class ReconDatasetd(Dataset):
    def __init__(self, data_x_names, transform=None):
        self.data_x_names = data_x_names
        print("loading 3D CT ...")
        self.data_x = [futil.load_itk(x, require_ori_sp=True) for x in tqdm(self.data_x_names)]
        self.data_x_np = [i[0] for i in self.data_x]

        normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
        print("normalizing data")
        self.data_x_np = [normalize0to1(x_np) for x_np in tqdm(self.data_x_np)]

        self.data_x_np = [x.astype(np.float32) for x in self.data_x_np]
        # print("padding data")
        # pad the whole 3D data along x and y axis
        # self.data_x_np = [np.pad(x, pad_width=((0, 0), (128, 128), (128, 128)), mode='constant') for x in
        #                   tqdm(self.data_x_np)]
        self.data_x_tensor = [torch.as_tensor(x) for x in self.data_x_np]
        self._shuffle_slice_idx()
        self.transform = transform

    def _shuffle_slice_idx(self):

        self.data_x_slice_idx = [list(range(len(x))) for x in self.data_x_np]
        for ls in self.data_x_slice_idx:
            random.shuffle(ls)  # shuffle list inplace
        # self.data_x_slice_idx_shuffled = [idx_ls for idx_ls in self.data_x_slice_idx]
        self.data_x_slice_idx_gen = []
        for ls in self.data_x_slice_idx:
            self.data_x_slice_idx_gen.append(iter(ls))
        # self.data_x_slice_idx_gen = [(idx for idx in idx_ls) for idx_ls in self.data_x_slice_idx]



    def __len__(self):
        return len(self.data_x_np)

    def __getitem__(self, idx):
        img = self.data_x_tensor[idx]
        try:
            slice_nb = next(self.data_x_slice_idx_gen[idx])
        except StopIteration:
            self._shuffle_slice_idx()
            slice_nb = next(self.data_x_slice_idx_gen[idx])
        # slice_nb = random.randint(0, len(img) - 1)  # random integer in range [a, b], including both end points.
        slice = img[slice_nb]

        data = {'image_key': slice}
        if self.transform:
            data = self.transform(data)
        return data


# class SScScoreDataset(Dataset):
#     """SSc scoring dataset."""
#
#     def __init__(self, data_x_names, data_y_list, index: List = None, transform=None):
#
#         self.data_x_names, self.data_y_list = np.array(data_x_names), np.array(data_y_list)
#         lenth = len(self.data_x_names)
#         if index is not None:
#             self.data_x_names = self.data_x_names[index]
#             self.data_y_list = self.data_y_list[index]
#         print('loading data ...')
#         self.data_x = [futil.load_itk(x, require_ori_sp=True) for x in tqdm(self.data_x_names)]
#         self.data_x_np = [i[0] for i in self.data_x]
#         normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
#
#         print('normalizing data')
#         self.data_x_np = [normalize0to1(x_np) for x_np in tqdm(self.data_x_np)]
#         # scale data to 0~1, it's convinent for future transform during dataloader
#         self.data_x_or_sp = [[i[1], i[2]] for i in self.data_x]
#
#         # self.data_x_np = [normalize(x) for x in self.data_x_np]
#
#         # log_dict['normalize_data'] = True
#
#         self.data_x_np = [x.astype(np.float32) for x in self.data_x_np]
#         self.data_y_np = [y.astype(np.float32) for y in self.data_y_list]
#         # self.min = [np.min(x) for x in self.data_x_np]
#         # print("padding")
#         # self.data_x_np = [np.pad(x, pad_width=((128, 128), (128, 128)), mode='constant') for x in tqdm(self.data_x_np)]
#         self.data_x_tensor = [torch.as_tensor(x) for x in self.data_x_np]
#         self.data_y_tensor = [torch.as_tensor(y) for y in self.data_y_np]
#
#         # self.min_value = [torch.min(x).item() for x in self.data_x_tensor]  # min values after normalization
#         # self.data_x_tensor = [functional.pad(x, padding=[128, 128], fill=min) for x, min in zip(self.data_x_tensor, self.min_value)]
#
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data_y_tensor)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         image = self.data_x_tensor[idx]
#         label = self.data_y_tensor[idx]
#
#         check_aug_effect = 0
#
#         if check_aug_effect and self.transform:
#             img_fpath = self.data_x_names[idx]
#             image_origin, image_spacing = self.data_x_or_sp[idx]
#
#             image_origin = np.append(image_origin, 1)
#             image_spacing = np.append(image_spacing, 1)
#
#             print(img_fpath)
#
#             def crop_center(img, cropx, cropy):
#                 y, x = img.shape
#                 startx = x // 2 - (cropx // 2)
#                 starty = y // 2 - (cropy // 2)
#                 return img[starty:starty + cropy, startx:startx + cropx]
#
#             img_before_aug = crop_center(image.numpy(), 512, 512)
#             futil.save_itk('aug_before_' + img_fpath.split('/')[-1],
#                            img_before_aug, image_origin, image_spacing, dtype='float')
#
#             image = self.transform(image)
#             futil.save_itk('aug_after_' + img_fpath.split('/')[-1],
#                            image.numpy(), image_origin, image_spacing, dtype='float')
#         if self.transform:
#             image = self.transform(image)
#
#         return image, label


class SynthesisDataset(Dataset):
    """SSc scoring dataset."""

    def __init__(self, data_x_names, data_y_list, index: List = None, transform=None):

        self.data_x_names, self.data_y_list = np.array(data_x_names), np.array(data_y_list)
        if index is not None:
            self.data_x_names = self.data_x_names[index]
            self.data_y_list = self.data_y_list[index]
        print('loading data ...')
        self.data_x = [futil.load_itk(x, require_ori_sp=True) for x in tqdm(self.data_x_names)]
        self.data_x_np = [i[0] for i in self.data_x]
        normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
        print('normalizing data')
        self.data_x_np = [normalize0to1(x_np) for x_np in tqdm(self.data_x_np)]
        # scale data to 0~1, it's convinent for future transform during dataloader
        self.data_x_or_sp = [[i[1], i[2]] for i in self.data_x]
        self.ori = np.array([i[1] for i in self.data_x])  # shape order: z, y, x
        self.sp = np.array([i[2] for i in self.data_x])  # shape order: z, y, x

        # self.data_x_np = [normalize(x) for x in self.data_x_np]

        # log_dict['normalize_data'] = True

        self.data_x_np = [x.astype(np.float32) for x in self.data_x_np]
        self.data_y_np = [y.astype(np.float32) for y in self.data_y_list]
        # self.min = [np.min(x) for x in self.data_x_np]
        # self.data_x_np = [np.pad(x, pad_width=((128, 128), (128, 128)), mode='constant') for x in self.data_x_np]
        self.data_x_tensor = [torch.as_tensor(x) for x in self.data_x_np]
        self.data_y_tensor = [torch.as_tensor(y) for y in self.data_y_np]

        # self.min_value = [torch.min(x).item() for x in self.data_x_tensor]  # min values after normalization
        # self.data_x_tensor = [functional.pad(x, padding=[128, 128], fill=min) for x, min in zip(self.data_x_tensor, self.min_value)]

        self.transform = transform

    def __len__(self):
        return len(self.data_y_np)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = {'image_key': self.data_x_tensor[idx],
                'label_key': self.data_y_tensor[idx],
                'space_key': self.sp[idx],
                'origin_key': self.ori[idx],
                'fpath_key': self.data_x_names[idx]}

        if self.transform:
            data = self.transform(data)
        return data


class Synthesisd:
    def __init__(self):
        self.elipse_upper_nb = 3
        self.gg_sample_fpath = None
        self.retp_sample_fpath = None
        self.gg_sample = futil.load_itk(self.gg_sample_fpath)
        self.retp_sample = futil.load_itk(self.retp_sample_fpath)

    def __call__(self, data):
        d = dict(data)

        self.img = d['image_key']
        self.y = d['label_key']
        gg_synthesis = self.synthesis_data(self.img, fill=self.gg_sample)
        retp_synthesis = self.synthesis_data(self.img, fill=self.retp_sample)

    def synthesis_data(self, img, fill=None):
        self.elipse_nb = random.randint(1, self.elipse_upper_nb)  # 1,2,or 3
        for i in range(self.elipse_nb):
            coordi_x = random.randint(0, self.img.shape[0])
            coordi_y = random.randint(0, self.img.shape[0])


class MyNormalize:
    def __call__(self, img: Union[np.ndarray, torch.Tensor]):
        if type(img) == np.ndarray:
            mean, std = np.mean(img), np.std(img)
        else:
            mean, std = torch.mean(img), torch.std(img)
        img = img - mean
        img = img / std
        return img


class MyNormalized(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.norm = MyNormalize()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.norm(d[key])
        return d



class AddChannel:
    def __call__(self, image):
        """
        Apply the transform to `img`.
        """
        if type(image) == np.ndarray:
            image = image[None]
        elif type(image) == torch.Tensor:
            image = image.unsqueeze(0)

        return image


class AddChanneld(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.add_chn = AddChannel()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.add_chn(d[key])
        return d


class RandGaussianNoised(MapTransform):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(keys)
        self.gaussian = RandGaussianNoise(*args, **kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.gaussian(d[key])
        return d


class Clip:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """

        img[img < self.min] = self.min
        img[img > self.max] = self.max
        return img


class Clipd(MapTransform):
    def __init__(self, keys, min, max):
        super().__init__(keys)
        self.clip = Clip(min, max)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.clip(d[key])
        return d


class Path:
    def __init__(self, id, check_id_dir=False) -> None:
        self.id = id  # type: int
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)))  # +'_fold_' + str(args.fold)
        if check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for dir in [self.slurmlog_dir, self.model_dir, self.data_dir, self.id_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print('successfully create directory:', dir)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')
        self.model_wt_structure_fpath = os.path.join(self.id_dir, 'model_wt_structure.pt')

    def label(self, mode: str):
        return os.path.join(self.id_dir, mode + '_label.csv')

    def pred(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred.csv')

    def pred_int(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int.csv')

    def pred_end5(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int_end5.csv')

    def loss(self, mode: str):
        return os.path.join(self.id_dir, mode + '_loss.csv')

    def data(self, mode: str):
        return os.path.join(self.id_dir, mode + '_data.csv')


class RandomAffined(MapTransform):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(keys)
        self.random_affine = RandomAffine(*args, **kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.random_affine(d[key])
        return d


class CenterCropd(MapTransform):
    def __init__(self, keys, *args, **kargs):
        super().__init__(keys)
        self.center_crop = CenterCrop(*args, **kargs)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.center_crop(d[key])
        return d


class RandomHorizontalFlipd(MapTransform):
    def __init__(self, keys, *args, **kargs):
        super().__init__(keys)
        self.random_hflip = RandomHorizontalFlip(*args, **kargs)


    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.random_hflip(d[key])
        return d


class RandomVerticalFlipd(MapTransform):
    def __init__(self, keys, *args, **kargs):
        super().__init__(keys)
        self.random_vflip = RandomVerticalFlip(*args, **kargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.random_vflip(d[key])
        return d


def ssc_transformd(mode ='train'):
    """
    The input image data is from 0 to 1.
    :param mode:
    :return:
    """
    keys = "image_key"
    rotation = 90
    image_size = 512
    vertflip = 0.5
    horiflip = 0.5
    shift = 10 / 512
    scale = 0.05
    xforms = [AddChanneld(keys)]
    if mode == 'train':
        xforms.extend([
            RandomAffined(keys=keys, degrees=rotation, translate=(shift, shift), scale=(1 - scale, 1 + scale)),
            # CenterCropd(image_size),
            RandomHorizontalFlipd(keys, p=horiflip),
            RandomVerticalFlipd(keys, p=vertflip),
            RandGaussianNoised(keys)
        ])
    # else:
    # xforms.append(CenterCropd(image_size))

    xforms.append(MyNormalized(keys))

    transform = transforms.Compose(xforms)
    global log_dict
    log_dict['RandomVerticalFlip'] = vertflip
    log_dict['RandomHorizontalFlip'] = horiflip
    log_dict['RandomRotation'] = rotation
    log_dict['RandomShift'] = shift
    log_dict['image_size'] = image_size
    log_dict['RandGaussianNoise'] = 0.1
    log_dict['RandScale'] = scale

    return transform


def recon_transformd(mode='train'):
    keys = "image_key"  # only transform image
    xforms = [AddChanneld(keys), RandomHorizontalFlipd(keys), RandomVerticalFlipd(keys)]
    xforms.append(MyNormalized(keys))
    xforms = transforms.Compose(xforms)
    return xforms


def _bytes_to_megabytes(value_bytes):
    return round((value_bytes / 1024) / 1024, 2)


def record_GPU_info():
    if args.outfile:
        jobid_gpuid = args.outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        log_dict['gpuname'] = gpuname
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_usage = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total)) + ' MB'
        log_dict['gpu_mem_usage'] = gpu_mem_usage
        gpu_util = 0
        for i in range(5):
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_util += res.gpu
            time.sleep(1)
        gpu_util = gpu_util / 5
        log_dict['gpu_util'] = str(gpu_util) + '%'
    return None


def round_to_5(pred: Union[torch.Tensor, np.ndarray], device=torch.device("cpu")) -> Union[torch.Tensor, np.ndarray]:
    if type(pred) == torch.Tensor:
        tensor_flag = True
        pred = pred.cpu().detach().numpy()
    else:
        tensor_flag = False

    # elif type(pred) == np.ndarray:
    pred = np.rint(pred / 5) * 5
    pred[pred > 100] = 100
    pred[pred < 0] = 0

    if tensor_flag:
        pred = torch.tensor(pred)
        pred = pred.to(device)

    return pred


def split_dir_pats(data_dir, label_file, ts_id):
    abs_dir_path = os.path.dirname(os.path.realpath(__file__))  # abosolute path of the current .py file
    data_dir = abs_dir_path + "/" + data_dir
    dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*")))

    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    pats_id_in_excel = pd.DataFrame(label_excel, columns=['PatID']).values
    pats_id_in_excel = [i[0] for i in pats_id_in_excel]
    assert len(dir_pats) == len(pats_id_in_excel)

    # assert the names of patients got from 2 ways
    pats_id_in_dir = [int(path.split('/')[-1].split('Pat_')[-1]) for path in dir_pats]
    pats_id_in_excel = [int(pat_id) for pat_id in pats_id_in_excel]
    assert pats_id_in_dir == pats_id_in_excel

    ts_dir, tr_vd_dir = [], []
    for id, dir_pt in zip(pats_id_in_dir, dir_pats):
        if id in ts_id:
            ts_dir.append(dir_pt)
        else:
            tr_vd_dir.append(dir_pt)
    return np.array(tr_vd_dir), np.array(ts_dir)


def get_dir_pats(data_dir: str, label_file: str) -> List:
    """
    get absolute directories of patients in this data_dir, use label_file to verify the existing directories.
    data_dir: relative path
    """
    abs_dir_path = os.path.dirname(os.path.realpath(__file__))  # abosolute path of the current .py file
    data_dir = abs_dir_path + "/" + data_dir
    dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*")))

    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    pats_id_in_excel = pd.DataFrame(label_excel, columns=['PatID']).values
    pats_id_in_excel = [i[0] for i in pats_id_in_excel]
    assert len(dir_pats) == len(pats_id_in_excel)

    # assert the names of patients got from 2 ways
    pats_id_in_dir = [int(path.split('/')[-1].split('Pat_')[-1]) for path in dir_pats]
    pats_id_in_excel = [int(pat_id) for pat_id in pats_id_in_excel]
    assert pats_id_in_dir == pats_id_in_excel

    return dir_pats


def start_run(mode, net, dataloader, amp, epochs, device, loss_fun, loss_fun_mae, opt, scaler, mypath, epoch_idx,
              valid_mae_best=None):
    print(mode + "ing ......")
    loss_path = mypath.loss(mode)
    if mode == 'train':
        net.train()
    else:
        net.eval()

    batch_idx = 0
    total_loss = 0
    total_loss_mae = 0
    total_loss_mae_end5 = 0
    for data in dataloader:
        if 'label_key' not in data:
            batch_x, batch_y = data['image_key'], data['image_key']
        else:
            batch_x, batch_y = data['image_key'], data['label_key']
            print('batch_y is: ')
            print(batch_y)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        print(f"batch_x.shape: {batch_x.shape}, batch_y.shape: {batch_y.shape} ")
        if args.r_c == "c":
            batch_y = batch_y.type(torch.LongTensor)  # crossentropy requires LongTensor
            batch_y = batch_y.to(device)
        if amp:
            with torch.cuda.amp.autocast():
                if mode != 'train':
                    with torch.no_grad():
                        pred = net(batch_x)
                else:
                    pred = net(batch_x)

                loss = loss_fun(pred, batch_y)

                if args.r_c == "c":
                    pred = torch.argmax(pred, dim=1)
                    pred = pred.type(torch.FloatTensor)
                    pred = pred.to(device)
                    pred = pred * 5  # convert back to original scores
                    batch_y = batch_y * 5  # convert back to original scores
                    # pred = pred.type(torch.LongTensor)

                loss_mae = loss_fun_mae(pred, batch_y)
                pred_end5 = round_to_5(pred, device)
                loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)
            if mode == 'train':  # update gradients only when training
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        else:
            if mode != 'train':
                with torch.no_grad():
                    pred = net(batch_x)
            else:
                pred = net(batch_x)

            loss = loss_fun(pred, batch_y)

            if args.r_c == "c":
                pred = torch.argmax(pred, dim=1)
                pred = pred.type(torch.FloatTensor)
                pred = pred.to(device)
                pred = pred * 5  # convert back to original scores
                batch_y = batch_y * 5  # convert back to original scores

            loss_mae = loss_fun_mae(pred, batch_y)
            pred_end5 = round_to_5(pred, device)
            loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)

            if mode == 'train':  # update gradients only when training
                opt.zero_grad()
                loss.backward()
                opt.step()

        print(f"loss: {loss.item()}, pred.shape: {pred.shape}")

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        total_loss_mae_end5 += loss_mae_end5.item()
        batch_idx += 1

        if epoch_idx == 1:
            try:
                p1 = threading.Thread(target=record_GPU_info)
                p1.start()
            except RuntimeError as e:
                print(e)

    ave_loss = total_loss / batch_idx
    ave_loss_mae = total_loss_mae / batch_idx
    ave_loss_mae_end5 = total_loss_mae_end5 / batch_idx
    print("mode:", mode, "loss: ", ave_loss, "loss_mae: ", ave_loss_mae, "loss_mae_end5: ", ave_loss_mae_end5)

    if not os.path.isfile(loss_path):
        with open(loss_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['step', 'loss', 'mae', 'mae_end5'])
    with open(loss_path, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([epoch_idx, ave_loss, ave_loss_mae, ave_loss_mae_end5])

    if valid_mae_best is not None:
        if ave_loss_mae < valid_mae_best:
            print("old valid loss mae is: ", valid_mae_best)
            print("new valid loss mae is: ", ave_loss_mae)

            valid_mae_best = ave_loss_mae

            print('this model is the best one, save it. epoch id: ', epoch_idx)
            torch.save(net.state_dict(), mypath.model_fpath)
            torch.save(net, mypath.model_wt_structure_fpath)
            print('save_successfully at ', mypath.model_fpath)
        return valid_mae_best
    else:
        return None


def get_column(n, tr_y):
    column = [i[n] for i in tr_y]
    column = [j / 5 for j in column]  # convert labels from [0,5,10, ..., 100] to [0, 1, 2, ..., 20]
    return column


def split_ts_data_by_levels(data_dir, label_file):
    level_x, level_y = load_level_names_from_dir(data_dir, label_file,
                                                 levels=[1, 2, 3, 4, 5])  # level names and level labels

    if args.ts_level_nb == 235:
        test_count = {0: 114, 5: 24, 10: 18, 15: 9, 20: 12, 25: 9, 30: 9, 35: 6, 40: 8, 45: 3, 50: 7, 55: 3, 60: 3,
                      65: 2, 70: 2, 75: 1, 80: 1, 85: 1, 90: 2, 95: 0, 100: 1}
        assert args.ts_level_nb == sum(test_count.values())
    else:
        raise Exception('ts_level_nb should be 235')

    tr_vd_x, tr_vd_y, test_x, test_y = [], [], [], []
    for x, y in zip(level_x, level_y):
        if test_count[y[0]] > 0:  # disext score > 0
            test_x.append(x)
            test_y.append(y)
            test_count[y[0]] -= 1
            continue
        else:
            tr_vd_x.append(x)
            tr_vd_y.append(y)
    tr_vd_x, tr_vd_y, test_x, test_y = map(np.array, [tr_vd_x, tr_vd_y, test_x, test_y])

    return tr_vd_x, tr_vd_y, test_x, test_y


def save_xy(xs, ys, mode, mypath):  # todo: check typing
    with open(mypath.data(mode), 'a') as f:
        writer = csv.writer(f)
        for x, y in zip(xs, ys):
            writer.writerow([x, y])


def prepare_data(mypath):
    # get data_x names
    kf5 = KFold(n_splits=args.total_folds, shuffle=True, random_state=49)  # for future reproduction
    log_dict['data_shuffle'] = True
    log_dict['data_shuffle_seed'] = 49

    data_dir = "dataset/SSc_DeepLearning"
    label_file = "dataset/SSc_DeepLearning/GohScores.xlsx"
    log_dict['data_dir'] = data_dir
    log_dict['label_file'] = label_file

    # pat_names: Set = set(pat_names)
    if args.ts_level_nb == 240:
        ts_id = [68, 83, 36, 187, 238, 12, 158, 189, 230, 11, 35, 37, 137, 144, 17, 42, 66, 70, 28, 64, 210, 3, 49, 32,
                 236, 206, 194, 196, 7, 9, 16, 19, 20, 21, 40, 46, 47, 57, 58, 59, 60, 62, 116, 117, 118, 128, 134, 216]
        tr_vd_pt, ts_pt = split_dir_pats(data_dir, label_file, ts_id)

        kf_list = list(kf5.split(tr_vd_pt))
        tr_pt_idx, vd_pt_idx = kf_list[args.fold - 1]
        tr_pt = tr_vd_pt[tr_pt_idx]
        vd_pt = tr_vd_pt[vd_pt_idx]
        log_dict['tr_pat_nb'] = len(tr_pt)
        log_dict['vd_pat_nb'] = len(vd_pt)
        log_dict['ts_pat_nb'] = len(ts_pt)

        tr_x, tr_y = load_data_of_pats(tr_pt, label_file)
        vd_x, vd_y = load_data_of_pats(vd_pt, label_file)
        ts_x, ts_y = load_data_of_pats(ts_pt, label_file)
    else:
        raise Exception('ts_level_nb is not correct')

    # else:
    #     tr_vd_level_names, tr_vd_level_y, test_level_names, test_level_y = split_ts_data_by_levels(data_dir, label_file)
    #
    #     kf_list = list(kf5.split(tr_vd_level_names))
    #     tr_level_idx, vd_level_idx = kf_list[args.fold - 1]
    #
    #     log_dict['train_level_nb'] = len(tr_level_idx)
    #     log_dict['valid_level_nb'] = len(vd_level_idx)
    #     log_dict['test_level_nb'] = len(test_level_names)
    #
    #     log_dict['train_index_head'] = tr_level_idx[:20]
    #     log_dict['valid_index_head'] = vd_level_idx[:20]
    #
    #     tr_level_names = tr_vd_level_names[tr_level_idx]
    #     tr_level_y = tr_vd_level_y[tr_level_idx]
    #     vd_level_names = tr_vd_level_names[vd_level_idx]
    #     vd_level_y = tr_vd_level_y[vd_level_idx]
    #
    #     tr_x, tr_y = load_data_of_levels(tr_level_names, tr_level_y)
    #     vd_x, vd_y = load_data_of_levels(vd_level_names, vd_level_y)
    #     ts_x, ts_y = load_data_of_levels(test_level_names, test_level_y)

    for x, y, mode in zip([tr_x, vd_x, ts_x], [tr_y, vd_y, ts_y], ['train', 'valid', 'test']):
        save_xy(x, y, mode, mypath)
    return tr_x, tr_y, vd_x, vd_y, ts_x, ts_y


class MSEHigher(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):

        if torch.sum(y_pred) > torch.sum(y_true):
            loss = self.mse(y_pred, y_true)
            print('mormal loss')
        else:
            loss = self.mse(y_pred, y_true) * 5
            print("higher loss")

        return loss


class MsePlusMae(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, y_pred, y_true):
        mse = self.mse(y_pred, y_true)
        mae = self.mae(y_pred, y_true)
        print(f"mse loss: {mse}, mae loss: {mae}")
        return mse + mae


def get_mae_best(fpath):
    loss = pd.read_csv(fpath)
    mae = min(loss['mae'].to_list())
    return mae


def sampler_by_disext(tr_y):
    disext_list = []
    for sample in tr_y:
        if type(sample) in [list, np.ndarray]:
            disext_list.append(sample[0])
        else:
            disext_list.append(sample)
    disext_np = np.array(disext_list)
    disext_unique = np.unique(disext_np)
    class_sample_count = np.array([len(np.where(disext_np == t)[0]) for t in disext_unique])
    weight = 1. / class_sample_count
    disext_unique_list = list(disext_unique)
    samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])

    # weight = [nb_nonzero/len(data_y_list) if e[0] == 0 else nb_zero/len(data_y_list) for e in data_y_list]
    samples_weight = samples_weight.astype(np.float32)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_loss(args):
    if args.r_c == "c":
        loss_fun = nn.CrossEntropyLoss()  # for classification task
        log_dict['loss_fun'] = 'CE'
    else:
        if args.loss == 'mae':
            loss_fun = nn.L1Loss()
        elif args.loss == 'smooth_mae':
            loss_fun = nn.SmoothL1Loss()
        elif args.loss == 'mse':
            loss_fun = nn.MSELoss()
        elif args.loss == 'mse+mae':
            loss_fun = nn.MSELoss() + nn.L1Loss()  # for regression task
        elif args.loss == 'msehigher':
            loss_fun = MSEHigher()
        else:
            raise Exception("loss function is not correct ", args.loss)
    return loss_fun


def train(id_: int):
    mypath = Path(id_)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = True
    else:
        device = torch.device("cpu")
        amp = False
    log_dict['amp'] = amp

    net = get_net(args.net, 3) if args.r_c == "r" else get_net(args.net, 21)
    if args.train_recon:  # use ReconNet and corresponding dataset
        net = ReconNet(net)

        data_dir: str = "dataset/SSc_DeepLearning"
        label_file: str = "dataset/SSc_DeepLearning/GohScores.xlsx"
        from run_pos import prepare_data as prepare_data_3D
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data_3D(mypath, data_dir, label_file,
                                                             kfold_seed=49, ts_level_nb=args.ts_level_nb,
                                                             fold=args.fold, total_folds = args.total_folds)
        tr_dataset = ReconDatasetd(tr_x[:30], transform=recon_transformd())  # do not need tr_y
        vd_dataset = ReconDatasetd(vd_x[:10], transform=recon_transformd())
        ts_dataset = ReconDatasetd(ts_x[:10], transform=recon_transformd())
        sampler = None
    else:
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data(mypath)
        tr_dataset = SynthesisDataset(tr_x, tr_y, transform=ssc_transformd())
        vd_dataset = SynthesisDataset(vd_x, vd_y, transform=ssc_transformd())
        ts_dataset = SynthesisDataset(ts_x, ts_y, transform=ssc_transformd())
        sampler = sampler_by_disext(tr_y) if args.sampler else None
        print(f'sampler is {sampler}')

        # else:
        #     raise Exception("synthesis_data can not be set with sampler !")

    batch_size = 10
    log_dict['batch_size'] = batch_size
    workers = 10
    log_dict['loader_workers'] = workers
    train_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                  sampler=sampler)
    valid_dataloader = DataLoader(vd_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    # valid_dataloader = train_dataloader
    test_dataloader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


    if args.eval_id:
        mypath2 = Path(args.eval_id)
        if args.mode == "transfer_learning":
            net_recon = ReconNet(net)
            net_recon.load_state_dict(torch.load(mypath2.model_fpath, map_location=torch.device("cpu")))
            net.features = copy.deepcopy(net_recon.features)  # only use the pretrained features
            del net_recon
            valid_mae_best = 10000

        elif args.mode in ["continue_train", "infer"]:
            shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
            for mo in ['train', 'valid', 'test']:
                shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model there

            net.load_state_dict(torch.load(mypath.model_fpath, map_location=torch.device("cpu")))
            valid_mae_best = get_mae_best(mypath2.loss('valid'))
            print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
        else:
            raise Exception("wrong mode: " + args.mode)
    else:
        valid_mae_best = 10000

    net = net.to(device)
    print('move net t device')

    loss_fun = get_loss(args)
    loss_fun_mae = nn.L1Loss()
    lr = 1e-4
    log_dict['lr'] = lr
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if amp else None
    epochs = args.epochs
    for i in range(epochs):  # 20000 epochs
        start_run('train', net, train_dataloader, amp, epochs, device, loss_fun, loss_fun_mae, opt, scaler, mypath, i)
        # run the validation
        valid_mae_best = start_run('valid', net, valid_dataloader, amp, epochs, device, loss_fun, loss_fun_mae, opt,
                                   scaler, mypath, i, valid_mae_best)
        start_run('test', net, test_dataloader, amp, epochs, device, loss_fun, loss_fun_mae, opt, scaler, mypath, i)

    record_best_preds(net, train_dataloader, valid_dataloader, test_dataloader, mypath, device, amp)
    for mode in ['train', 'valid', 'test']:
        if args.mode == "infer" and args.eval_id:
            mypath2 = Path(args.eval_id)
            for mo in ['train', 'valid', 'test']:
                shutil.copy(mypath2.data(mo), mypath.data(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.label(mo), mypath.label(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.pred(mo), mypath.pred(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.pred_int(mo), mypath.pred_int(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.pred_end5(mo), mypath.pred_end5(mo))  # make sure there is at least one model there

        if args.train_recon == 0:
            out_dt = confusion.confusion(mypath.label(mode), mypath.pred_end5(mode))
            log_dict.update(out_dt)
            icc_ = futil.icc(mypath.label(mode), mypath.pred_end5(mode))
            log_dict.update(icc_)
            log_dict.update(icc_)


def record_best_preds(net, train_dataloader, valid_dataloader, test_dataloader, mypath, device, amp):
    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # load the best weights to do evaluation
    dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}

    for mode, dataloader in dataloader_dict.items():
        print("Start write pred to disk for ", mode)
        for data in dataloader:
            if 'label_key' not in data:
                batch_x, batch_y = data['image_key'], data['image_key']
            else:
                batch_x, batch_y = data['image_key'], data['label_key']

            print('batch_y is: ')
            print(batch_y)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if args.r_c == "c":
                batch_y = batch_y.type(torch.LongTensor)  # crossentropy requires LongTensor
                batch_y = batch_y.to(device)
            if amp:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        pred = net(batch_x)
            else:
                with torch.no_grad():
                    pred = net(batch_x)
            if args.r_c == "c":
                pred = torch.argmax(pred, dim=1)
                pred = pred.type(torch.FloatTensor)
                pred = pred.to(device)
                pred = pred * 5  # convert back to original scores
                batch_y = batch_y * 5  # convert back to original scores

            record_preds(mode, batch_y, pred, mypath)


def record_preds(mode, batch_y, pred, mypath):
    batch_label = batch_y.cpu().detach().numpy().astype('Int64')
    batch_preds = pred.cpu().detach().numpy()
    batch_preds_int = batch_preds.astype('Int64')
    batch_preds_end5 = round_to_5(batch_preds_int)
    batch_preds_end5 = batch_preds_end5.astype('Int64')

    head = ['disext', 'gg', 'retp']
    futil.appendrows_to(mypath.label(mode), batch_label, head=head)
    futil.appendrows_to(mypath.pred(mode), batch_preds, head=head)
    futil.appendrows_to(mypath.pred_int(mode), batch_preds_int, head=head)
    futil.appendrows_to(mypath.pred_end5(mode), batch_preds_end5, head=head)


def fill_running(df: pd.DataFrame):
    for index, row in df.iterrows():
        if 'State' not in list(row.index) or row['State'] in [None, np.nan, 'RUNNING']:
            try:
                jobid = row['outfile'].split('-')[-1].split('_')[0]  # extract job id from outfile name
                seff = os.popen('seff ' + jobid)  # get job information
                for line in seff.readlines():
                    line = line.split(
                        ': ')  # must have space to be differentiated from time format 00:12:34
                    if len(line) == 2:
                        key, value = line
                        key = '_'.join(key.split(' '))  # change 'CPU utilized' to 'CPU_utilized'
                        value = value.split('\n')[0]
                        df.at[index, key] = value
            except:
                pass
    return df


def correct_type(df: pd.DataFrame):
    for column in df:
        ori_type = type(df[column].to_list()[-1])
        if ori_type is int:
            df[column] = df[column].astype('Int64')  # correct type
    return df


def record_experiment(record_file: str, current_id: Optional[int] = None):
    if current_id is None:  # before the experiment
        lock = FileLock(record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            with open(record_file, 'a') as csv_file:
                if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
                    new_id = 1
                    df = pd.DataFrame()
                else:
                    df = pd.read_csv(record_file)
                    last_id = df['ID'].to_list()[-1]
                    new_id = int(last_id) + 1
                mypath = Path(new_id, check_id_dir=True)  # to check if id_dir already exist

                date = datetime.date.today().strftime("%Y-%m-%d")
                time = datetime.datetime.now().time().strftime("%H:%M:%S")
                # row = [new_id, date, time, ]
                idatime = {'ID': new_id, 'start_date': date, 'start_time': time}

                args_dict = vars(args)
                idatime.update(args_dict)
                if len(df) == 0:
                    df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
                else:
                    for key, value in idatime.items():
                        try:
                            df.at[new_id - 1, key] = value  #
                        except ValueError:  # some times, the old values are NAN, so the whole columns is float64,
                            # it will raise error if we put a string to float64 cell
                            df[key] = df[key].astype(object)
                            df.at[new_id - 1, key] = value

                df = fill_running(df)
                df = correct_type(df)

                df.to_csv(record_file, index=False)
                shutil.copy(record_file, 'cp_' + record_file)

                df_lastrow = df.iloc[[-1]]
                df_lastrow.to_csv(mypath.id_dir + '/' + record_file, index=False)  # save the record of the current ex
        return new_id
    else:  # at the end of this experiments, find the line of this id, and record the final information
        lock = FileLock(record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            df = pd.read_csv(record_file)
            index = df.index[df['ID'] == current_id].to_list()
            if len(index) > 1:
                raise Exception("over 1 row has the same id", id)
            elif len(index) == 0:  # only one line,
                index = 0
            else:
                index = index[0]

            date = datetime.date.today().strftime("%Y-%m-%d")
            time = datetime.datetime.now().time().strftime("%H:%M:%S")
            df.at[index, 'end_date'] = date
            df.at[index, 'end_time'] = time

            # usage
            f = "%Y-%m-%d %H:%M:%S"
            t1 = datetime.datetime.strptime(df['start_date'][index] + ' ' + df['start_time'][index], f)
            t2 = datetime.datetime.strptime(df['end_date'][index] + ' ' + df['end_time'][index], f)
            elapsed_time = check_time_difference(t1, t2)
            df.at[index, 'elapsed_time'] = elapsed_time

            mypath = Path(current_id)  # evaluate old model
            for mode in ['train', 'valid', 'test']:
                lock2 = FileLock(mypath.loss(mode) + ".lock")
                # when evaluating old mode3ls, those files would be copied to new the folder
                with lock2:
                    try:
                        loss_df = pd.read_csv(mypath.loss(mode))
                    except:
                        mypath2 = Path(args.eval_id)
                        for mo in ['train', 'valid', 'test']:
                            shutil.copy(mypath2.loss(mo), mypath.loss(mo))
                        loss_df = pd.read_csv(mypath.loss(mode))

                    if args.train_recon:
                        best_index = loss_df['mae'].idxmin()
                        log_dict['metrics_min'] = 'mae'
                    else:
                        best_index = loss_df['mae_end5'].idxmin()
                        log_dict['metrics_min'] = 'mae_end5'
                    loss = loss_df['loss'][best_index]
                    mae = loss_df['mae'][best_index]
                    mae_end5 = loss_df['mae_end5'][best_index]
                df.at[index, mode + '_loss'] = round(loss, 2)
                df.at[index, mode + '_mae'] = round(mae, 2)
                df.at[index, mode + '_mae_end5'] = round(mae_end5, 2)

            for key, value in log_dict.items():  # write all log_dict to csv file
                if type(value) is np.ndarray:
                    str_v = ''
                    for v in value:
                        str_v += str(v)
                        str_v += '_'
                    value = str_v
                df.loc[index, key] = value
                if type(value) is int:
                    df[key] = df[key].astype('Int64')

            for column in df:
                if type(df[column].to_list()[-1]) is int:
                    df[column] = df[column].astype('Int64')  # correct type

            args_dict = vars(args)
            args_dict.update({'ID': current_id})
            for column in df:
                if column in args_dict.keys() and type(args_dict[column]) is int:
                    # print(f'convert {df[column]} to float and then to Int64')
                    df[column] = df[column].astype(float).astype('Int64')  # correct str to float and then int

            df.to_csv(record_file, index=False)
            shutil.copy(record_file, 'cp_' + record_file)
            df_lastrow = df.iloc[[-1]]
            df_lastrow.to_csv(mypath.id_dir + '/' + record_file, index=False)  # save the record of the current ex


def check_time_difference(t1: datetime, t2: datetime):
    t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2_date - t1_date

    return str(t_elapsed).split('.')[0]  # drop out microseconds


if __name__ == "__main__":
    record_file = 'records_700.csv'
    id = record_experiment(record_file)
    train(id)
    record_experiment(record_file, current_id=id)
