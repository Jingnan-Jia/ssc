# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import csv
import datetime
import glob
import os
import random
import shutil
import threading
import time
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple, Hashable, Mapping

import monai
import myutil.myutil as futil
import numpy as np
import nvidia_smi
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from filelock import FileLock
from monai.transforms import ScaleIntensityRange, RandGaussianNoise
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from ssc_scoring import confusion
from ssc_scoring import myresnet3d
from ssc_scoring.set_args_pos import args

TransInOut = Mapping[Hashable, Optional[Union[np.ndarray, str]]]


class Cnn3fc1(nn.Module):
    def __init__(self, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 4 * 6 * 6 * 6, args.fc_m1),
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


class Cnn3fc2(nn.Module):
    def __init__(self, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 4 * 6 * 6 * 6, args.fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc1_nodes, args.fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc2_nodes, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn4fc2(nn.Module):
    def __init__(self, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 8 * 6 * 6 * 6, args.fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc1_nodes, args.fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc2_nodes, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn5fc2(nn.Module):
    def __init__(self, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 16 * 6 * 6 * 6, args.fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc1_nodes, args.fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc2_nodes, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn6fc2(nn.Module):
    def __init__(self, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 16, base * 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 32),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 32 * 6 * 6 * 6, args.fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc1_nodes, args.fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc2_nodes, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Vgg11_3d(nn.Module):
    def __init__(self, num_classes: int = 5, base: int = 8, in_level: int = 1):
        super().__init__()
        if args.InsNorm:
            self.features = nn.Sequential(
                nn.Conv3d(1, base, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.InstanceNorm3d(base),
                nn.MaxPool3d(kernel_size=3, stride=2),

                nn.Conv3d(base, base * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.InstanceNorm3d(base * 2),
                nn.MaxPool3d(kernel_size=3, stride=2),

                nn.Conv3d(base * 2, base * 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.InstanceNorm3d(base * 4),
                nn.Conv3d(base * 4, base * 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.InstanceNorm3d(base * 4),
                nn.MaxPool3d(kernel_size=3, stride=2),

                nn.Conv3d(base * 4, base * 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.InstanceNorm3d(base * 8),
                nn.Conv3d(base * 8, base * 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.InstanceNorm3d(base * 8),
                nn.MaxPool3d(kernel_size=3, stride=2),

                nn.Conv3d(base * 8, base * 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.InstanceNorm3d(base * 16),
                nn.Conv3d(base * 16, base * 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.InstanceNorm3d(base * 16),
                nn.MaxPool3d(kernel_size=3, stride=2),

            )
        else:
            self.features = nn.Sequential(
                nn.Conv3d(1, base, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(base),
                nn.MaxPool3d(kernel_size=3, stride=2),

                nn.Conv3d(base, base * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(base * 2),
                nn.MaxPool3d(kernel_size=3, stride=2),

                nn.Conv3d(base * 2, base * 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(base * 4),
                nn.Conv3d(base * 4, base * 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(base * 4),
                nn.MaxPool3d(kernel_size=3, stride=2),

                nn.Conv3d(base * 4, base * 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(base * 8),
                nn.Conv3d(base * 8, base * 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(base * 8),
                nn.MaxPool3d(kernel_size=3, stride=2),

                nn.Conv3d(base * 8, base * 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(base * 16),
                nn.Conv3d(base * 16, base * 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(base * 16),
                nn.MaxPool3d(kernel_size=3, stride=2),

            )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 16 * 6 * 6 * 6, args.fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc1_nodes, args.fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc2_nodes, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_net_pos(name: str, nb_cls: int):
    if name == 'cnn3fc1':
        net = Cnn3fc1(num_classes=nb_cls)
    elif name == 'cnn3fc2':
        net = Cnn3fc2(num_classes=nb_cls)
    elif name == 'cnn4fc2':
        net = Cnn4fc2(num_classes=nb_cls)
    elif name == 'cnn5fc2':
        net = Cnn5fc2(num_classes=nb_cls)
    elif name == 'cnn6fc2':
        net = Cnn6fc2(num_classes=nb_cls)
    elif name == "vgg11_3d":
        net = Vgg11_3d(num_classes=nb_cls)
    elif name == "r3d_resnet":
        if args.pretrained:  # inplane=64
            net = models.video.r3d_18(pretrained=True, progress=True)
            net.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                    padding=(1, 3, 3), bias=False)
            net.fc = torch.nn.Linear(in_features=512, out_features=nb_cls)
        else:  # inplane = 8
            net = myresnet3d.r3d_18(pretrained=False, num_classes=nb_cls)
            net.stem[0] = nn.Conv3d(1, 8, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                    padding=(1, 3, 3), bias=False)
    else:
        raise Exception('wrong net name', name)
    net_parameters = futil.count_parameters(net)
    net_parameters = str(net_parameters // 1024 // 1024)  # convert to M
    log_dict['net_parameters'] = net_parameters

    return net


def load_data_of_pats(dir_pats: Union[List, np.ndarray], label_file: str) -> list:
    df_excel = pd.read_excel(label_file, engine='openpyxl')
    df_excel = df_excel.set_index('PatID')
    x, y = [], []
    for dir_pat in dir_pats:
        x_pat, y_pat = load_data_5labels(dir_pat, df_excel)
        x.append(x_pat)
        y.append(y_pat)
    return x, y


def load_data_5labels(dir_pat: str, df_excel: pd.DataFrame) -> Tuple[str, np.ndarray]:
    data_name = dir_pat
    idx = int(dir_pat.split('Pat_')[-1][:3])
    data_label = []
    for level in [1, 2, 3, 4, 5]:
        y = df_excel.at[idx, 'L' + str(level) + '_pos']
        data_label.append(y)
    return data_name, np.array(data_label)


class DatasetPos(Dataset):
    """SSc scoring dataset."""

    def __init__(self, data: Sequence, xform: Union[Sequence[Callable], Callable] = None):
        self.data = data
        self.transform = xform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            data = self.transform(self.data[idx])
        else:
            data = self.data[idx]

        data['image_key'] = torch.as_tensor(data['image_key'])
        data['label_in_patch_key'] = torch.as_tensor(data['label_in_patch_key'])

        return data


#
#
#
# class DatasetPos(Dataset):
#     """SSc scoring dataset."""
#
#     def __init__(self, x_fpaths: Sequence, world_list: Sequence, index: Sequence = None, xform=None):
#
#         self.data_x_names, self.world_list = np.array(x_fpaths), np.array(world_list)
#
#         if index is not None:
#             self.data_x_names = self.data_x_names[index]
#             self.world_list = self.world_list[index]
#         print('loading data ...')
#         self.data_x = [futil.load_itk(x, require_ori_sp=True) for x in tqdm(self.data_x_names)]
#         self.data_x_np = [i[0] for i in self.data_x]  # shape order: z, y, x
#         normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
#         print("normalizing ... ")
#         self.data_x_np = [normalize0to1(x_np) for x_np in tqdm(self.data_x_np)]
#         # scale data to 0~1, it's convinent for future transform during dataloader
#         self.data_x_or_sp = [[i[1], i[2]] for i in self.data_x]
#         self.ori = np.array([i[1] for i in self.data_x])  # shape order: z, y, x
#         self.sp = np.array([i[2] for i in self.data_x])  # shape order: z, y, x
#         self.y = []
#         for world, ori, sp in zip(self.world_list, self.ori, self.sp):
#             labels = [int((level_pos - ori[0]) / sp[0]) for level_pos in world]  # ori[0] is the ori of z axil
#             self.y.append(np.array(labels))
#
#         if args.fine_level:
#             x_ls = []
#             y_ls = []
#             for x, y in zip(self.data_x_np, self.y):
#                 start = y[0] - args.fine_window
#                 end = y[0] + args.fine_window
#                 x = x[start: end]
#                 x_ls.append(x)
#
#                 y = y[0] - start
#                 y_ls.append(y)
#
#             self.data_x_np = x_ls
#             self.y = y_ls
#
#         self.data_x_np = [x.astype(np.float32) for x in self.data_x_np]
#         self.data_y_np = [y.astype(np.float32) for y in self.y]
#
#         # randomcrop = RandomCropPos()
#         # image_, label_ = [], []
#         # for image, label in zip(self.data_x_np, self.data_y_np):
#         #     i, l = randomcrop(image, label)
#         #     image_.append(i)
#         #
#         #     label_.append(l)
#         #
#         # noise = RandGaussianNoisePos()
#         # for image, label in zip(self.data_x_np, self.data_y_np):
#         #     image, label = noise(image, label)
#         #
#
#         self.transform = xform
#
#     def __len__(self):
#         return len(self.data_y_np)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         data = {'image_key': self.data_x_np[idx],
#                 'label_key': self.data_y_np[idx],
#                 'world_key': self.world_list[idx],
#                 'space_key': self.sp[idx],
#                 'origin_key': self.ori[idx],
#                 'fpath_key': self.data_x_names[idx]}
#
#         check_aug_effect = 0s
#         if check_aug_effect:
#             def crop_center(img, cropx, cropy):
#                 y, x = img.shape
#                 startx = x // 2 - (cropx // 2)
#                 starty = y // 2 - (cropy // 2)
#                 return img[starty:starty + cropy, startx:startx + cropx]
#
#             img_before_aug = crop_center(data['image_key'], 512, 512)
#             futil.save_itk('aug_before_' + data['fpath_key'].split('/')[-1],
#                            img_before_aug, data['origin_key'], data['space_key'], dtype='float')
#         # if self.transform:
#         #     self.data_xy=[self.transform(image, label) for image, label in zip(self.data_x_np, self.data_y_np)]
#         #     self.data_x = [x for x in self.data_xy[0]]
#         #     self.data_y = [y for y in self.data_xy[1]]
#         #     self.data_x_np = np.array(self.data_x)
#         #     self.data_y_np = np.array(self.data_y)
#         if self.transform:
#             data = self.transform(data)
#
#         if check_aug_effect:
#             futil.save_itk('aug_after_' + data['fpath_key'].split('/')[-1],
#                            data['image_key'], data['origin_key'], data['space_key'], dtype='float')
#
#         data['image_key'] = torch.as_tensor(data['image_key'])
#         data['label_key'] = torch.as_tensor(data['label_key'])
#
#         return data
#
#
class LoadDatad:
    def __init__(self):
        self.normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)

    def __call__(self, data: Mapping[str, Union[np.ndarray, str]]) -> Dict[str, np.ndarray]:
        fpath = data['fpath_key']
        world_pos = np.array(data['world_key']).astype(np.float32)
        data_x = futil.load_itk(fpath, require_ori_sp=True)
        x = data_x[0]  # shape order: z, y, x
        print("cliping ... ")
        x[x < -1500] = -1500
        x[x > 1500] = 1500
        # x = self.normalize0to1(x)
        # scale data to 0~1, it's convinent for future transform (add noise) during dataloader
        ori = np.array(data_x[1]).astype(np.float32)  # shape order: z, y, x
        sp = np.array(data_x[2]).astype(np.float32)  # shape order: z, y, x
        y = ((world_pos - ori[0]) / sp[0]).astype(int)

        data_x_np = x.astype(np.float32)
        data_y_np = y.astype(np.float32)

        data = {'image_key': data_x_np,  # original image
                'label_in_patch_key': data_y_np,  # relative label (slice number) in  a patch, np.array with shape(-1, )
                'label_in_img_key': data_y_np,  # label in  the whole image, keep fixed, a np.array with shape(-1, )
                'world_key': world_pos,  # world position in mm, keep fixed,  a np.array with shape(-1, )
                'space_key': sp,  # space,  a np.array with shape(-1, )
                'origin_key': ori,  # origin,  a np.array with shape(-1, )
                'fpath_key': fpath}  # full path, a string
        return data


class AddChannelPosd:
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        """
        Apply the transform to `img`.
        """
        d = dict(data)
        d['image_key'] = d['image_key'][None]
        return d


class MyNormalizeImagePosd:
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)

        mean, std = np.mean(d['image_key']), np.std(d['image_key'])
        d['image_key'] = d['image_key'] - mean
        d['image_key'] = d['image_key'] / std
        return d


class Path:
    def __init__(self, id, model_dir=None, check_id_dir=False) -> None:
        self.id = id  # type: int
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models_pos'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)))  # +'_fold_' + str(args.fold)
        if args.mode == 'train' and check_id_dir:  # when infer, do not check
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

    def pred_world(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_world.csv')

    def world(self, mode: str):
        return os.path.join(self.id_dir, mode + '_world.csv')

    def loss(self, mode: str):
        return os.path.join(self.id_dir, mode + '_loss.csv')

    def data(self, mode: str):
        return os.path.join(self.id_dir, mode + '_data.csv')


class RandGaussianNoisePosd:
    def __init__(self, *args, **kargs):
        self.noise = RandGaussianNoise(*args, **kargs)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        d['image_key'] = self.noise(d['image_key'])
        return d


def shiftd(d, start, z_size, y_size, x_size):
    d['image_key'] = d['image_key'][start[0]:start[0] + z_size, start[1]:start[1] + y_size,
                     start[2]:start[2] + x_size]
    d['label_in_patch_key'] = d['label_in_img_key'] - start[0]  # image is shifted up, and relative position down

    d['label_in_patch_key'][d['label_in_patch_key'] < 0] = 0  # position outside the edge would be set as edge
    d['label_in_patch_key'][d['label_in_patch_key'] > z_size] = z_size  # position outside the edge would be set as edge

    return d


class CenterCropPosd:
    def __init__(self, z_size=args.z_size, y_size=args.y_size, x_size=args.x_size):
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

    def __call__(self, data: TransInOut) -> TransInOut:
        d = dict(data)
        keys = set(d.keys())
        assert {'image_key', 'label_in_img_key', 'label_in_patch_key'}.issubset(keys)
        img_shape = d['image_key'].shape
        # print(f'img_shape: {img_shape}')
        assert img_shape[0] >= self.z_size
        assert img_shape[1] >= self.y_size
        assert img_shape[2] >= self.x_size
        middle_point = [shape // 2 for shape in img_shape]
        start = [middle_point[0] - self.z_size // 2, middle_point[1] - self.y_size // 2,
                 middle_point[2] - self.y_size // 2]
        d = shiftd(d, start, self.z_size, self.y_size, self.x_size)

        return d


class RandomCropPosd:
    def __init__(self, z_size=args.z_size, y_size=args.y_size, x_size=args.x_size):
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        # if 'image_key' in data:
        img_shape = d['image_key'].shape  # shape order: z,y x
        assert img_shape[0] >= self.z_size
        assert img_shape[1] >= self.y_size
        assert img_shape[2] >= self.x_size

        valid_range = (img_shape[0] - self.z_size, img_shape[1] - self.y_size, img_shape[2] - self.x_size)
        start = [random.randint(0, v_range) for v_range in valid_range]
        d = shiftd(d, start, self.z_size, self.y_size, self.x_size)
        return d


class CropLevelRegiond:
    """
    Only keep the label of the current level: label_in_img.shape=(1,), label_in_patch.shape=(1,)
    """

    def __init__(self, level: int, height: int, rand_start: bool, start: Optional[int] = None):
        """

        :param level: int
        :param rand_start: during training (rand_start=True), inference (rand_start=False).
        :param start: If rand_start is True, start would be ignored.
        """
        self.level = level
        self.height = height
        self.rand_start = rand_start
        self.start = start

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        d = dict(data)

        if self.height > d['image_key'].shape[0]:
            raise Exception(
                f"desired height {self.height} is greater than image size along z {d['image_key'].shape[0]}")

        d['label_in_img_key'] = np.array(d['label_in_img_key'][self.level - 1]).reshape(-1, )  # keep the current label for the current level
        label: int = d['label_in_img_key']  # z slice number
        lower: int = max(0, label - self.height)
        if self.rand_start:
            start = random.randint(lower, label)  # between lower and label
        else:
            start = int(self.start)
            if start < lower:
                raise Exception(f"start position {start} is lower than the lower line {lower}")
            if start > label:
                raise Exception(f"start position {start} is higher than the label line {label}")

        end = int(start + self.height)
        if end > d['image_key'].shape[0]:
            end = d['image_key'].shape[0]
            start = end - self.height
        d['image_key'] = d['image_key'][start: end].astype(np.float32)

        d['label_in_patch_key'] = d['label_in_img_key'] - start

        d['world_key'] = np.array(d['world_key'][self.level - 1]).reshape(-1, )
        # d['world_key'] = np.array(d['world_key']).reshape(-1, ).astype(np.float32)

        return d


class ComposePosd:
    """My Commpose to handle with img and label at the same time.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def get_xformd(mode=None, level=None):
    xforms = [LoadDatad()]
    if level:
        xforms.append(CropLevelRegiond(level, height=args.z_size, rand_start=True))
    else:
        if mode == 'train':
            # xforms.extend([RandomCropPosd(), RandGaussianNoisePosd()])
            xforms.extend([RandomCropPosd()])

        else:
            xforms.extend([CenterCropPosd()])

    xforms.extend([MyNormalizeImagePosd(), AddChannelPosd()])
    transform = ComposePosd(xforms)

    return transform


def _bytes_to_megabytes(value_bytes):
    return round((value_bytes / 1024) / 1024, 2)


def record_mem_info():
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    print('int(memusage.strip())')

    return int(memusage.strip())


def record_cpu_info():
    pass


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


def split_dir_pats(data_dir, label_file, ts_id):
    abs_dir_path = os.path.dirname(os.path.realpath(__file__))  # abosolute path of the current .py file
    data_dir = abs_dir_path + "/" + data_dir

    dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*CTimage.mha")))
    if len(dir_pats) == 0:  # does not find patients in this directory
        dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*CTimage_low.mha")))
        if len(dir_pats) == 0:
            dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*", "CTimage.mha")))

    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    pats_id_in_excel = pd.DataFrame(label_excel, columns=['PatID']).values
    pats_id_in_excel = [i[0] for i in pats_id_in_excel]
    print(f"len(dir): {len(dir_pats)}, len(pats_in_excel): {len(pats_id_in_excel)} ")
    print("======================")
    assert len(dir_pats) == len(pats_id_in_excel)

    # assert the names of patients got from 2 ways
    pats_id_in_dir = [int(path.split('Pat_')[-1][:3]) for path in dir_pats]
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


def start_run(mode, net, dataloader, loss_fun, loss_fun_mae, opt, mypath, epoch_idx,
              valid_mae_best=None):
    print(mode + "ing ......")
    loss_path = mypath.loss(mode)
    if mode == 'train' or mode == 'validaug':
        net.train()
    else:
        net.eval()

    batch_idx = 0
    total_loss = 0
    total_loss_mae = 0

    t0 = time.time()
    t_load_data, t_to_device, t_train_per_step = [], [], []
    for data in dataloader:

        t1 = time.time()
        t_load_data.append(t1 - t0)

        batch_x = data['image_key'].to(device)
        batch_y = data['label_in_patch_key'].to(device)
        sp_z = data['space_key'][:, 0].reshape(-1, 1).to(device)

        t2 = time.time()
        t_to_device.append(t2 - t1)

        if amp:
            with torch.cuda.amp.autocast():
                if mode != 'train':
                    with torch.no_grad():
                        pred = net(batch_x)
                else:
                    pred = net(batch_x)
                pred *= sp_z
                batch_y *= sp_z

                loss = loss_fun(pred, batch_y)

                loss_mae = loss_fun_mae(pred, batch_y)
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
            pred *= sp_z
            batch_y *= sp_z

            loss = loss_fun(pred, batch_y)

            loss_mae = loss_fun_mae(pred, batch_y)

            if mode == 'train':  # update gradients only when training
                opt.zero_grad()
                loss.backward()
                opt.step()

        t3 = time.time()
        t_train_per_step.append(t3 - t2)

        print('loss:', loss.item(), 'pred:', (pred[0] / sp_z).clone().detach().cpu().numpy(),
              'label:', (batch_y / sp_z).clone().detach().cpu().numpy())

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        batch_idx += 1

        p1 = threading.Thread(target=record_GPU_info)
        p1.start()

        t0 = t3  # reset the t0

    t_load_data, t_to_device, t_train_per_step = mean(t_load_data), mean(t_to_device), mean(t_train_per_step)
    if "t_load_data" not in log_dict:
        log_dict.update({"t_load_data": t_load_data,
                         "t_to_device": t_to_device,
                         "t_train_per_step": t_train_per_step})
    print({"t_load_data": t_load_data,
           "t_to_device": t_to_device,
           "t_train_per_step": t_train_per_step})

    ave_loss = total_loss / batch_idx
    ave_loss_mae = total_loss_mae / batch_idx
    print("mode:", mode, "loss: ", ave_loss, "loss_mae: ", ave_loss_mae)

    if not os.path.isfile(loss_path):
        with open(loss_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['step', 'loss', 'mae'])
    with open(loss_path, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([epoch_idx, ave_loss, ave_loss_mae])

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


def save_xy(xs: list, ys: list, mode: str, mypath: Path):
    with open(mypath.data(mode), 'a') as f:
        writer = csv.writer(f)
        for x, y in zip(xs, ys):
            writer.writerow([x, y])


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

    # weight = [nb_nonzero/len(world_list) if e[0] == 0 else nb_zero/len(world_list) for e in world_list]
    samples_weight = samples_weight.astype(np.float32)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_loss(args):
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


def prepare_data(mypath, data_dir, label_file, kfold_seed=49, fold=1, total_folds=4):
    # get data_x names
    kf = KFold(n_splits=total_folds, shuffle=True, random_state=kfold_seed)  # for future reproduction

    if args.ts_level_nb == 240:
        ts_id = [68, 83, 36, 187, 238, 12, 158, 189, 230, 11, 35, 37, 137, 144, 17, 42, 66, 70, 28, 64, 210, 3, 49, 32,
                 236, 206, 194, 196, 7, 9, 16, 19, 20, 21, 40, 46, 47, 57, 58, 59, 60, 62, 116, 117, 118, 128, 134, 216]
        tr_vd_pt, ts_pt = split_dir_pats(data_dir, label_file, ts_id)

        kf_list = list(kf.split(tr_vd_pt))
        tr_pt_idx, vd_pt_idx = kf_list[fold - 1]
        tr_pt = tr_vd_pt[tr_pt_idx]
        vd_pt = tr_vd_pt[vd_pt_idx]

        tr_x, tr_y = load_data_of_pats(tr_pt, label_file)
        vd_x, vd_y = load_data_of_pats(vd_pt, label_file)
        ts_x, ts_y = load_data_of_pats(ts_pt, label_file)

    else:
        raise Exception('please use correct testing dataset')

    for x, y, mode in zip([tr_x, vd_x, ts_x], [tr_y, vd_y, ts_y], ['train', 'valid', 'test']):
        save_xy(x, y, mode, mypath)
    return tr_x, tr_y, vd_x, vd_y, ts_x, ts_y


def dataset_dir(resample_z: int) -> str:
    if resample_z == 0:  # use original images
        data_dir: str = "dataset/SSc_DeepLearning"
    elif resample_z == 256:
        data_dir: str = "dataset/LowResolution_fix_size"
    elif resample_z == 512:
        data_dir: str = "dataset/LowRes512_192_192"
    elif resample_z == 800:
        data_dir: str = "dataset/LowRes800_160_160"
    elif resample_z == 1024:
        data_dir: str = "dataset/LowRes1024_256_256"
    else:
        raise Exception("wrong resample_z:" + str(args.resample_z))
    return data_dir


def eval_net_mae(eval_id: int, net: torch.nn.Module, mypath: Path):
    mypath2 = Path(eval_id)
    shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
    for mo in ['train', 'validaug', 'valid', 'test']:
        try:
            shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model
        except FileNotFoundError:
            pass

    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
    valid_mae_best = get_mae_best(mypath2.loss('valid'))
    print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
    return net, valid_mae_best


def all_loader(mypath, data_dir, label_file, kfold_seed=49):
    log_dict['data_dir'] = data_dir
    log_dict['label_file'] = label_file
    log_dict['data_shuffle_seed'] = kfold_seed

    tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data(mypath, data_dir, label_file, kfold_seed=kfold_seed,
                                                      fold=args.fold, total_folds=args.total_folds)
    # tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = tr_x[:6], tr_y[:6], vd_x[:6], vd_y[:6], ts_x[:6], ts_y[:6]
    log_dict['tr_pat_nb'] = len(tr_x)
    log_dict['vd_pat_nb'] = len(vd_x)
    log_dict['ts_pat_nb'] = len(ts_x)
    tr_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(tr_x, tr_y)]
    vd_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(vd_x, vd_y)]
    ts_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(ts_x, ts_y)]

    # tr_dataset = DatasetPos(tr_data, xform=get_xformd('train', level=args.fine_level))
    # vdaug_dataset = DatasetPos(vd_data[:5], xform=get_xformd('train', level=args.fine_level))
    # vd_dataset =DatasetPos(vd_data[:5], xform=get_xformd('valid', level=args.fine_level))
    if args.if_test:
        ts_dataset = monai.data.PersistentDataset(data=ts_data, transform=get_xformd('test', level=args.fine_level),
                                                  cache_dir="persistent_cache")
    else:
        ts_dataset = None
    tr_dataset = monai.data.SmartCacheDataset(data=tr_data, transform=get_xformd('train', level=args.fine_level),
                                              replace_rate=0.2, cache_num=40, num_init_workers=4,
                                              num_replace_workers=8)  # or self.n_train > self.tr_nb_cache
    vd_dataset = monai.data.CacheDataset(data=vd_data, transform=get_xformd('valid', level=args.fine_level),
                                         num_workers=4, cache_rate=1)
    vdaug_dataset = monai.data.CacheDataset(data=vd_data, transform=get_xformd('train', level=args.fine_level),
                                            num_workers=4, cache_rate=1)

    # tr_dataset = DatasetPos(x_fpaths=tr_x, world_list=tr_y, xform=get_xformd('train', level=args.fine_level))
    # have compatible learning curve
    # vdaug_dataset = DatasetPos(x_fpaths=vd_x, world_list=vd_y, xform=get_xformd('train', level=args.fine_level))
    # vd_dataset = DatasetPos(x_fpaths=vd_x, world_list=vd_y, xform=get_xformd('valid', level=args.fine_level))
    # ts_dataset = DatasetPos(x_fpaths=ts_x, world_list=ts_y, xform=get_xformd('test', level=args.fine_level))

    train_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True, persistent_workers=True)
    validaug_dataloader = DataLoader(vdaug_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                     pin_memory=True, persistent_workers=True)
    valid_dataloader = DataLoader(vd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=True, persistent_workers=True)
    if args.if_test:
        test_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                     pin_memory=True, persistent_workers=True)
    else:
        test_dataloader = None

    return train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader


def compute_metrics(mypath: Path):
    for mode in ['train', 'valid', 'test', 'validaug']:
        try:
            if args.eval_id:
                mypath2 = Path(args.eval_id)
                shutil.copy(mypath2.data(mode), mypath.data(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.loss(mode), mypath.loss(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.world(mode), mypath.world(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.pred(mode), mypath.pred(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.pred_int(mode),
                            mypath.pred_int(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.pred_world(mode),
                            mypath.pred_world(mode))  # make sure there is at least one modedel there

            out_dt = confusion.confusion(mypath.world(mode), mypath.pred_world(mode), label_nb=args.z_size, space=1)
            log_dict.update(out_dt)

            icc_ = futil.icc(mypath.world(mode), mypath.pred_world(mode))
            log_dict.update(icc_)
        except FileNotFoundError:
            continue


def train(id: int):
    mypath = Path(id)
    if args.fine_level:
        outs = 1
    else:
        outs = 5
    net: torch.nn.Module = get_net_pos(args.net, outs)
    data_dir = dataset_dir(args.resample_z)
    label_file = "dataset/SSc_DeepLearning/GohScores.xlsx"
    train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader(mypath, data_dir, label_file)

    net = net.to(device)
    if args.eval_id:
        net, valid_mae_best = eval_net_mae(args.eval_id, net, mypath)
    else:
        valid_mae_best = 10000

    loss_fun = get_loss(args)
    loss_fun_mae = nn.L1Loss()
    lr = 1e-4
    log_dict['lr'] = lr
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)
    epochs = 0 if args.mode == 'infer' else args.epochs
    for i in range(epochs):  # 20000 epochs
        if args.mode in ['train', 'continue_train']:
            start_run('train', net, train_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)
        if i % args.valid_period == 0:
            # run the validation
            valid_mae_best = start_run('valid', net, valid_dataloader, loss_fun, loss_fun_mae, opt, mypath, i,
                                       valid_mae_best)
            start_run('validaug', net, validaug_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)
            if args.if_test:
                start_run('test', net, test_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)

    dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'validaug': validaug_dataloader}
    if args.if_test:
        dataloader_dict.update({'test': test_dataloader})
    record_best_preds(net, dataloader_dict, mypath)
    if not args.fine_level:
        compute_metrics(mypath)
    print('Finish all things!')


def SlidingLoader(fpath, world_pos, z_size, stride=1, batch_size=1):
    print(f'start load {fpath} for sliding window inference')
    trans = ComposePosd([LoadDatad(), MyNormalizeImagePosd()])
    data = trans(data={'fpath_key': fpath, 'world_key': world_pos})

    raw_x = data['image_key']
    label = data['label_in_img_key']

    assert raw_x.shape[0] > z_size
    start_lower: int = label - z_size
    start_higher: int = label + z_size
    start_lower = max(0, start_lower)
    start_higher = min(raw_x.shape[0], start_higher)

    # ranges = raw_x.shape[0] - z_size
    print(f'ranges: {start_lower} to {start_higher}')

    batch_patch = []
    batch_new_label = []
    batch_start = []
    i = 0

    start = start_lower
    while start < label:
        if i < batch_size:
            print(f'start: {start}, i: {i}')
            crop = CropLevelRegiond(level=args.fine_level, height=args.z_size, rand_start=False, start=start)
            new_data = crop(data)
            new_patch, new_label = new_data['image_key'], new_data['label_in_patch_key']
            # patch: np.ndarray = raw_x[start:start + z_size]  # z, y, z
            # patch = patch.astype(np.float32)
            # new_label: torch.Tensor = label - start
            new_patch = new_patch[None]  # add a channel
            batch_patch.append(new_patch)
            batch_new_label.append(new_label)
            batch_start.append(start)

            start += stride
            i += 1

        if start >= start_higher or i >= batch_size:
            batch_patch = torch.tensor(np.array(batch_patch))
            batch_new_label = torch.tensor(batch_new_label)
            batch_start = torch.tensor(batch_start)

            yield batch_patch, batch_new_label, batch_start

            batch_patch = []
            batch_new_label = []
            batch_start = []
            i = 0


class Evaluater():
    def __init__(self, net, dataloader, mode, mypath):
        self.net = net
        self.dataloader = dataloader
        self.mode = mode
        self.mypath = mypath

    def run(self):
        for batch_data in self.dataloader:
            for idx in range(len(batch_data['image_key'])):
                sliding_loader = SlidingLoader(batch_data['fpath_key'][idx], batch_data['world_key'][idx],
                                               z_size=args.z_size, stride=args.infer_stride, batch_size=args.batch_size)
                pred_in_img_ls = []
                pred_in_patch_ls = []
                label_in_patch_ls = []
                for patch, new_label, start in sliding_loader:
                    batch_x = patch.to(device)

                    if self.mode == 'train':
                        p1 = threading.Thread(target=record_GPU_info)
                        p1.start()

                    if amp:
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                pred = self.net(batch_x)
                    else:
                        with torch.no_grad():
                            pred = self.net(batch_x)

                    # pred = pred.cpu().detach().numpy()
                    pred_in_patch = pred.cpu().detach().numpy()
                    pred_in_patch_ls.append(pred_in_patch)

                    start_np = start.numpy().reshape((-1, 1))
                    pred_in_img = pred_in_patch + start_np  # re organize it to original coordinate
                    pred_in_img_ls.append(pred_in_img)

                    new_label_ = new_label + start_np
                    label_in_patch_ls.append(new_label_)

                pred_in_img_all = np.concatenate(pred_in_img_ls, axis=0)
                pred_in_patch_all = np.concatenate(pred_in_patch_ls, axis=0)
                label_in_patch_all = np.concatenate(label_in_patch_ls, axis=0)

                batch_label: np.ndarray = batch_data['label_in_img_key'][idx].cpu().detach().numpy().astype('Int64')
                batch_preds_ave: np.ndarray = np.mean(pred_in_img_all, 0)
                batch_preds_int: np.ndarray = batch_preds_ave.astype('Int64')
                batch_preds_world: np.ndarray = batch_preds_ave * batch_data['space_key'][idx][0].item() + \
                                                batch_data['origin_key'][idx][0].item()
                batch_world: np.ndarray = batch_data['world_key'][idx].cpu().detach().numpy()
                head = ['L1', 'L2', 'L3', 'L4', 'L5']
                if args.fine_level:
                    head = [head[args.fine_level - 1]]
                if idx < 5:
                    futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '.csv',
                                        pred_in_img_all, head=head)
                    futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '_in_patch.csv',
                                        pred_in_patch_all, head=head)
                    futil.appendrows_to(
                        self.mypath.label(self.mode).split('.csv')[0] + '_' + str(idx) + '_in_patch.csv',
                        label_in_patch_all, head=head)

                    pred_all_world = pred_in_img_all * batch_data['space_key'][idx][0].item() + \
                                     batch_data['origin_key'][idx][0].item()
                    futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '_world.csv',
                                        pred_all_world, head=head)

                if args.fine_level:
                    batch_label = np.array(batch_label).reshape(-1, )
                    batch_preds_ave = np.array(batch_preds_ave).reshape(-1, )
                    batch_preds_int = np.array(batch_preds_int).reshape(-1, )
                    batch_preds_world = np.array(batch_preds_world).reshape(-1, )
                    batch_world = np.array(batch_world).reshape(-1, )
                futil.appendrows_to(self.mypath.label(self.mode), batch_label, head=head)  # label in image
                futil.appendrows_to(self.mypath.pred(self.mode), batch_preds_ave, head=head)  # pred in image
                futil.appendrows_to(self.mypath.pred_int(self.mode), batch_preds_int, head=head)
                futil.appendrows_to(self.mypath.pred_world(self.mode), batch_preds_world, head=head)  # pred in world
                futil.appendrows_to(self.mypath.world(self.mode), batch_world, head=head)  # 33 label in world


def record_best_preds(net: torch.nn.Module, dataloader_dict: Dict[str, DataLoader], mypath: Path):
    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # load the best weights to do evaluation
    net.eval()
    for mode, dataloader in dataloader_dict.items():
        evaluater = Evaluater(net, dataloader, mode, mypath)
        evaluater.run()
        # except:
        #     continue


def record_preds(mode, batch_y, pred, mypath, data):
    batch_label = batch_y.cpu().detach().numpy().astype('Int64')
    batch_preds = pred.cpu().detach().numpy()
    batch_preds_int = batch_preds.astype('Int64')
    batch_preds_world = batch_preds_int * data['space_key'] + data['origin_key']

    futil.appendrows_to(mypath.label(mode), batch_label)
    futil.appendrows_to(mypath.pred(mode), batch_preds)
    futil.appendrows_to(mypath.pred_int(mode), batch_preds_int)
    futil.appendrows_to(mypath.pred_world(mode), batch_preds_world)


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
        ori_type = type(df[column].to_list()[-1])  # find the type of the last valuable in this column
        if ori_type is int:
            df[column] = df[column].astype('Int64')  # correct type
    return df


def get_df_id(record_file: str):
    if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
        new_id = 1
        df = pd.DataFrame()
    else:
        df = pd.read_csv(record_file)  # read the record file,
        last_id = df['ID'].to_list()[-1]  # find the last ID
        new_id = int(last_id) + 1
    return df, new_id


def record_1st(record_file, current_id):
    lock = FileLock(record_file + ".lock")  # lock the file, avoid other processes write other things
    with lock:  # with this lock,  open a file for exclusive access
        with open(record_file, 'a') as csv_file:
            df, new_id = get_df_id(record_file)
            mypath = Path(new_id, check_id_dir=True)  # to check if id_dir already exist

            start_date = datetime.date.today().strftime("%Y-%m-%d")
            start_time = datetime.datetime.now().time().strftime("%H:%M:%S")
            # start record by id, date,time row = [new_id, date, time, ]
            idatime = {'ID': new_id, 'start_date': start_date, 'start_time': start_time}
            args_dict = vars(args)
            idatime.update(args_dict)  # followed by super parameters
            if len(df) == 0:  # empty file
                df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
            else:
                index = df.index.to_list()[-1]  # last index
                for key, value in idatime.items():  # write new line
                    df.at[index + 1, key] = value  #

            df = fill_running(df)  # fill the state information for other experiments
            df = correct_type(df)  # aviod annoying thing like: ID=1.00
            write_and_backup(df, record_file, mypath)
    return new_id


def add_best_metrics(df: pd.DataFrame, mypath: Path, index: int) -> pd.DataFrame:
    modes = ['train', 'validaug', 'valid']
    if args.if_test:
        modes.append('test')
    for mode in modes:
        lock2 = FileLock(mypath.loss(mode) + ".lock")
        # when evaluating/inference old models, those files would be copied to new the folder
        with lock2:
            try:
                loss_df = pd.read_csv(mypath.loss(mode))
            except FileNotFoundError:  # copy loss files from old directory to here
                mypath2 = Path(args.eval_id)
                shutil.copy(mypath2.loss(mode), mypath.loss(mode))
                try:
                    loss_df = pd.read_csv(mypath.loss(mode))
                except FileNotFoundError:  # still cannot find the loss file in old directory, pass this mode
                    continue

            best_index = loss_df['mae'].idxmin()
            log_dict['metrics_min'] = 'mae'
            loss = loss_df['loss'][best_index]
            mae = loss_df['mae'][best_index]
        df.at[index, mode + '_loss'] = round(loss, 2)
        df.at[index, mode + '_mae'] = round(mae, 2)
    return df


def write_and_backup(df: pd.DataFrame, record_file: str, mypath: Path):
    df.to_csv(record_file, index=False)
    shutil.copy(record_file, 'cp_' + record_file)
    df_lastrow = df.iloc[[-1]]
    df_lastrow.to_csv(mypath.id_dir + '/' + record_file, index=False)  # save the record of the current ex


def record_2nd(record_file, current_id):
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

        start_date = datetime.date.today().strftime("%Y-%m-%d")
        start_time = datetime.datetime.now().time().strftime("%H:%M:%S")
        df.at[index, 'end_date'] = start_date
        df.at[index, 'end_time'] = start_time

        # usage
        f = "%Y-%m-%d %H:%M:%S"
        t1 = datetime.datetime.strptime(df['start_date'][index] + ' ' + df['start_time'][index], f)
        t2 = datetime.datetime.strptime(df['end_date'][index] + ' ' + df['end_time'][index], f)
        elapsed_time = check_time_difference(t1, t2)
        df.at[index, 'elapsed_time'] = elapsed_time

        mypath = Path(current_id)  # evaluate old model
        df = add_best_metrics(df, mypath, index)

        for key, value in log_dict.items():  # convert numpy to str before writing all log_dict to csv file
            if type(value) in [np.ndarray, list]:
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
                df[column] = df[column].astype('Int64')  # correct type again, avoid None/1.00/NAN, etc.

        args_dict = vars(args)
        args_dict.update({'ID': current_id})
        for column in df:
            if column in args_dict.keys() and type(args_dict[column]) is int:
                df[column] = df[column].astype(float).astype('Int64')  # correct str to float and then int
        write_and_backup(df, record_file, mypath)


def record_experiment(record_file: str, current_id: Optional[int] = None):
    if current_id is None:  # before the experiment
        new_id = record_1st(record_file, current_id)
        return new_id
    else:  # at the end of this experiments, find the line of this id, and record the other information
        record_2nd(record_file, current_id)


def check_time_difference(t1: datetime, t2: datetime):
    t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2_date - t1_date

    return str(t_elapsed).split('.')[0]  # drop out microseconds


if __name__ == "__main__":
    # set some global variables here, like log_dict, device, amp
    LogType = Optional[Union[int, float, str]]  # int includes bool
    log_dict: Dict[str, LogType] = {}  # a global dict to store immutable variables saved to log files

    if torch.cuda.is_available():  # set device and amp
        device = torch.device("cuda")
        amp = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        device = torch.device("cpu")
        amp = False
        scaler = None
    log_dict['amp'] = amp

    record_file: str = 'records_pos.csv'
    id: int = record_experiment(record_file)  # write super parameters from set_args.py to record file.
    train(id)
    record_experiment(record_file, current_id=id)  # write other parameters and metrics to record file.
