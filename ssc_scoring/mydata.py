# -*- coding: utf-8 -*-
# @Time    : 7/5/21 6:02 PM
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

import confusion
import myresnet3d
from set_args_pos import args
from networks import med3d_resnet as med3d
from networks import get_net_pos

from mytrans import LoadDatad, MyNormalizeImagePosd, AddChannelPosd, RandomCropPosd, \
    RandGaussianNoise, CenterCropPosd, CropLevelRegiond, ComposePosd

def get_xformd(mode=None, level_node=0, train_on_level=0, z_size=192, y_size=256, x_size=256):
    xforms = [LoadDatad()]
    if level_node or train_on_level:
        xforms.append(CropLevelRegiond(level_node, train_on_level, height=z_size, rand_start=True))
    else:
        if mode == 'train':
            # xforms.extend([RandomCropPosd(), RandGaussianNoisePosd()])
            xforms.extend([RandomCropPosd(z_size=z_size, y_size=y_size, x_size=x_size)])

        else:
            xforms.extend([CenterCropPosd(z_size=z_size, y_size=y_size, x_size=x_size)])

    xforms.extend([MyNormalizeImagePosd(), AddChannelPosd()])
    transform = ComposePosd(xforms)

    return transform


class AllLoader:
    def __init__(self, mypath, label_file, kfold_seed, fold, total_folds, ts_level_nb, level_node,
                 train_on_level, z_size, y_size, x_size, batch_size, workers):
        self.mypath = mypath
        self.label_file = label_file
        df_excel = pd.read_excel(self.label_file, engine='openpyxl')
        self.df_excel = df_excel.set_index('PatID')

        self.kfold_seed = kfold_seed
        self.fold = fold
        self.total_folds = total_folds
        self.ts_level_nb = ts_level_nb
        if self.ts_level_nb == 240:
            self.ts_id = [68, 83, 36, 187, 238, 12, 158, 189, 230, 11, 35, 37, 137, 144, 17, 42, 66, 70, 28, 64, 210, 3, 49,
                     32, 236, 206, 194, 196, 7, 9, 16, 19, 20, 21, 40, 46, 47, 57, 58, 59, 60, 62, 116, 117, 118, 128,
                     134, 216]
        else:
            raise Exception('please use correct testing dataset')
        self.level_node = level_node
        self.train_on_level = train_on_level
        self.z_size = z_size
        self.y_size = y_size
        self.x_size = x_size

        self.batch_size = batch_size
        self.workers = workers

    def save_xy(self, xs: list, ys: list, mode: str):
        with open(self.mypath.data(mode), 'a') as f:
            writer = csv.writer(f)
            for x, y in zip(xs, ys):
                writer.writerow([x, y])

    def load_data_5labels(self, dir_pat: str) -> Tuple[str, np.ndarray]:
        data_name = dir_pat
        idx = int(dir_pat.split('Pat_')[-1][:3])
        data_label = []
        for level in [1, 2, 3, 4, 5]:
            y = self.df_excel.at[idx, 'L' + str(level) + '_pos']
            data_label.append(y)
        return data_name, np.array(data_label)

    def load_data_of_pats(self, dir_pats: Union[List, np.ndarray]) -> list:
        x, y = [], []
        for dir_pat in dir_pats:
            x_pat, y_pat = self.load_data_5labels(dir_pat)
            x.append(x_pat)
            y.append(y_pat)
        return x, y

    def split_dir_pats(self):
        dir_pats = sorted(glob.glob(os.path.join(self.mypath.data_dir, "Pat_*CTimage.mha")))
        if len(dir_pats) == 0:  # does not find patients in this directory
            dir_pats = sorted(glob.glob(os.path.join(self.mypath.data_dir, "Pat_*CTimage_low.mha")))
            if len(dir_pats) == 0:
                dir_pats = sorted(glob.glob(os.path.join(self.mypath.data_dir, "Pat_*", "CTimage.mha")))

        label_excel = pd.read_excel(self.label_file, engine='openpyxl')

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
            if id in self.ts_id:
                ts_dir.append(dir_pt)
            else:
                tr_vd_dir.append(dir_pt)
        return np.array(tr_vd_dir), np.array(ts_dir)

    def prepare_data(self):
        # get data_x names
        kf = KFold(n_splits=self.total_folds, shuffle=True, random_state=self.kfold_seed)  # for future reproduction

        tr_vd_pt, ts_pt = self.split_dir_pats()

        kf_list = list(kf.split(tr_vd_pt))
        tr_pt_idx, vd_pt_idx = kf_list[self.fold - 1]
        tr_pt = tr_vd_pt[tr_pt_idx]
        vd_pt = tr_vd_pt[vd_pt_idx]

        tr_x, tr_y = self.load_data_of_pats(tr_pt)
        vd_x, vd_y = self.load_data_of_pats(vd_pt)
        ts_x, ts_y = self.load_data_of_pats(ts_pt)

        for x, y, mode in zip([tr_x, vd_x, ts_x], [tr_y, vd_y, ts_y], ['train', 'valid', 'test']):
            self.save_xy(x, y, mode)
        return tr_x, tr_y, vd_x, vd_y, ts_x, ts_y

    def xformd(self, mode):
        return get_xformd(mode, level_node=self.level_node,
                   train_on_level=self.train_on_level,
                   z_size=self.z_size, y_size = self.y_size, x_size=self.x_size)
        
    def load(self):
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = self.prepare_data()
        # tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = tr_x[:10], tr_y[:10], vd_x[:10], vd_y[:10], ts_x[:10], ts_y[:10]

        tr_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(tr_x, tr_y)]
        vd_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(vd_x, vd_y)]
        ts_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(ts_x, ts_y)]

        tr_dataset = monai.data.SmartCacheDataset(data=tr_data, transform=self.xformd('train'), replace_rate=0.2,
                                                  cache_num=50, num_init_workers=4, num_replace_workers=8)
        vdaug_dataset = monai.data.CacheDataset(data=vd_data, transform=self.xformd('train'), num_workers=4, cache_rate=1)
        vd_dataset = monai.data.CacheDataset(data=vd_data, transform=self.xformd('valid'), num_workers=4, cache_rate=1)
        ts_dataset = monai.data.PersistentDataset(data=ts_data, transform=self.xformd('test'), cache_dir="persistent_cache")

        train_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
                                      pin_memory=True, persistent_workers=True)
        validaug_dataloader = DataLoader(vdaug_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                         pin_memory=True, persistent_workers=True)
        valid_dataloader = DataLoader(vd_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                      pin_memory=True, persistent_workers=True)
        test_dataloader = DataLoader(ts_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                     pin_memory=True, persistent_workers=True)

        return train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader



