# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
sys.path.append("..")

import csv
import os
import threading
import time
from statistics import mean
from typing import Dict, Optional, Union
import numpy as np
import myutil.myutil as futil
from mymodules.path import PathPos, PathPosInit

import myutil.myutil as futil
import torch
import torch.nn as nn

from mymodules.inference import record_best_preds
from mymodules.mydata import LoadPos
from mymodules.myloss import get_loss
from mymodules.networks import get_net_pos
from mymodules.networks import med3d_resnet as med3d
from mymodules.path import PathPos, PathPosInit

from mymodules.set_args_pos import args
from mymodules.tool import record_1st, record_2nd, record_GPU_info, eval_net_mae, compute_metrics
import pandas as pd

class CoresPos:
    def __init__(self, corse_fpath, data_fpath):
        self.corse_fpath = corse_fpath
        self.data_fpath = data_fpath

    def __call__(self, data):
        df_corse_pos = pd.read_csv(self.corse_fpath)
        df_data = pd.read_csv(self.data_fpath)
        pat_idx = None
        for idx, row in enumerate(df_corse_pos.iterrows()):
            if data['pat_id'] in row['fpath']:
                pat_idx = idx
                break
        corse_pred = df_corse_pos[pat_idx]
        data['corse_world_key '] = corse_pred
        return data

class SliceFromCorsePos():
    def __call__(self, d: dict):

        img_3d = d['image_key']
        img_2d_ls = []
        img_2d_name_ls = []
        for i, slice in enumerate([j for j in d['label_in_img']]):
            img_2d_ls.append(img_3d[slice])
            img_2d_name_ls.append('Level'+str(i+1)+'_middle.mha')
            img_2d_ls.append(img_3d[slice+1])
            img_2d_name_ls.append('Level' + str(i + 1) + '_up.mha')
            img_2d_ls.append(img_3d[slice - 1])
            img_2d_name_ls.append('Level' + str(i + 1) + '_down.mha')

        img_2d = np.array(img_2d_ls)
        img_2d_name_ls = np.array(img_2d_name_ls)
        d['image_key'] = img_2d
        d['fpath2save'] = img_2d_name_ls
        return d


def train(id: int, log_dict: dict):
    mypath = PathPos(id)

    label_file = "dataset/SSc_DeepLearning/GohScores.xlsx"
    log_dict['label_file'] = label_file
    seed = 49
    log_dict['data_shuffle_seed'] = seed

    all_loader = LoadPos(0, mypath, label_file, seed, args.fold, args.total_folds, args.ts_level_nb, args.level_node,
                           args.train_on_level, args.z_size, args.y_size, args.x_size, args.batch_size, args.workers)
    train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load(noxform=True)

    dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'validaug': validaug_dataloader}
    dataloader_dict.update({'test': test_dataloader})
    for mode, loader in dataloader_dict.items():
        print(f'start save slices for {mode}')
        corse_pos = CoresPos(corse_fpath=mypath.world(mode), data_fpath=mypath.data(mode))
        pos2slice = SliceFromCorsePos()  # d['image_key'] become 2D
        for data in loader:
            data = corse_pos(data)
            data = pos2slice(data)  # one pos to 3 slices, total 15 slices
            for img, fpath in zip(data['image_key'], data['fpath2save']):
                futil.save_itk(img, fpath)

        print('Finish all things!')
    return log_dict


if __name__ == "__main__":
    # set some global variables here, like log_dict, device, amp
    LogType = Optional[Union[int, float, str]]  # int includes bool
    LogDict = Dict[str, LogType]
    log_dict: LogDict = {}  # a global dict to store immutable variables saved to log files

    record_file: str = PathPosInit().record_file
    id: int = record_1st(record_file, args)  # write super parameters from set_args.py to record file.
    log_dict = train(id, log_dict)
    record_2nd(record_file, current_id=id, log_dict=log_dict, args=args)  # write other parameters and metrics to record file.
