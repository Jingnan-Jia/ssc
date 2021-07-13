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
from mymodules.mydata import LoadPos2Score
from mymodules.myloss import get_loss
from mymodules.networks import get_net_pos
from mymodules.networks import med3d_resnet as med3d
from mymodules.path import PathPos, PathPosInit

from mymodules.set_args_pos import args
from mymodules.tool import record_1st, record_2nd, record_GPU_info, eval_net_mae, compute_metrics
import pandas as pd

from monai.transforms import CastToTyped

def train():
    mypath = PathPos(args.eval_id)

    label_file = "dataset/SSc_DeepLearning/GohScores.xlsx"
    seed = 49
    all_loader = LoadPos2Score(0, mypath, label_file, seed, args.fold, args.total_folds, args.ts_level_nb, args.level_node,
                           args.train_on_level, args.z_size, args.y_size, args.x_size, 1, 1)
    train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load()

    dataloader_dict = {'valid': valid_dataloader}
    # , 'valid': valid_dataloader, 'validaug': validaug_dataloader}
    # dataloader_dict.update({'test': test_dataloader})
    for mode, loader in dataloader_dict.items():
        print(f'start save slices for {mode}')
        for batch_data in loader:  # one data, with shape (1, channel, x, y)
            for slice, pth in zip(batch_data['image_key'][0], batch_data['fpath2save'][0]):  # img and path of each slice
                full_pth = os.path.join(mypath.id_dir, 'predicted_slices', pth)
                if not os.path.isdir(os.path.dirname(full_pth)):
                    os.makedirs(os.path.dirname(full_pth))
                print(full_pth)
                print(batch_data['origin_key'][0][1:])
                print(batch_data['space_key'][0][1:])
                origin_ls = [float(i) for i in batch_data['origin_key'][0]][1:]
                space_ls = [float(i) for i in batch_data['space_key'][0]][1:]
                print('type_1', type(space_ls[0]))
                futil.save_itk(full_pth, slice, origin_ls, space_ls)  # slice does not have origin and space along z

    print('Finish all things!')


if __name__ == "__main__":

    train()
