# -*- coding: utf-8 -*-
# @Time    : 7/6/21 7:17 PM
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

import confusion
import myresnet3d
from networks import med3d_resnet as med3d
from networks import get_net_pos

from mytrans import LoadDatad, MyNormalizeImagePosd, AddChannelPosd, RandomCropPosd, \
    RandGaussianNoise, CenterCropPosd, CropLevelRegiond, ComposePosd, CropCorseRegiond
from mydata import AllLoader
from path import Path
from tool import record_GPU_info

def SlidingLoader(fpath, world_pos, z_size, stride=1, batch_size=1, mode='valid', args=None):
    print(f'start load {fpath} for sliding window inference')
    xforms = [LoadDatad(), MyNormalizeImagePosd()]

    trans = ComposePosd(xforms)


    data = trans(data={'fpath_key': fpath, 'world_key': world_pos})

    raw_x = data['image_key']
    data['label_in_img_key'] = np.array(data['label_in_img_key'][args.train_on_level - 1])

    label = data['label_in_img_key']
    print('data_world_key', data['world_key'])

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
            if args.infer_2nd:
                mypath2 = Path(args.eval_id)
                crop = CropCorseRegiond(level=args.train_on_level, height=args.z_size, start=start,
                                        data_fpath=mypath2.data(mode), pred_world_fpath=mypath2.pred_world(mode))
            else:
                crop = CropLevelRegiond(level_node=args.level_node, train_on_level=args.train_on_level, height=args.z_size, rand_start=False, start=start)
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
    def __init__(self, net, dataloader, mode, mypath, args):
        self.net = net
        self.dataloader = dataloader
        self.mode = mode
        self.mypath = mypath
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = self.net.to(self.device).eval()
        self.amp = True if torch.cuda.is_available() else False
        self.args = args

    def run(self):
        for batch_data in self.dataloader:
            for idx in range(len(batch_data['image_key'])):
                print('len_batch', len(batch_data))
                print(batch_data['fpath_key'][idx], batch_data['ori_world_key'][idx])
                sliding_loader = SlidingLoader(batch_data['fpath_key'][idx], batch_data['ori_world_key'][idx],
                                               z_size=self.args.z_size, stride=self.args.infer_stride, batch_size=self.args.batch_size,
                                               mode=self.args.mode, args=self.args)
                pred_in_img_ls = []
                pred_in_patch_ls = []
                label_in_patch_ls = []
                for patch, new_label, start in sliding_loader:
                    batch_x = patch.to(self.device)
                    if self.args.level_node != 0:
                        batch_level = torch.ones((len(batch_x), 1)) * self.args.train_on_level
                        batch_level = batch_level.to(self.device)
                        print('batch_level', batch_level.clone().cpu().numpy())
                        batch_x = [batch_x, batch_level]

                    if self.amp:
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

                batch_label: np.ndarray = batch_data['label_in_img_key'][idx].cpu().detach().numpy().astype(int)
                batch_preds_ave: np.ndarray = np.mean(pred_in_img_all, 0)
                batch_preds_int: np.ndarray = batch_preds_ave.astype(int)
                batch_preds_world: np.ndarray = batch_preds_ave * batch_data['space_key'][idx][0].item() + \
                                                batch_data['origin_key'][idx][0].item()
                batch_world: np.ndarray = batch_data['world_key'][idx].cpu().detach().numpy()
                head = ['L1', 'L2', 'L3', 'L4', 'L5']
                if self.args.train_on_level:
                    head = [head[self.args.train_on_level - 1]]
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

                if self.args.train_on_level:
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


def record_best_preds(net: torch.nn.Module, dataloader_dict: Dict[str, DataLoader], mypath: Path, args):
    net.load_state_dict(torch.load(mypath.model_fpath))  # load the best weights to do evaluation
    for mode, dataloader in dataloader_dict.items():
        evaluater = Evaluater(net, dataloader, mode, mypath, args)
        evaluater.run()
