# -*- coding: utf-8 -*-
# @Time    : 7/11/21 2:31 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import random

import myutil.myutil as futil
import numpy as np
import torch
# import streamlit as st
from tqdm import tqdm

from monai.transforms import ScaleIntensityRange

from torch.utils.data import Dataset


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



class SysDataset(Dataset):
    """SSc scoring dataset."""

    def __init__(self, data_x_names, data_y_list, index: list = None, transform=None, synthesis=False):

        self.data_x_names, self.data_y_list = np.array(data_x_names), np.array(data_y_list)
        if index is not None:
            self.data_x_names = self.data_x_names[index]
            self.data_y_list = self.data_y_list[index]
        print('loading data ...')
        self.data_x = [futil.load_itk(x, require_ori_sp=True) for x in tqdm(self.data_x_names)]
        self.systhesis = synthesis
        if self.systhesis:
            mask_end = "_lung_mask"
            self.lung_masks_names =  [x.split('.mha')[0]+ mask_end + ".mha" for x in tqdm(self.data_x_names)]

            self.lung_masks = [futil.load_itk(x, require_ori_sp=False) for x in tqdm(self.lung_masks_names)]


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
        if self.systhesis:
            new_dict = {'lung_mask_key': self.lung_masks[idx]}
            data.update(new_dict)
        if self.transform:
            data = self.transform(data)
        return data
