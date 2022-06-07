# -*- coding: utf-8 -*-
# @Time    : 7/5/21 4:01 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import os
import random
from typing import Dict, Optional, Union, Hashable, Sequence

from medutils.medutils import load_itk

import numpy as np
import pandas as pd
import torch
from monai.transforms import RandGaussianNoise, Transform, RandomizableTransform, ThreadUnsafe
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomAffine

TransInOut = Dict[Hashable, Optional[Union[np.ndarray, torch.Tensor, str, int]]]
# Note: all transforms here must inheritage Transform, Transform, or RandomTransform.


class LoadDatad(Transform):
    """Load data. The output image values range from -1500 to 1500.

        #. Load data from `data['fpath_key']`;
        #. truncate data image to [-1500, 1500];
        #. Get origin, spacing;
        #. Calculate relative slice number;
        #. Build a data dict.

    Examples:
        :func:`ssc_scoring.mymodules.composed_trans.xformd_pos2score` and
        :func:`ssc_scoring.mymodules.composed_trans.xformd_pos`

    """

    def __call__(self, data: TransInOut) -> TransInOut:
        fpath = data['fpath_key']
        world_pos = np.array(data['world_key']).astype(np.float32)
        data_x = load_itk(fpath, require_ori_sp=True)
        # print('load a image')
        x = data_x[0]  # shape order: z, y, x
        # print("cliping ... ")
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
                'label_in_img_key': data_y_np,  # label in  the whole image, keep fixed, a np.array with shape(-1, )
                'label_in_patch_key': data_y_np,  # relative label (slice number) in  a patch, np.array with shape(-1, )
                'ori_label_in_img_key': data_y_np,  # label in  the whole image, keep fixed, a np.array with shape(-1, )
                'world_key': world_pos,  # world position in mm, keep fixed,  a np.array with shape(-1, )
                'ori_world_key': world_pos,  # world position in mm, keep fixed,  a np.array with shape(-1, )
                'space_key': sp,  # space,  a np.array with shape(-1, )
                'origin_key': ori,  # origin,  a np.array with shape(-1, )
                'fpath_key': fpath}  # full path, a array of string

        return data


class AddChanneld(Transform):
    """Add a channel to the first dimension."""
    def __init__(self, key='image_key'):
        self.key = key

    def __call__(self, data: TransInOut) -> TransInOut:
        data[self.key] = data[self.key][None]
        return data


class NormImgPosd(Transform):
    """Normalize image to standard Normalization distribution"""
    def __init__(self, key='image_key'):
        self.key = key

    def __call__(self, data: TransInOut) -> TransInOut:
        d = data

        if isinstance(d[self.key], torch.Tensor):
            mean, std = torch.mean(d[self.key]), torch.std(d[self.key])
        else:
            mean, std = np.mean(d[self.key]), np.std(d[self.key])

        d[self.key] = d[self.key] - mean
        d[self.key] = d[self.key] / std
        # print('end norm')

        return d


class RandGaussianNoised(RandomizableTransform):
    """ Add noise to data[key]"""

    def __init__(self, key='image_key', **kargs):
        super().__init__()
        self.noise = RandGaussianNoise(**kargs)
        self.key = key

    def __call__(self, data: TransInOut) -> TransInOut:
        d = dict(data)
        d[self.key] = self.noise(d[self.key])
        return d


def cropd(d: TransInOut, start: Sequence[int], z_size: int, y_size: int, x_size: int, key: str = 'image_key') -> TransInOut:
    """ Crop 3D image

    :param d: data dict, including an 3D image
    :param key: image key to be croppeed
    :param start: start coordinate values, ordered by [z, y, x]
    :param z_size: sub-image size along z
    :param y_size: sub-image size along y
    :param x_size: sub-image size along x
    :return: data dict, including cropped sub-image, along with updated `label_in_patch_key`
    """
    d[key] = d[key][start[0]:start[0] + z_size, start[1]:start[1] + y_size,
             start[2]:start[2] + x_size]
    d['label_in_patch_key'] = d['label_in_img_key'] - start[0]  # image is shifted up, and relative position down
    d['label_in_patch_key'][d['label_in_patch_key'] < 0] = 0  # position outside the edge would be set as edge
    d['label_in_patch_key'][d['label_in_patch_key'] > z_size] = z_size  # position outside the edge would be set as edge
    return d


class CenterCropPosd(RandomizableTransform):
    """ Crop image at the center point."""
    def __init__(self, z_size, y_size, x_size, key='image_key'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.key = key
        super().__init__()

    def __call__(self, data: TransInOut) -> TransInOut:
        keys = set(data.keys())
        assert {self.key, 'label_in_img_key', 'label_in_patch_key'}.issubset(keys)
        img_shape = data[self.key].shape
        # print(f'img_shape: {img_shape}')
        assert img_shape[0] >= self.z_size
        assert img_shape[1] >= self.y_size
        assert img_shape[2] >= self.x_size
        middle_point = [shape // 2 for shape in img_shape]
        start = [middle_point[0] - self.z_size // 2, middle_point[1] - self.y_size // 2,
                 middle_point[2] - self.y_size // 2]
        data = cropd(data, start, self.z_size, self.y_size, self.x_size)

        return data


class RandomCropPosd(RandomizableTransform):
    """ Random crop a patch from a 3D image, and update the labels"""

    def __init__(self, z_size, y_size, x_size, key='image_key'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.key = key
        super().__init__()

    def __call__(self, data: TransInOut) -> TransInOut:
        d = dict(data)
        # if 'image_key' in data:
        img_shape = d[self.key].shape  # shape order: z,y x
        assert img_shape[0] >= self.z_size
        assert img_shape[1] >= self.y_size
        assert img_shape[2] >= self.x_size

        valid_range = (img_shape[0] - self.z_size, img_shape[1] - self.y_size, img_shape[2] - self.x_size)
        start = [random.randint(0, v_range) for v_range in valid_range]
        d = cropd(d, start, self.z_size, self.y_size, self.x_size, self.key)
        return d


class CropPosd(ThreadUnsafe):
    def __init__(self, start: Optional[int], height: Optional[int], key = 'image_key' ):
        self.start = start
        self.key = key
        self.height = height
        self.end = int(self.start + self.height)

    def __call__(self, data):
        d = data
        if self.height > d[self.key].shape[0]:
            raise Exception(f"desired height {self.height} is greater than image size_z {d['image_key'].shape[0]}")
        if self.end > d[self.key].shape[0]:
            self.end = d[self.key].shape[0]
            self.start = self.end - self.height
        d[self.key] = d[self.key][self.start: self.end].astype(np.float32)

        d['label_in_patch_key'] = d['label_in_img_key'] - self.start
        d['world_key'] = d['ori_world_key']

        return d

class RandCropLevelRegiond(RandomizableTransform):
    """ Crop a 3D patch which include one specified level (the position of 5 levels are given).
    If start is not given, use random start pointt; else use the given start coordinate values.
    Only keep the label of the current level: label_in_img.shape=(1,), label_in_patch.shape=(1,)
    and add a level_key to data dick.

    # do not need to check the start.
    """

    def __init__(self, level_node: int, train_on_level: int, height: int, rand_start: bool,
                 start: Optional[int] = None, key='image_key'):
        """
        used if self.train_on_level or self.level_node:

        :param rand_start: during training (rand_start=True), inference (rand_start=False).
        :param start: If rand_start is True, start would be ignored.
        """
        self.level_node = level_node
        self.train_on_level = train_on_level
        self.height = height
        self.rand_start = rand_start
        self.start = int(start)
        self.key = key
        super().__init__()

    def __call__(self, data: TransInOut) -> TransInOut:
        d = data

        if self.height > d[self.key].shape[0]:
            raise Exception(f"desired height {self.height} is greater than image size_z {d['image_key'].shape[0]}")

        if self.train_on_level != 0:
            level = self.train_on_level  # only input data from this level
        else:
            level = random.randint(1, 5)  # 1,2,3,4,5 level is randomly selected

        d['label_in_img_key'] = np.array(d['ori_label_in_img_key'][level - 1]).reshape(-1, )
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

        d['world_key'] = np.array(d['ori_world_key'][level - 1]).reshape(-1, )
        d['level_key'] = np.array(level).reshape(-1, )

        end = int(start + self.height)
        if end > d[self.key].shape[0]:
            end = d[self.key].shape[0]
            start = end - self.height
        d[self.key] = d[self.key][start: end].astype(np.float32)
        d['label_in_patch_key'] = d['label_in_img_key'] - self.start

        return d


class CropCorseRegiond(RandomizableTransform):
    """
    Only keep the label of the current level: label_in_img.shape=(1,), label_in_patch.shape=(1,)
    and add a level_key to data dick.
    d['corse_pred_in_img_key'].shape is (5,)
    d['corse_pred_in_img_1_key'].shape is (1,)
    """

    def __init__(self,
                 level_node: int,
                 train_on_level: int,
                 height: int,
                 rand_start: bool,
                 start: Optional[int] = None,
                 data_fpath: Optional[str] = None,
                 pred_world_fpath: Optional[str] = None):
        """

        :param rand_start: during training (rand_start=True), inference (rand_start=False).
        :param start: If rand_start is True, start would be ignored.
        """
        super().__init__()
        self.level_node = level_node
        self.train_on_level = train_on_level
        self.height = height
        self.rand_start = rand_start
        self.start = start
        self.data_fpath = data_fpath
        self.pred_world_fpath = pred_world_fpath
        self.df_data = pd.read_csv(self.data_fpath, delimiter=',')
        self.df_pred_world = pd.read_csv(self.pred_world_fpath, delimiter=',')
        if len(self.df_pred_world) == (len(self.df_data) + 1):  # df_data should not have header
            self.df_data = pd.read_csv(self.data_fpath, header=None, delimiter=',')
            self.df_data.columns = ['img_fpath', 'world_pos']
        elif len(self.df_pred_world) == len(self.df_data):
            pass
        else:
            raise Exception(
                f"the length of data: {len(self.df_data)} and pred_world: {len(self.df_pred_world)} is not the same")

    def get_img_idx(self, image_fpath: str) -> int:
        id_str = image_fpath.split("Pat_")[-1].split("_")[0]  # like: Pat_012
        for i in range(len(self.df_data)):
            if id_str in self.df_data['img_fpath'].iloc[i]:
                return i
        raise Exception(f"Can not find the image id from data file")

    def corse_pred(self, image_fpath):
        img_idx = self.get_img_idx(image_fpath)
        level_pred = self.df_pred_world.iloc[img_idx].to_numpy()
        return level_pred

    def update_label(self, d):
        corse_pred: int = self.corse_pred(d['fpath_key'])  # get corse predictions for this data
        d['corse_pred_in_img_key'] = corse_pred
        return d

    def __call__(self, data: TransInOut) -> TransInOut:
        d = dict(data)
        d = self.update_label(d)
        if self.height > d['image_key'].shape[0]:
            raise Exception(
                f"desired height {self.height} is greater than image size along z {d['image_key'].shape[0]}")

        if self.train_on_level != 0:
            self.level = self.train_on_level  # only input data from this level
        else:
            if self.level_node != 0:
                self.level = random.randint(1, 5)  # 1,2,3,4,5 level is randomly selected
            else:
                raise Exception("Do not need RandCropLevelRegiond because level_node==0 and train_on_level==0")
        # keep the current label for the current level
        d['corse_pred_in_img_1_key'] = np.array(d['corse_pred_in_img_key'][self.level - 1]).reshape(-1, )
        label: int = d['corse_pred_in_img_1_key']  # z slice number
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

        d['label_in_patch_key'] = d['corse_pred_in_img_1_key'] - start  # todo: clear the concept here

        d['world_key'] = np.array(d['world_key'][self.level - 1]).reshape(-1, )
        d['level_key'] = np.array(self.level).reshape(-1, )

        return d


#
# class CropCorseRegiond:
#     """
#     1. Receive a data dict, get the file fpath,
#     2. Get its 5-level predictions from corse results.
#     3. Get one patches whose center is the {level}th predicted positions.
#     """
#
#     def __init__(self, level, height, data_fpath, pred_world_fpath):
#         self.level = level
#         self.level_name = "L" + str(self.level)
#         self.height = height
#         self.start = start
#         self.data_fpath = data_fpath
#         self.pred_world_fpath = pred_world_fpath
#         self.df_data = pd.read_csv(self.data_fpath, delimiter=',')
#         self.df_pred_world = pd.read_csv(self.pred_world_fpath, delimiter=',')
#         if len(self.df_pred_world)==(len(self.df_data)+1):  # df_data should not have header
#             self.df_data = pd.read_csv(self.data_fpath, header=None, delimiter=',')
#             self.df_data.columns = ['img_fpath', 'world_pos']
#         elif len(self.df_pred_world)==len(self.df_data):
#             pass
#         else:
#             raise Exception(f"the length of data: {len(self.df_data)} and pred_world: {len(self.df_pred_world)} is not the same")
#
#     def get_img_idx(self, image_fpath: str) -> int:
#         id_str = image_fpath.split("Pat_")[-1].split("_")[0]  # like: Pat_012
#         for i in range(len(self.df_data)):
#             if id_str in self.df_data['img_fpath'].iloc[i]:
#                 return i
#         raise Exception(f"Can not find the image id from data file")
#
#     def corse_pred(self, image_fpath):
#         img_idx = self.get_img_idx(image_fpath)
#         level_pred = self.df_pred_world[self.level_name].iloc[img_idx]
#         return level_pred
#
#     def crop(self, img, label_img):
#         start = label_img - self.height
#         end = label_img + self.height
#         if start < 0:
#             start = 0
#             end = self.height
#         if end > img.shape[0]:
#             end = img.shape[0]
#             start = end - self.height
#
#         patch = img[start, end]
#         return patch
#
#
#     def __call__(self, data: TransInOut) -> TransInOut:
#         d = dict(data)  # get data
#         corse_pred: int = self.corse_pred(d['fpath_key'])  # get corse predictions for this data
#         corse_pred = corse_pred[self.level - 1]
#
#         if self.height > d['image_key'].shape[0]:
#             raise Exception(
#                 f"desired height {self.height} is greater than image size along z {d['image_key'].shape[0]}")
#
#         patch = self.crop(d['image_key'], corse_pred)
#
#         d['image_key'] = patch.astype(np.float32)
#
#         d['label_in_img_key'] = np.array(d['label_in_img_key'][self.level - 1]).reshape(-1, )
#         d['label_in_patch_key'] = d['label_in_img_key'] - start
#
#         d['world_key'] = np.array(d['world_key'][self.level - 1]).reshape(-1, )
#         d['level_key'] = np.array(self.level).reshape(-1, )
#
#         return d
#

# class ComposePosd:
#     """My Commpose to handle with img and label at the same time.
#
#     """
#
#     def __init__(self, transforms):
#         self.transforms = transforms
#
#     def __call__(self, data: TransInOut) -> TransInOut:
#         for t in self.transforms:
#             data = t(data)
#         return data
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + '('
#         for t in self.transforms:
#             format_string += '\n'
#             format_string += '    {0}'.format(t)
#         format_string += '\n)'
#         return format_string
#


class RandomAffined(RandomizableTransform):
    def __init__(self, key, *args, **kwargs):
        self.random_affine = RandomAffine(*args, **kwargs)
        self.key = key
        super().__init__()

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.random_affine(d[self.key])
        return d


class CenterCropd(Transform):
    def __init__(self, key, *args, **kargs):
        self.center_crop = CenterCrop(*args, **kargs)
        self.key = 'image_key'

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.center_crop(d[self.key])
        return d


class RandomHorizontalFlipd(RandomizableTransform):
    def __init__(self, key, *args, **kargs):
        self.random_hflip = RandomHorizontalFlip(*args, **kargs)
        self.key = key
        super().__init__()

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.random_hflip(d[self.key])
        return d


class RandomVerticalFlipd(RandomizableTransform):
    def __init__(self, key, *args, **kargs):
        self.key = key
        self.random_vflip = RandomVerticalFlip(*args, **kargs)
        super().__init__()

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.random_vflip(d[self.key])
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


class Clipd(Transform):
    def __init__(self, min: int, max: int, key: str = 'image_key'):
        self.clip = Clip(min, max)
        self.key  = key

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.clip(d[self.key])
        return d


class CascadedSlices:
    """
    0. get 2D images with their "real labels" from excel
    1. get the 3D images and their 5 world positions using the Pos dataset loader
    2. Get the position predictions of each 3D images, replace the former 5 world positions, using
    corse_pred() in CropCorsePosd
    3. Get 5 slices by 1 and 2
    4. output 5 slices from 3 and their "real labels" from 1
    """

    def __call__(self, *args, **kwargs):
        return None


class CoresPosd(Transform):
    """The predicted **world position** is extracted.

    We do not extract 'relative slice number' because the spacing may be different for each experiments.

    """
    def __init__(self, corse_fpath, data_fpath):
        self.corse_fpath = corse_fpath  # valid_pred_world.csv
        self.data_fpath = data_fpath  # valid_data.csv


    def __call__(self, data):
        print('start corse pos extration ...')
        df_corse_pos = pd.read_csv(self.corse_fpath, delimiter=',')
        df_data = pd.read_csv(self.data_fpath, delimiter=',', header=None)  # no header for dat.csv
        print(f'len_cores_pred: ', len(df_corse_pos))
        print(f'len_data: ', len(df_data))
        if len(df_corse_pos) != len(df_data):
            print(f'df_corse_pos:{df_corse_pos}')
            print(f'df_data: {df_data}')
            raise Exception(f'the lenth of {self.corse_fpath} and {self.data_fpath} IS Not equal.')

        pat_idx = None
        # print("df_data", df_data)
        for idx, row in df_data.iterrows():
            # print(f'idx: ', idx)
            # print('========')
            # print(data['fpath_key'].split('Pat_')[-1][:3])
            # print(row.iloc[0].split('Pat_')[-1][:3])
            if data['fpath_key'].split('Pat_')[-1][:3] == row.iloc[0].split('Pat_')[-1][:3]:
                pat_idx = idx
                break
        print(f'pat_idx: {pat_idx}')

        corse_pred = df_corse_pos.iloc[pat_idx]
        # print('type:', type(corse_pred))
        data['coarse_pred_world_key'] = corse_pred.to_numpy().astype(np.int32)
        # data['fpath'] = np.array(data['fpath'])
        # print('corse_pred', data['corse_pred_int_key'])
        # print('type_pred', type(data['corse_pred_int_key']))
        print('coarse_pred_world_key:')
        print(data['coarse_pred_world_key'][0],
              data['coarse_pred_world_key'][1],
              data['coarse_pred_world_key'][2],
              data['coarse_pred_world_key'][3],
              data['coarse_pred_world_key'][4])
        return data


class SliceFromCorsePosd(Transform):
    def __call__(self, d: dict):
        print('start slice from corse pos ...')
        img_3d = d['image_key']
        img_2d_ls = []
        img_2d_name_ls = []
        pat_id = d['fpath_key'].split('Pat_')[-1][:3]
        save_pat_dir = 'Pat_' + pat_id
        for i, pred_world in enumerate([j for j in d['coarse_pred_world_key']]):
            space_z: float = d['space_key'][0]
            origin_z: float = d['origin_key'][0]
            slice_nb: int = int((pred_world - origin_z)/space_z)
            print(f'slice_nb: {slice_nb}')

            img_2d_ls.append(img_3d[slice_nb])
            img_2d_name_ls.append(os.path.join(save_pat_dir, 'Level' + str(i + 1) + '_middle.mha'))
            img_2d_ls.append(img_3d[slice_nb + 1])
            img_2d_name_ls.append(os.path.join(save_pat_dir, 'Level' + str(i + 1) + '_up.mha'))
            img_2d_ls.append(img_3d[slice_nb - 1])
            img_2d_name_ls.append(os.path.join(save_pat_dir, 'Level' + str(i + 1) + '_down.mha'))

        img_2d = np.array(img_2d_ls)
        img_2d_name_ls = np.array(img_2d_name_ls)
        d['image_key'] = img_2d.astype(np.float32)
        d['fpath2save'] = img_2d_name_ls
        d['fpath_key'] = np.array([d['fpath_key']])

        # print("d['fpath2save']", d['fpath2save'])
        # print(d.keys())
        # for key in d.keys():
        #     print(key, type(d[key]))
        return d
