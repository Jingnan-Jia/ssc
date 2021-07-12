# -*- coding: utf-8 -*-
# @Time    : 7/5/21 4:01 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import random
from typing import Dict, Optional, Union, Hashable, Mapping
import pandas as pd
import torch
import myutil.myutil as futil
import numpy as np
from monai.transforms import RandGaussianNoise
from monai.transforms import ScaleIntensityRange, RandGaussianNoise, MapTransform, AddChannel
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomAffine
import matplotlib.pyplot as plt

TransInOut = Mapping[Hashable, Optional[Union[np.ndarray, str]]]

class LoadDatad:
    # def __init__(self):
        # self.normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
    def __call__(self, data: Mapping[str, Union[np.ndarray, str]]) -> Dict[str, np.ndarray]:
        fpath = data['fpath_key']
        world_pos = np.array(data['world_key']).astype(np.float32)
        data_x = futil.load_itk(fpath, require_ori_sp=True)
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
                'fpath_key': fpath}  # full path, a string

        return data


class AddChanneld:
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        """
        Apply the transform to `img`.
        """
        d = dict(data)
        d['image_key'] = d['image_key'][None]
        return d


class NormImgPosd:
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)

        if isinstance(d['image_key'], torch.Tensor):
            mean, std = torch.mean(d['image_key']), torch.std(d['image_key'])
        else:
            mean, std = np.mean(d['image_key']), np.std(d['image_key'])

        d['image_key'] = d['image_key'] - mean
        d['image_key'] = d['image_key'] / std
        # print('end norm')

        return d


class RandGaussianNoised:
    def __init__(self, *args, **kargs):
        self.noise = RandGaussianNoise(*args, **kargs)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        print('start add noise')
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
    def __init__(self, z_size, y_size, x_size):
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
    def __init__(self, z_size, y_size, x_size):
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
    and add a level_key to data dick.
    """

    def __init__(self, level_node: int, train_on_level: int, height: int, rand_start: bool, start: Optional[int] = None):
        """

        :param level: int
        :param rand_start: during training (rand_start=True), inference (rand_start=False).
        :param start: If rand_start is True, start would be ignored.
        """
        self.level_node = level_node
        self.train_on_level = train_on_level
        self.height = height
        self.rand_start = rand_start
        self.start = start

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        d = dict(data)

        if self.height > d['image_key'].shape[0]:
            raise Exception(
                f"desired height {self.height} is greater than image size along z {d['image_key'].shape[0]}")

        if self.train_on_level != 0:
            self.level = self.train_on_level  # only input data from this level
        else:
            if self.level_node!=0:
                self.level = random.randint(1, 5)  # 1,2,3,4,5 level is randomly selected
            else:
                raise Exception("Do not need CropLevelRegiond because level_node==0 and train_on_level==0")

        d['label_in_img_key'] = np.array(d['ori_label_in_img_key'][self.level - 1]).reshape(-1, )  # keep the current label for the current level
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
        d['level_key'] = np.array(self.level).reshape(-1, )

        return d


class CropCorseRegiond:
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

        :param level: int
        :param rand_start: during training (rand_start=True), inference (rand_start=False).
        :param start: If rand_start is True, start would be ignored.
        """
        self.level_node = level_node
        self.train_on_level = train_on_level
        self.height = height
        self.rand_start = rand_start
        self.start = start
        self.data_fpath = data_fpath
        self.pred_world_fpath = pred_world_fpath
        self.df_data = pd.read_csv(self.data_fpath, delimiter=',')
        self.df_pred_world = pd.read_csv(self.pred_world_fpath, delimiter=',')
        if len(self.df_pred_world)==(len(self.df_data)+1):  # df_data should not have header
            self.df_data = pd.read_csv(self.data_fpath, header=None, delimiter=',')
            self.df_data.columns = ['img_fpath', 'world_pos']
        elif len(self.df_pred_world)==len(self.df_data):
            pass
        else:
            raise Exception(f"the length of data: {len(self.df_data)} and pred_world: {len(self.df_pred_world)} is not the same")


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

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        d = dict(data)
        d = self.update_label(d)
        if self.height > d['image_key'].shape[0]:
            raise Exception(
                f"desired height {self.height} is greater than image size along z {d['image_key'].shape[0]}")

        if self.train_on_level != 0:
            self.level = self.train_on_level  # only input data from this level
        else:
            if self.level_node!=0:
                self.level = random.randint(1, 5)  # 1,2,3,4,5 level is randomly selected
            else:
                raise Exception("Do not need CropLevelRegiond because level_node==0 and train_on_level==0")
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
#     def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
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
#     def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
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

