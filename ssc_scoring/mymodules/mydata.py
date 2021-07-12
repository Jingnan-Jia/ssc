# -*- coding: utf-8 -*-
# @Time    : 7/5/21 6:02 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import csv
import glob
import os
from typing import List, Union, Tuple

import monai
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from ssc_scoring.mymodules.composed_trans import xformd_pos, xformd_score
from ssc_scoring.mymodules.datasets import SysDataset
from ssc_scoring.mymodules.tool import sampler_by_disext


class LoaderInit:
    """
    load_per_xy(), xformd() and load() need to be implemented.
    """
    def __init__(self, resample_z, mypath, label_file, kfold_seed, fold, total_folds, ts_level_nb, level_node,
                 train_on_level, z_size, y_size, x_size, batch_size, workers):
        self.resample_z = resample_z
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

    def load_data_of_pats(self, dir_pats: Union[List, np.ndarray]) -> Tuple[list, list]:
        x, y = [], []
        for dir_pat in dir_pats:
            x_pat, y_pat = self.load_per_xy(dir_pat)
            if isinstance(x_pat, list):
                x.extend(x_pat)
                y.extend(y_pat)
            else:
                x.append(x_pat)
                y.append(y_pat)
        return x, y

    def split_dir_pats(self):
        if self.mypath.project_name == 'score':
            dir_pats = sorted(glob.glob(os.path.join(self.mypath.ori_data_dir, "Pat_*")))
        else:
            dir_pats = sorted(glob.glob(os.path.join(self.mypath.dataset_dir(self.resample_z), "Pat_*", "CTimage.mha")))
            if len(dir_pats) == 0:  # does not find patients in this directory
                dir_pats = sorted(glob.glob(os.path.join(self.mypath.dataset_dir(self.resample_z), "Pat_*CTimage*.mha")))
            # if len(dir_pats) == 0:
            #     dir_pats = sorted(glob.glob(os.path.join(self.mypath.dataset_dir, "Pat_*", "CTimage.mha")))

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
        raise NotImplementedError

    def load_per_xy(self, *args, **kargs):
        raise NotImplementedError

    def load(self, *args, **kargs):
        raise NotImplementedError  #


class LoadPos(LoaderInit):
    def load_per_xy(self, dir_pat: str) -> Tuple[str, np.ndarray]:
        data_name = dir_pat
        idx = int(dir_pat.split('Pat_')[-1][:3])
        data_label = []
        for level in [1, 2, 3, 4, 5]:
            y = self.df_excel.at[idx, 'L' + str(level) + '_pos']
            data_label.append(y)
        return data_name, np.array(data_label)


    def xformd(self, mode):
        return xformd_pos(mode, level_node=self.level_node,
                   train_on_level=self.train_on_level,
                   z_size=self.z_size, y_size = self.y_size, x_size=self.x_size)


    def load(self):
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = self.prepare_data()
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = tr_x[:6], tr_y[:6], vd_x[:6], vd_y[:6], ts_x[:6], ts_y[:6]
        print(tr_x)
        cache_nb = 10 if len(tr_x) < 50 else 50

        tr_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(tr_x, tr_y)]
        vd_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(vd_x, vd_y)]
        ts_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(ts_x, ts_y)]

        tr_dataset = monai.data.SmartCacheDataset(data=tr_data, transform=self.xformd('train'), replace_rate=0.2,
                                                  cache_num=cache_nb, num_init_workers=4, num_replace_workers=8)
        vdaug_dataset = monai.data.CacheDataset(data=vd_data, transform=self.xformd('train'), num_workers=4,
                                                cache_rate=1)
        vd_dataset = monai.data.CacheDataset(data=vd_data, transform=self.xformd('valid'), num_workers=4, cache_rate=1)
        ts_dataset = monai.data.PersistentDataset(data=ts_data, transform=self.xformd('test'),
                                                  cache_dir="persistent_cache")

        train_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
                                      pin_memory=True, persistent_workers=True)
        validaug_dataloader = DataLoader(vdaug_dataset, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.workers,
                                         pin_memory=True, persistent_workers=True)
        valid_dataloader = DataLoader(vd_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                      pin_memory=True, persistent_workers=True)
        test_dataloader = DataLoader(ts_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                     pin_memory=True, persistent_workers=True)

        return train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader


class LoadScore(LoaderInit):
    def __init__(self, mypath, label_file, kfold_seed, args):
        super().__init__(resample_z=None, mypath=mypath, label_file=label_file, kfold_seed=kfold_seed,
                         fold=args.fold, total_folds=args.total_folds, ts_level_nb=args.ts_level_nb, level_node=0,
                 train_on_level=0, z_size=None, y_size=None,
                         x_size=None, batch_size=10, workers=args.workers)
        self.sys = args.sys
        self.sampler = args.sampler
        self.sys_ratio = args.sys_ratio
        self.args = args

    def load_per_xy(self, dir_pat: str) -> Tuple[List, List]:
        """
        Load the data for the specific level.
        :param df_excel:
        :param dir_pat:
        :param level:
        :return:
        """
        x, y = [], []
        for level in [1, 2, 3, 4, 5]:
            x_level, y_level = self.load_data_of_a_level(dir_pat, level)
            x.extend(x_level)
            y.extend(y_level)
        return x, y
    def load_data_of_a_level(self, dir_pat, level):
        df_excel = self.df_excel
        file_prefix = "Level" + str(level)
        # 3 neighboring slices for one level
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


    def xformd(self, *arg, **karg):
        return xformd_score(*arg, **karg)


    def load(self):
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = self.prepare_data()
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = tr_x[:10], tr_y[:10], vd_x[:10], vd_y[:10], ts_x[:10], ts_y[:10]
        tr_dataset = SysDataset(tr_x, tr_y, transform=self.xformd("train", synthesis=self.sys, args=self.args), synthesis=self.sys)
        vd_data_aug = SysDataset(vd_x, vd_y, transform=self.xformd("validaug", synthesis=self.sys, args=self.args),
                                 synthesis=self.sys)
        vd_dataset = SysDataset(vd_x, vd_y, transform=self.xformd("valid", synthesis=False, args=self.args), synthesis=False)
        ts_dataset = SysDataset(ts_x, ts_y, transform=self.xformd("test", synthesis=False, args=self.args),
                                synthesis=False)  # valid original data
        sampler = sampler_by_disext(tr_y, self.sys_ratio) if self.sampler else None
        print(f'sampler is {sampler}')

        # else:
        #     raise Exception("synthesis_data can not be set with sampler !")

        tr_shuffle = True if sampler is None else False
        train_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=tr_shuffle, num_workers=self.workers,
                                      sampler=sampler, pin_memory=True)
        validaug_dataloader = DataLoader(vd_data_aug, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                         pin_memory=True)

        valid_dataloader = DataLoader(vd_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                      pin_memory=True)
        # valid_dataloader = train_dataloader
        test_dataloader = DataLoader(ts_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                     pin_memory=True)
        return train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader



