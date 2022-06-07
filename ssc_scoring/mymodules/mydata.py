# -*- coding: utf-8 -*-
# @Time    : 7/5/21 6:02 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import csv
import glob
import os
import time
from typing import List, Union, Tuple
from abc import ABC, abstractmethod

import monai
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from monai.data import DataLoader
from monai.transforms import Transform
from ssc_scoring.mymodules.composed_trans import xformd_pos, xformd_score, xformd_pos2score
from ssc_scoring.mymodules.datasets import SynDataset
from ssc_scoring.mymodules.tool import sampler_by_disext


class LoaderInit(ABC):
    """Abstract class for `LoadScore`, `LoadPos` and `LoadPos2Score`. Methods of and `load`, `xformd` and `load_per_xy`,
    need to be implemented for the three class. The reason is:

    #. `load` method need to use different `Dataset`.
    #. `xformd` needs to use different `Transforms`.
    #. `load_per_xy` need to use different images (2D or 3D) and labels (scores or positions).

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
            # self.ts_id = [68, 83, 36, 187, 238, 12, 158, 189, 230, 11, 35, 37, 137, 144, 17, 42, 66, 70, 28, 64, 210, 3, 49,
            #          32, 236, 206, 194, 196, 7, 9, 16, 19, 20, 21, 40, 46, 47, 57, 58, 59, 60, 62, 116, 117, 118, 128,
            #          134, 216]
            self.ts_id = [3, 7, 9, 11, 12, 16, 17, 19, 20, 21, 28, 32, 35, 36, 37, 40, 42, 46, 47,
                           49, 57, 58, 59, 60, 62, 64, 66, 68, 70, 83, 116, 117, 118, 128, 134, 137,
                           144, 158, 187, 189, 194, 196, 206, 210, 216, 230, 236, 238]

        elif self.ts_level_nb == 250:  # 50 patients
            self.ts_id = [7, 9, 11, 12, 16, 19, 20, 21, 26, 28, 35, 36, 37, 40, 46, 47, 49, 57,
                          58, 59, 60, 62, 66, 68, 70, 77, 83, 116, 117, 118, 128, 134, 137, 140, 144,
                          149, 158, 170, 179, 187, 189, 196, 203, 206, 209, 210, 216, 227, 238, 263]
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
            if hasattr(self.mypath, 'corse_pred_dir'):
                data_dir = self.mypath.corse_pred_dir
            else:
                data_dir = self.mypath.ori_data_dir
            dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*")))
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

        # tr_pt = tr_vd_pt  # todo: the two lines needs to be commented if after berend's style training.
        # vd_pt = ts_pt  # todo: the two lines needs to be commented if after berend's style training.
        print('tr_pats:\n', tr_pt)
        print('vd_pats:\n', vd_pt)
        print('ts_pats:\n', ts_pt)


        tr_x, tr_y = self.load_data_of_pats(tr_pt)
        vd_x, vd_y = self.load_data_of_pats(vd_pt)
        ts_x, ts_y = self.load_data_of_pats(ts_pt)


        return tr_x, tr_y, vd_x, vd_y, ts_x, ts_y

    @abstractmethod
    def xformd(self, mode):
        raise NotImplementedError

    @abstractmethod
    def load_per_xy(self, *args, **kargs):
        raise NotImplementedError

    @abstractmethod
    def load(self, *args, **kargs):
        raise NotImplementedError  #


class LoadPos(LoaderInit):
    """ LoadData for Position prediction."""
    def __init__(self, resample_z, mypath, label_file, kfold_seed, fold, total_folds, ts_level_nb, level_node,
                 train_on_level, z_size, y_size, x_size, batch_size, workers):
        super().__init__(resample_z, mypath, label_file, kfold_seed, fold, total_folds, ts_level_nb, level_node,
                 train_on_level, z_size, y_size, x_size, batch_size, workers)

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
        for x, y, mode in zip([tr_x, vd_x, ts_x], [tr_y, vd_y, ts_y], ['train', 'valid', 'test']):
            self.save_xy(x, y, mode)
        # tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = tr_x[:2], tr_y[:2], vd_x[:2], vd_y[:2], ts_x[:2], ts_y[:2]
        # print(tr_x)
        # cache_nb = len(tr_x) if len(tr_x) < 50 else 50
        tr_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(tr_x, tr_y)]
        vd_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(vd_x, vd_y)]
        ts_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(ts_x, ts_y)]
        tr_dataset = monai.data.CacheDataset(data=tr_data, transform=self.xformd('train'), num_workers=4, cache_rate=1)
        vdaug_dataset = monai.data.CacheDataset(data=vd_data, transform=self.xformd('train'), num_workers=4, cache_rate=1)
        vd_dataset = monai.data.CacheDataset(data=vd_data, transform=self.xformd('valid'), num_workers=4, cache_rate=1)
        ts_dataset = monai.data.CacheDataset(data=ts_data, transform=self.xformd('valid'), num_workers=4, cache_rate=1)
        # self.workers = 0
        train_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
                                      pin_memory=False, persistent_workers=True)
        validaug_dataloader = DataLoader(vdaug_dataset, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.workers,
                                         pin_memory=False, persistent_workers=True)
        valid_dataloader = DataLoader(vd_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                      pin_memory=False, persistent_workers=True)
        test_dataloader = DataLoader(ts_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                     pin_memory=False, persistent_workers=True)


        # for i in range(3):
        #     t0 = time.time()
        #     idx = 0
        #     t_tmp0 = t0
        #     for tr in valid_dataloader:
        #         idx +=1
        #         print('load_t: ', time.time() - t_tmp0)
        #         t_tmp0 = time.time()
        #     t_tmp = time.time()
        #     print('time per batch data for train_dataloader via cache_nb50, cacherate0.2: ', (t_tmp - t0)/idx)
        # # tr_dataset.shutdown()
        # print('yes')


        #
        # tr_dataset = monai.data.PersistentDataset(data=tr_data, transform=self.xformd('train'), cache_dir="train_cache")
        # train_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
        #                               pin_memory=True, persistent_workers=True)
        # for i in range(5):
        #     t0 = time.time()
        #     idx = 0
        #     for tr in train_dataloader:
        #         idx +=1
        #     t_tmp = time.time()
        #     print('time per batch data for train_dataloader via cache_nb50, cacherate0.2: ', (t_tmp - t0)/idx)
        data_dt = {'train': {'dl': train_dataloader, 'ds': tr_dataset},
                 'valid': {'dl': valid_dataloader, 'ds': vd_dataset},
                 'validaug': {'dl': validaug_dataloader, 'ds': vdaug_dataset},
                 'test': {'dl': test_dataloader, 'ds': ts_dataset}}
        # tr_dataset.start()
        return data_dt


class LoadScore(LoaderInit):
    """ LoadData for Goh score prediction."""
    def __init__(self, mypath, label_file, kfold_seed, args, nb_img = None, require_lung_mask=False):
        super().__init__(resample_z=None, mypath=mypath, label_file=label_file, kfold_seed=kfold_seed,
                         fold=args.fold, total_folds=args.total_folds, ts_level_nb=args.ts_level_nb, level_node=0,
                         train_on_level=0, z_size=None, y_size=None,
                         x_size=None, batch_size=args.batch_size, workers=args.workers)
        self.sys = args.sys
        self.sampler = args.sampler
        self.sys_ratio = args.sys_ratio
        self.args = args
        self.nb_img = nb_img
        self.masked_by_lung = args.masked_by_lung
        self.require_lung_mask = require_lung_mask

    def load_per_xy(self, dir_pat: str) -> Tuple[List, List]:
        """
        Load the data for the specific level.
        :param dir_pat:
        :return:
        """
        x, y = [], []
        for level in [1, 2, 3, 4, 5]:
            x_level, y_level = self._load_data_of_a_level(dir_pat, level)
            x.extend(x_level)
            y.extend(y_level)
        return x, y

    def _load_data_of_a_level(self, dir_pat, level):
        df_excel = self.df_excel
        file_prefix = "Level" + str(level)
        # 3 neighboring slices for one level
        # print(dir_pat)
        if self.masked_by_lung:
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

    def xformd(self, *arg, **karg):
        return xformd_score(*arg, **karg)

    def load(self, merge: bool = False):
        """
        Load all dataloaders: train, validaug, valid, test.
        Args:
            merge: if True, merge all data together into valid_dataset, return valid_dataloader which include all data.
            This is switched on at :func:`ssc_scoring.run.train`.
        Returns:
            All data_loaders.

        """
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = self.prepare_data()
        for x, y, mode in zip([tr_x, vd_x, ts_x], [tr_y, vd_y, ts_y], ['train', 'valid', 'test']):
            self.save_xy(x, y, mode)
        # tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = tr_x[:2], tr_y[:2], vd_x[:2], vd_y[:2], ts_x[:2], ts_y[:2]

        if self.nb_img:
            tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = tr_x[:self.nb_img], tr_y[:self.nb_img], vd_x[:self.nb_img], \
                                                 vd_y[:self.nb_img], ts_x[:self.nb_img], ts_y[:self.nb_img]
        # print(f'valid_x for score: \n {vd_x}')

        if merge != 0:
            all_x = [*tr_x, *vd_x, *ts_x]
            all_y = [*tr_y, *vd_y, *ts_y]
            all_dataset = SynDataset(all_x, all_y, transform=self.xformd("valid", synthesis=self.sys, args=self.args,
                                                                         require_lung_mask=self.require_lung_mask, tr_x=tr_x),
                                     synthesis=False)
            all_loader = DataLoader(all_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                         pin_memory=True)
            return all_loader

        if not self.sampler:
            sampler = None
        else:
            if self.sys_ratio:
                sampler, self.args.sys_pro_in_0 = sampler_by_disext(tr_y, self.sys_ratio)
            else:
                sampler = sampler_by_disext(tr_y)

        tr_dataset = SynDataset(tr_x, tr_y, transform=self.xformd("train", synthesis=self.sys, args=self.args, tr_x=tr_x),
                                synthesis=self.sys, require_lung_mask=self.require_lung_mask)
        vd_data_aug = SynDataset(vd_x, vd_y, transform=self.xformd("validaug", synthesis=self.sys, args=self.args, tr_x=tr_x),
                                 synthesis=self.sys, require_lung_mask=self.require_lung_mask)
        vd_dataset = SynDataset(vd_x, vd_y, transform=self.xformd("valid", synthesis=False, args=self.args, tr_x=tr_x),
                                synthesis=False, require_lung_mask=self.require_lung_mask)  # valid original data, without synthetic images
        ts_dataset = SynDataset(ts_x, ts_y, transform=self.xformd("test", synthesis=False, args=self.args, tr_x=tr_x),
                                synthesis=False, require_lung_mask=self.require_lung_mask)  # test original data, without synthetic images


        print(f'sampler is {sampler}')
        tr_shuffle = True if sampler is None else False
        train_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=tr_shuffle, num_workers=self.workers,
                                      sampler=sampler, pin_memory=True)
        validaug_dataloader = DataLoader(vd_data_aug, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                         pin_memory=True)

        valid_dataloader = DataLoader(vd_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                      pin_memory=True)

        test_dataloader = DataLoader(ts_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                                     pin_memory=True)
        return train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader


class LoadPos2Score(LoaderInit):
    """TODO: None"""

    def __init__(self, mypath, label_file, kfold_seed, fold, total_folds, ts_level_nb):
        # Most arguments are not importent at all. The only arg which matter is resample_z=0, batch_size=1, worker=0.
        super().__init__(0, mypath, label_file, kfold_seed, fold, total_folds, ts_level_nb, 0, 0, None, None, None, 1, 0)


    def load_per_xy(self, dir_pat: str) -> Tuple[str, np.ndarray]:
        data_name = dir_pat
        idx = int(dir_pat.split('Pat_')[-1][:3])
        data_label = []
        for level in [1, 2, 3, 4, 5]:
            y = self.df_excel.at[idx, 'L' + str(level) + '_pos']
            data_label.append(y)
        return data_name, np.array(data_label)

    def xformd(self, mode):
        return xformd_pos2score(mode, self.mypath)

    def load(self):  # only load validation dataset
        tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = self.prepare_data()
        cache_nb = 10 if len(tr_x) < 50 else 50
        # print('valid_x for pos2score')
        # print(vd_x)

        tr_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(tr_x, tr_y)]
        vd_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(vd_x, vd_y)]
        ts_data = [{'fpath_key': x, 'world_key': y} for x, y in zip(ts_x, ts_y)]
        # tr_dataset = monai.data.CacheDataset(data=tr_data, transform=self.xformd('train'),  num_workers=4,
        #                                         cache_rate=0.1)
        # vdaug_dataset = monai.data.CacheDataset(data=vd_data, transform=self.xformd('train'), num_workers=4,
        #                                         cache_rate=0.1)
        vd_dataset = monai.data.CacheDataset(data=vd_data, transform=self.xformd('valid'), num_workers=0,
                                             cache_rate=1)
        # ts_dataset = monai.data.PersistentDataset(data=ts_data, transform=self.xformd('valid'),
        #                                           cache_dir="persistent_cache")
        # train_dataloader = iter(tr_dataset)
        # validaug_dataloader = iter(vdaug_dataset)
        valid_dataloader = iter(vd_dataset)
        # test_dataloader = iter(ts_dataset)

        # return train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader

        return valid_dataloader


