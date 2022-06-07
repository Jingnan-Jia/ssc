# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
sys.path.append("..")

import os

from medutils.medutils import save_itk

from ssc_scoring.mymodules.mydata import LoadPos2Score
from ssc_scoring.mymodules.path import PathPos

from ssc_scoring.mymodules.set_args_pos import get_args


def save_corse_slice(args, ex_dt):
    """Save corse slices according to the predicted slice numbers from 4-fold experiments.
    Detailed steps are:

    #. Get the dataloader in which the 2D slices have already been obtained.
    #. Save slices from the dataloader.

    :param args: args instance
    :param ex_dt: a dict with keys of [1,2,3,4] respresenting 4 folds and values of ID of 4 different experiments.
    :return: None. Reults are saved to disk.

    Example:

    >>> args = get_args()
    >>> ex_dt = {1: 193, 2: 194, 3: 276, 4: 277}
    >>> save_corse_slice(args, ex_dt)

    .. Warning:
        Following arguments need to be noted for this script:
        total_folds, ts_level_nb

    """
    for fold, ex_id in ex_dt.items():
        print(f'------fold: {fold}   ex_id: {ex_id}-------')
        args.eval_id = ex_id
        args.fold = fold
        mypath = PathPos(args.eval_id)

        label_file = mypath.label_excel_fpath  # "dataset/SSc_DeepLearning/GohScores.xlsx"
        seed = 49
        all_loader = LoadPos2Score(mypath, label_file, seed, args.fold, args.total_folds, args.ts_level_nb)
        valid_dataloader = all_loader.load()

        dataloader_dict = {'valid': valid_dataloader}
        # , 'valid': valid_dataloader, 'validaug': validaug_dataloader}
        # dataloader_dict.update({'test': test_dataloader})
        for mode, loader in dataloader_dict.items():
            print(f'start save slices for {mode}')
            for batch_data in loader:  # one data, with shape (1, channel, x, y)
                # print('all paths for this data')
                print(batch_data['fpath2save'])
                print('shape of image')
                print(batch_data['image_key'].shape)
                for slice, pth in zip(batch_data['image_key'][0], batch_data['fpath2save']):  # img and path of each slice
                    full_pth = os.path.join(mypath.id_dir, 'predicted_slices', pth)
                    if not os.path.isdir(os.path.dirname(full_pth)):
                        os.makedirs(os.path.dirname(full_pth))
                    print(full_pth)
                    origin_ls = [float(i) for i in batch_data['origin_key'][1:]]
                    space_ls = [float(i) for i in batch_data['space_key'][1:]]
                    save_itk(full_pth, slice, origin_ls, space_ls)  # slice does not have origin and space along z

    print('Finish all things!')


if __name__ == "__main__":
    args = get_args()
    ex_dt = {1: 193,
             2: 194,
             3: 276,
             4: 277}

    save_corse_slice(args, ex_dt)
