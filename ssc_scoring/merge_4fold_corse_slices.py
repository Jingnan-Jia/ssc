# -*- coding: utf-8 -*-
# @Time    : 7/13/21 11:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import glob
import os
import shutil

from ssc_scoring.mymodules.path import PathPos


def merge_corse_slices(ex_ls) -> None:
    """Merge/copy the predicted slices from 4 folds into the same directory.

    :param ex_ls:
    :return:

    Example:

        >>> ex_ls = [193, 194, 276, 277]
        >>> merge_corse_slices(ex_ls)

    """
    all_dir = '/data/jjia/ssc_scoring/ssc_scoring/results/models_pos/' + '_'.join([str(i) for i in ex_ls]) + '/predicted_slices'
    if not os.path.isdir(all_dir):
        os.makedirs(all_dir)

    for ex_id in ex_ls:
        print(f'copy validation files from {ex_id} to {all_dir} ...')
        source_dir = PathPos(id=ex_id).id_dir + '/predicted_slices'
        file_names = os.listdir(source_dir)
        for file_name in file_names:
            print(f'copy {file_name}')
            shutil.copytree(os.path.join(source_dir, file_name), os.path.join(all_dir,file_name))

    nb_pats = 0
    for folder in os.listdir(all_dir):
        nb_pats += 1

    print(f'there are already {nb_pats} patients in {all_dir}')

    source_dir = PathPos().ori_data_dir
    # source_dir = '/data/jjia/ssc_scoring/ssc_scoring/dataset/SSc_DeepLearning'
    print(source_dir)
    dir_names = os.listdir(source_dir)
    dir_names = [i for i in dir_names if 'Pat' in i]
    print(len(dir_names))
    for dir_name in dir_names:
        src_files = glob.glob(os.path.join(source_dir, dir_name, 'Level*_up.mha'))
        src_files.extend(glob.glob(os.path.join(source_dir, dir_name, 'Level*_middle.mha')))
        src_files.extend(glob.glob(os.path.join(source_dir, dir_name, 'Level*_down.mha')))

        tgt_dir = os.path.join(all_dir,dir_name)
        print(tgt_dir)
        if not os.path.isdir(tgt_dir):
            os.makedirs(tgt_dir)

        for file in src_files:
            print(file)
            tgt_file = os.path.join(tgt_dir, os.path.basename(file))
            # print(tgt_file)
            if not os.path.isfile(tgt_file):
                print('to: ', tgt_file)
                shutil.copy(file, tgt_file )


if __name__ == "__main__":
    ex_ls = [193, 194, 276, 277]

    merge_corse_slices(ex_ls)
