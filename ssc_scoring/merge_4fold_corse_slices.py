# -*- coding: utf-8 -*-
# @Time    : 7/13/21 11:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import shutil
import os
from pathlib import Path
import glob
import shutil
from mymodules.path import PathPos
ex_ls = [193, 194, 276, 277]
all_dir = '/data/jjia/ssc_scoring/ssc_scoring/results/models_pos/' + '_'.join([str(i) for i in ex_ls]) + '/predicted_slices'
if not os.path.isdir(all_dir):
    os.makedirs(all_dir)

# for ex_id in ex_ls:
#     source_dir = PathPos(id=ex_id).id_dir + '/predicted_slices'
#     file_names = os.listdir(source_dir)
#     for file_name in file_names:
#         shutil.copy(os.path.join(source_dir, file_name), os.path.join(all_dir,file_name))


source_dir = PathPos().ori_data_dir
source_dir = '/data/jjia/ssc_scoring/ssc_scoring/dataset/SSc_DeepLearning'
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
