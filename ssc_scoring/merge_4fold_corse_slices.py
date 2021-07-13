# -*- coding: utf-8 -*-
# @Time    : 7/13/21 11:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import shutil
import os

from mymodules.path import PathPos
ex_ls = [193, 194, 276, 277]
all_dir = PathPos().model_dir + '/' + '_'.join([str(i) for i in ex_ls]) + '/predicted_slices'
if not os.path.isdir(all_dir):
    os.makedirs(all_dir)

for ex_id in ex_ls:
    source_dir = PathPos(id=ex_id).id_dir + '/predicted_slices'
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(all_dir,file_name))
