# -*- coding: utf-8 -*-
# @Time    : 4/9/21 8:00 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import glob
import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
from jjnutils.util import get_all_ct_names, load_itk, save_itk
import matplotlib.pyplot as plt


abs_dir_path = os.path.dirname(os.path.realpath(__file__))
ct_dir = abs_dir_path + "/dataset/SSc_DeepLearning"

ct_fpath = sorted(glob.glob(ct_dir + '/*/' + 'CTimage.mha'))
lu_fpath = sorted(glob.glob(ct_dir + '/*/' + 'CTimage_lung.mha'))

excel = "dataset/SSc_DeepLearning/GohScores.xlsx"
label_excel = pd.read_excel(excel, engine='openpyxl')
pos: pd.DataFrame = pd.DataFrame(label_excel, columns=['PatID', 'L1_pos', 'L2_pos', 'L3_pos', 'L4_pos', 'L5_pos'])

assert len(ct_fpath) == len(lu_fpath) == len(pos)

for pos, ct_f, lu_f in zip(pos.iterrows(), ct_fpath, lu_fpath):
    index, po = pos
    ct_f: str
    lu_f: str
    po: pd.Series

    ct, ori, sp = load_itk(ct_f, require_ori_sp=True)
    lu, _, __ = load_itk(lu_f, require_ori_sp=True)
    edge_value = ct[0, 0, 0]
    ct[lu==0] = edge_value

    # select specific slices
    # slice_index_middle = []
    # for position in po.to_list()[1:]:
    #     slice_index_middle.append(int((position - ori[0]) / sp[0]))
    slice_index_middle = [int((position - ori[0]) / sp[0]) for position in po.to_list()[1:]]
    slice_index_up = [i - 1 for i in slice_index_middle]
    slice_index_down = [i + 1 for i in slice_index_middle]
    for up, middle, down, lv in zip(slice_index_up, slice_index_middle, slice_index_down, [1,2,3,4,5]):
        save_itk(os.path.dirname(lu_f) + "/" + "Level" + str(lv) + "_up_MaskedByLung.mha", ct[up], ori, sp)
        save_itk(os.path.dirname(lu_f) + "/" + "Level" + str(lv) + "_middle_MaskedByLung.mha", ct[middle], ori, sp)
        save_itk(os.path.dirname(lu_f) + "/" + "Level" + str(lv) + "_down_MaskedByLung.mha", ct[down], ori, sp)
    print(lu_f)
