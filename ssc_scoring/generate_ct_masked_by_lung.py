# -*- coding: utf-8 -*-
# @Time    : 4/9/21 8:00 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# generate ct images masked by lung and save them to the same directory as the original image.

import glob
import os

import pandas as pd
from medutils.medutils import load_itk, save_itk

from ssc_scoring.mymodules.path import PathPos as Path

def main():
    mypath = Path()

    abs_dir_path = os.path.dirname(os.path.realpath(__file__))
    ct_dir = mypath.dataset_dir(resample_z=0)  # "/dataset/SSc_DeepLearning"

    ct_fpath = sorted(glob.glob(ct_dir + '/*/' + 'CTimage.mha'))
    lu_fpath = sorted(glob.glob(ct_dir + '/*/' + 'CTimage_lung.mha'))

    excel = mypath.label_excel_fpath
    label_excel = pd.read_excel(excel, engine='openpyxl')
    pos: pd.DataFrame = pd.DataFrame(label_excel, columns=['PatID', 'L1_pos', 'L2_pos', 'L3_pos', 'L4_pos', 'L5_pos'])

    assert len(ct_fpath) == len(lu_fpath) == len(pos)

    for pos, ct_f, lu_f in zip(pos.iterrows(), ct_fpath, lu_fpath):
        if ('108' in ct_f or '226' in ct_f or '247' in ct_f):

        # if 'Pat_135' not in ct_f:
            #     continue
            print('start process ...', ct_f)
            index, po = pos
            ct_f: str
            lu_f: str
            po: pd.Series

            ct, ori, sp = load_itk(ct_f, require_ori_sp=True)
            lu, ori, sp = load_itk(lu_f, require_ori_sp=True)
            # edge_value = ct[0, 0, 0]
            # ct[lu==0] = edge_value

            # select specific slices
            # slice_index_middle = []
            # for position in po.to_list()[1:]:
            #     slice_index_middle.append(int((position - ori[0]) / sp[0]))
            slice_index_middle = [int((position - ori[0]) / sp[0]) for position in po.to_list()[1:]]
            slice_index_up = [i - 1 for i in slice_index_middle]
            slice_index_down = [i + 1 for i in slice_index_middle]
            for up, middle, down, lv in zip(slice_index_up, slice_index_middle, slice_index_down, [1, 2, 3, 4, 5]):
                # save_itk(os.path.join(os.path.dirname(lu_f), "Level" + str(lv) + "_up_lung_mask.mha"), lu[up], ori, sp)
                # save_itk(os.path.join(os.path.dirname(lu_f), "Level" + str(lv) + "_middle_lung_mask.mha"), lu[middle], ori, sp)
                # save_itk(os.path.join(os.path.dirname(lu_f), "Level" + str(lv) + "_down_lung_mask.mha"), lu[down], ori, sp)

                save_itk(os.path.join(os.path.dirname(lu_f), "Level" + str(lv) + "_up_MaskedByLung.mha"), lu[up] * ct[up] - (1-lu[up])*2048, ori, sp)
                save_itk(os.path.join(os.path.dirname(lu_f), "Level" + str(lv) + "_middle_MaskedByLung.mha"), lu[middle] * ct[middle] - (1-lu[middle])*2048, ori, sp)
                save_itk(os.path.join(os.path.dirname(lu_f), "Level" + str(lv) + "_down_MaskedByLung.mha"), lu[down] * ct[down] - (1-lu[down])*2048, ori, sp)

            print(lu_f)


if __name__ == '__main__':
    main()