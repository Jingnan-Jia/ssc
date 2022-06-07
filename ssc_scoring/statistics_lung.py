# -*- coding: utf-8 -*-
# @Time    : 4/9/21 8:00 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# get the width and length of patch which can cover the whole lung.
# (This is used to extract to crop the whole lung in the future)
import glob
import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
from medutils.medutils import load_itk
import matplotlib.pyplot as plt
import csv


def clip(x_np, min, max):
    """Pixel value truncation"""
    x_np[x_np > max] = max
    x_np[x_np < min] = min
    return x_np


def bbox2(img):
    """Return the box boundary for valuable voxels."""
    min = np.min(img)
    rows = np.max(img, axis=1)
    cols = np.max(img, axis=0)
    rmin, rmax = np.where(rows>min)[0][[0, -1]]
    cmin, cmax = np.where(cols>min)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def main():
    abs_dir_path = os.path.dirname(os.path.realpath(__file__))
    ct_dir = abs_dir_path + "/dataset/SSc_DeepLearning"

    ct_fpath = sorted(glob.glob(ct_dir + '/*/*' + 'MaskedByLung.mha'))

    rmin_ls, rmax_ls, cmin_ls, cmax_ls = [], [], [], []
    err_ls = []
    for i, name in enumerate(ct_fpath):
        # if i == 422:
        # print(i)
        ct_masked, ori, sp = load_itk(name, require_ori_sp=True)
        # ct_masked =
        ct_masked = clip(ct_masked, -1500, 1500)
        try:
            rmin, rmax, cmin, cmax= bbox2(ct_masked)
            for ls, bb in zip([rmin_ls, rmax_ls, cmin_ls, cmax_ls], [rmin, rmax, cmin, cmax]):
                ls.append(bb)
                print(i, rmin, rmax, cmin, cmax, name)
        except:
            print(i, 'Error', name)
            err_ls.append(name)


    with open('box_clip_1500.csv', 'a') as f:
        writer = csv.writer(f)
        for a,b,c,d, f in zip(rmin_ls, rmax_ls, cmin_ls, cmax_ls, ct_fpath):
            writer.writerow([a,b,c,d, f])

    print(err_ls)
    print('finish')


if __name__ == "__main__":
    main()

