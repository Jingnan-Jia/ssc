# -*- coding: utf-8 -*-
# @Time    : 4/10/21 11:59 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# Extract the lungs using morphological operations and save them to same directory with the original files
from typing import Sequence
from medutils.medutils import load_itk, save_itk, get_all_ct_names

from scipy.ndimage import morphology
import numpy as np
import time
from skimage.measure import label
from ssc_scoring.mymodules.path import PathPos as Path


def largest_connected_parts(bw_img: np.ndarray, nb_need_saved=2):
    bw_img[:10] = 0  # exclude the noise at the bottom
    bw_img[-10:] = 0  # exclude the noise at the top

    t0 = time.time()
    labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)
    t1 = time.time()
    print('it cost this time to compute label: ' + str(t1 - t0))
    pixel_label_list, pixel_count_list = np.unique(labeled_img, return_counts=True)
    pixel_label_list, pixel_count_list = list(pixel_label_list), list(pixel_count_list)
    t2 = time.time()
    tt = t2 - t1
    print('it cost this time to compute pixel_count_list: ' + str(tt))

    pixel_count_list, pixel_label_list = zip(*sorted(zip(pixel_count_list, pixel_label_list), reverse=True))
    print('original connected parts number: ' + str(len(pixel_count_list)))
    pixel_count_list, pixel_label_list = pixel_count_list[:4], pixel_label_list[:4]  # exclude background

    # connect_part_list = [(labeled_img == l).astype(int) for l in pixel_label_list]
    print("candidate number: " + str(len(pixel_count_list)))

    out = np.zeros(bw_img.shape)
    nb_saved: int = 1
    for idx in range(len(pixel_count_list)):
        if nb_saved <= nb_need_saved:
            pixel_label = (labeled_img == pixel_label_list[idx]).astype(int)
            if pixel_label_list[idx] > 0 and (pixel_label[int(len(pixel_label) / 2), 0, 0] == 0):
                if (np.sum(out) == 0) or (np.sum(pixel_label) > np.sum(out) * 0.1):
                    print("nb_saved: " + str(nb_saved))
                    out += pixel_label  # to differentiate different parts.
                nb_saved += 1

    bw_img[out == 0] = 0
    print(f"all parts are found, prepare write result")
    return bw_img


def extract(scan_files: Sequence) -> None:
    """Extract the lung area of a sequence of lung CT files. Steps are:

    #. Binary threshold.
    #. Dilation + erosion
    #. Invert Color
    #. Dilation + Largest_connected_parts + erosion
    #. Save image

    :param scan_files: file full paths of CT images.

    Example:

    >>> mypath = Path()
    >>> scan_files = get_all_ct_names(mypath.dataset_dir(resample_z=0), name_suffix="CTimage")
    >>> extract(scan_files)


    """
    for scan in scan_files:

        ct, ori, sp = load_itk(scan, require_ori_sp=True)
        ct[ct > -141] = 1  # 141 is a value selected after I tried several times
        ct[ct < -141] = 0

        ct_dia = morphology.binary_dilation(ct, np.ones((3, 3, 3))).astype(int)
        ct_ero = morphology.binary_erosion(ct_dia, np.ones((3, 3, 3))).astype(int)
        ct_neg = 1 - ct_ero  # get the opposite numbers of ct

        ct_dia2 = morphology.binary_dilation(ct_neg, np.ones((6, 6, 6))).astype(int)
        ct_lung = largest_connected_parts(ct_dia2, 2)
        ct_ero2 = morphology.binary_erosion(ct_lung, np.ones((6, 6, 6))).astype(int)
        ct_ero2 = morphology.binary_erosion(ct_ero2, np.ones((3, 3, 3))).astype(int)

        save_itk(scan.split('.mha')[0] + '_lung.mha', ct_ero2, ori, sp)
        print('save lung to ', scan.split('.mha')[0] + '_lung.mha')


if __name__ == "__main__":
    mypath = Path()
    scan_files = get_all_ct_names(mypath.dataset_dir(resample_z=0), name_suffix="CTimage")
    scan_files = [file for file in scan_files if ('108' in file or '226' in file or '247' in file)]
    print(scan_files)
    extract(scan_files)