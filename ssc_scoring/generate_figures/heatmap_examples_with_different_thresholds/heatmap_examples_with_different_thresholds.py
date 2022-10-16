import medpy
import medpy.metric
import numpy as np
import SimpleITK as sitk
import time
import copy
import matplotlib.pyplot as plt
import glob
import cv2
from ssc_scoring.mymodules.colormap import get_continuous_cmap
import os
# This file is used to search/explore the best threshold to let the heat map respresent the Goh Score.

def apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):

    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)

parent_folder = "/home/jjia/data/ssc_scoring/ssc_scoring/results/models/1903/test_data_occlusion_maps_occ_by_healthy"
lung_fpath_ls = sorted(glob.glob(f"{parent_folder}/Pat_*/Level*/lung_mask.npy"))


fig = plt.figure(figsize=(12, 4))
fig_abs = plt.figure(figsize=(12, 4))
fig_abs_all = plt.figure(figsize=(4, 4))
fig_abs_sum = plt.figure(figsize=(4, 4))
fig_merge = plt.figure(figsize=(4, 4))

title_ls = ['TOT', 'GG', 'RET']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
y_max, y_max_abs = 0, 0
thresholds = np.arange(0, -3, -0.3)
abs_sum_ls = []
for fig_idx, pattern in enumerate(['disext', 'gg', 'rept']):
    ax = fig.add_subplot(1, 3, fig_idx + 1)
    ax_abs = fig_abs.add_subplot(1, 3, fig_idx + 1)
    ax_abs_all = fig_abs_all.add_subplot(1, 1, 1)
    ax_merge = fig_merge.add_subplot(1, 1, 1)

    map_fpath_ls = sorted(glob.glob(f"{parent_folder}/Pat_*/Level*/{pattern}_ori_label_*_pred_*_mae_diff.npy"))
    ori_img_ls =  sorted(glob.glob(f"{parent_folder}/Pat_*/Level*/ori_img.jpg"))
    print(len(map_fpath_ls),len(lung_fpath_ls), len(ori_img_ls) )
    assert len(map_fpath_ls) == len(lung_fpath_ls) == len(ori_img_ls)
    diff_ls = [[], [], [], [], [], [], [], [], [], [],
               [], [], [], [], [], [], [], [], [], [],
               [], [], [], [], [], [], [], [], [], []]  # 10 points
    for map_fpath, lung_fpath, ori_img_fpath in zip(map_fpath_ls, lung_fpath_ls, ori_img_ls):
        pat_level_1 = map_fpath.split('occlusion_maps_occ_by_healthy')[-1][:16]
        pat_level_2 = lung_fpath.split('occlusion_maps_occ_by_healthy')[-1][:16]
        assert  pat_level_1 == pat_level_2
        pred_score = float(map_fpath.split('pred_')[-1].split('_mae_diff.npy')[0])
        if pred_score < 40:  # higher socre can better show the effect of thresholds
            continue
        if 'Pat_035' not in ori_img_fpath:  # only perform this image to save time
            continue
        if 'Level3' not in ori_img_fpath:  # only perform this image to save time
            continue

        if pred_score <0:
            pred_score = 0
        if pred_score > 100:
            pred_score = 100

        lung_mask = np.load(lung_fpath)
        map = np.load(map_fpath)
        ori_img = cv2.imread(ori_img_fpath)[:, :,0] # only use one channel, because the other two channels are the same

        for idx, THRESHOLD in enumerate(thresholds):
            map_ = copy.deepcopy(map)
            map_ = np.where(map_ >= THRESHOLD, 0, 1)
            # plt.figure()
            plt.imshow(map_)
            plt.axis('off')

            # map_[map_ < THRESHOLD] = 1
            # map_[map_ >= THRESHOLD] = 0
            area_highted = np.sum(map_)
            area_lung = np.sum(lung_mask)
            ratio = area_highted / area_lung * 100
            diff = ratio - pred_score
            diff_ls[idx].append(diff)
            row = f"{pattern} threshold: {THRESHOLD} ratio: {ratio: .2f}, pred: {pred_score: .2f}, diff: {diff: .2f}"
            print(row)
            plt.savefig(f"Pat_{map_fpath.split('Pat_')[-1].replace('/', '_').replace('.npy', '')}_threshold_{THRESHOLD: .2f}_ratio_{ratio: .2f}diff_{diff: .2f}.png", bbox_inches = "tight", pad_inches=0)

            # cmp = get_continuous_cmap(hex_list = ['E1D015'])
            # # plt.get_cmap('twilight')
            # hm = apply_custom_colormap(np.uint8(map_), cmap=cmp)
            # print(f'before, map_hm: {hm}')
            map_ *= 200  # rescale [0,1] to [0, 256] to show color
            map_ = np.uint8(map_)
            hm = cv2.applyColorMap(map_, cv2.COLORMAP_COOL)  # use different cmap to differenciate original heat map
            # hm[map==0] = np.array([0,0,0])
            # hm = cv2.applyColorMap(np.uint8(map_mae), cv2.COLORMAP_JET)
            # print(f'after, map_hm: {hm}')
            w, h = map_.shape
            map_mask = copy.deepcopy(map_).reshape(w, h, 1)
            map_mask[map_!=0] = 1
            img_with_map_mae_dif = 0.2 * hm + 0.8 * ori_img.reshape(w, h, 1)
            temp1 = img_with_map_mae_dif * map_mask
            temp2 = ori_img.reshape(w, h, 1) * (1-map_mask)
            saved_map = temp1 + temp2
            cv2.imwrite(f"Pat_{map_fpath.split('Pat_')[-1].replace('/', '_').replace('.npy', '')}_threshold_{THRESHOLD: .2f}_ratio_{ratio: .2f}diff_{diff: .2f}_.png", saved_map)
