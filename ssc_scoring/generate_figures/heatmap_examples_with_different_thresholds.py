import medpy
import medpy.metric
import numpy as np
import SimpleITK as sitk
import time
import copy
import matplotlib.pyplot as plt
import glob

# This file is used to search/explore the best threshold to let the heat map respresent the Goh Score.

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
thresholds = np.arange(0, -3, -0.1)
abs_sum_ls = []
for fig_idx, pattern in enumerate(['disext', 'gg', 'rept']):
    ax = fig.add_subplot(1, 3, fig_idx + 1)
    ax_abs = fig_abs.add_subplot(1, 3, fig_idx + 1)
    ax_abs_all = fig_abs_all.add_subplot(1, 1, 1)
    ax_merge = fig_merge.add_subplot(1, 1, 1)

    map_fpath_ls = sorted(glob.glob(f"{parent_folder}/Pat_*/Level*/{pattern}_ori_label_*_pred_*_mae_diff.npy"))
    print(len(map_fpath_ls),len(lung_fpath_ls) )
    assert len(map_fpath_ls) == len(lung_fpath_ls)
    diff_ls = [[], [], [], [], [], [], [], [], [], [],
               [], [], [], [], [], [], [], [], [], [],
               [], [], [], [], [], [], [], [], [], []]  # 10 points
    for map_fpath, lung_fpath in zip(map_fpath_ls, lung_fpath_ls):
        pat_level_1 = map_fpath.split('occlusion_maps_occ_by_healthy')[-1][:16]
        pat_level_2 = lung_fpath.split('occlusion_maps_occ_by_healthy')[-1][:16]
        assert  pat_level_1 == pat_level_2
        pred_score = float(map_fpath.split('pred_')[-1].split('_mae_diff.npy')[0])
        if pred_score <0:
            pred_score = 0
        if pred_score > 100:
            pred_score = 100

        lung_mask = np.load(lung_fpath)
        map = np.load(map_fpath)

        for idx, THRESHOLD in enumerate(thresholds):
            map_ = copy.deepcopy(map)
            map_ = np.where(map_ >= THRESHOLD, 0, 1)
            # map_[map_ < THRESHOLD] = 1
            # map_[map_ >= THRESHOLD] = 0
            area_highted = np.sum(map_)
            area_lung = np.sum(lung_mask)
            ratio = area_highted / area_lung * 100
            diff = ratio - pred_score
            diff_ls[idx].append(diff)
            row = f"{pattern} ratio: {ratio: .2f}, pred: {pred_score: .2f}, diff: {diff: .2f}"
            print(row)
    diff_np = np.array(diff_ls)  # shape: (10, 256)
    diff_abs_np = np.abs(np.array(diff_ls))
    diff_ave = np.mean(diff_np, axis=1)   # shape: (10， )
    diff_abs_ave = np.mean(diff_abs_np, axis=1)   # shape: (10， )
    abs_sum_ls.append(diff_abs_ave)
    scatter_kwds = {'c': colors[fig_idx]}
    ax.scatter(thresholds, diff_ave, **scatter_kwds)
    ax_abs.scatter(thresholds, diff_abs_ave, **scatter_kwds)

    ax_abs_all.plot(thresholds, diff_abs_ave, **scatter_kwds)

    scatter_kwds_merge = {'c': colors[fig_idx], 'linestyle': 'dashed'}
    ax_merge.plot(thresholds, diff_abs_ave, **scatter_kwds_merge)

    ax.plot([-3,0.1], [0,0], 'k--')
    ax.set_title(title_ls[fig_idx])
    ax.set_ylabel(f"highlighted ratio - Automatic scoring")
    ax.set_xlabel(f"Threshold of highlighted area")
    y_max = max(y_max, np.max(diff_ave))

    # ax_abs.plot([-2,0.1], [0,0], 'k--')
    ax_abs.set_title(title_ls[fig_idx])
    ax_abs.set_ylabel(f"Abs(highlighted ratio - Automatic scoring)")
    ax_abs.set_xlabel(f"Threshold of highlighted area")
    y_max_abs = max(y_max, np.max(diff_abs_ave))



for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_ylim([-10, y_max + 1])
    ax_abs = fig_abs.add_subplot(1, 3, i + 1)
    ax_abs.set_ylim([0, y_max_abs + 1])

ax_abs_all = fig_abs_all.add_subplot(1, 1, 1)
ax_abs_all.set_ylabel(f"MAE")
ax_abs_all.set_xlabel(f"Threshold of heat map")
ax_abs_all.set_ylim([0, y_max_abs + 1])
ax_abs_all.legend(['TOT', 'GG', 'RET'])

abs_sum_np = np.array(abs_sum_ls)
abs_sum_np = np.mean(abs_sum_np, axis=0)
ax_abs_sum = fig_abs_sum.add_subplot(1, 1, 1)

ax_abs_sum.plot(thresholds, abs_sum_np)
ax_abs_sum.set_ylabel(f"Average MAE")
ax_abs_sum.set_xlabel(f"Threshold of heat map")
# ax_abs_sum.set_title("Mean(TOT,GG,RET)")
ax_abs_sum.set_ylim([0, y_max_abs + 1])
print(abs_sum_np)

fig.show()
fig_abs.show()
fig_abs_all.show()
fig_abs_sum.show()

fig.savefig(f"heatmap_mean_error_vs_threshold.png")
fig_abs.savefig(f"heatmap_mean_abs_error_vs_threshold.png")
fig_abs_all.savefig(f"heatmap_mean_abs_error_vs_threshold_in_1.png")
fig_abs_sum.savefig(f"heatmap_mean_abs_error_vs_threshold_in_1_mean.png")

for fg in [fig, fig_abs, fig_abs_all, fig_abs_sum]:
    plt.close(fg)

scatter_kwds_merge = {'c': 'k', 'linestyle': '-'}
ax_merge.plot(thresholds, abs_sum_np, **scatter_kwds_merge )
ax_merge.set_ylabel(f"Mean absolute difference")
ax_merge.set_xlabel(f"Threshold of heat map")
ax_merge.set_ylim([0, y_max_abs + 1])
ax_merge.legend(['TOT', 'GG', 'RET', 'Average'])
fig_merge.savefig(f"heatmap_mean_abs_error_vs_threshold_in_1_all.png")

# plt.tight_layout()
# plt.show()
# plt.savefig(f'explore_threshold.png')


