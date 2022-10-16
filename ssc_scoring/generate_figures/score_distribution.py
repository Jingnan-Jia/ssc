"""
This script is to generate the score distribution for SSc patients.

1. read the original label excel file.
2. Split the dataset to training/validation/testing datasets.
3. Count the number of patients for each score in three groups.
4. Bar plot with the values on the bar.

The csv file includes the following contents (The first column is the score, the other columns is the number of patients
in the corresponding score):

Label,TOT,GG,RET
0,560,690,615
5,120,102,151
10,92,110,98
15,46,34,38
20,52,66,61
25,44,24,33
30,44,42,54
35,30,19,11
40,40,25,29
45,14,5,4
50,36,18,23
55,15,2,8
60,15,2,9
65,9,0,2
70,10,2,5
75,4,2,5
80,5,2,2
85,1,1,0
90,8,7,0
95,0,0,0
100,5,3,2

"""
import os
import sys
sys.path.append("../..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from ssc_scoring.mymodules.path import PathScore as Path
from ssc_scoring.mymodules.mydata import LoadScore
from ssc_scoring.mymodules.set_args import get_args
from sklearn.model_selection import KFold
import glob
from collections import OrderedDict
import copy

def split_dir_pats(mypath, ts_id, label_excel):
    data_dir = mypath.ori_data_dir
    dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*")))
    # 3 labels for one level
    # pats_id_in_excel = pd.DataFrame(label_excel, columns=['PatID']).values
    # pats_id_in_excel = [i[0] for i in pats_id_in_excel]
    pats_id_in_excel = label_excel.index.to_list()
    print(f"len(dir): {len(dir_pats)}, len(pats_in_excel): {len(pats_id_in_excel)} ")
    print("======================")
    assert len(dir_pats) == len(pats_id_in_excel)

    # assert the names of patients got from 2 ways
    pats_id_in_dir = [int(path.split('Pat_')[-1][:3]) for path in dir_pats]
    assert pats_id_in_dir == pats_id_in_excel

    ts_dir, tr_vd_dir = [], []
    for id, dir_pt in zip(pats_id_in_dir, dir_pats):
        if id in ts_id:
            ts_dir.append(dir_pt)
        else:
            tr_vd_dir.append(dir_pt)
    return np.array(tr_vd_dir), np.array(ts_dir)

mypath = Path('0')
label_file = mypath.label_excel_fpath  # "dataset/SSc_DeepLearning/GohScores.xlsx"  # labels are from here
label_excel = pd.read_excel(label_file, engine='openpyxl')
label_excel.set_index('PatID', inplace=True, verify_integrity=True)
seed = 49  # for split of  cross-validation
fold = 1
total_folds = 4
args = get_args()
ts_id = [7, 9, 11, 12, 16, 19, 20, 21, 26, 28, 35, 36, 37, 40, 46, 47, 49, 57,
                          58, 59, 60, 62, 66, 68, 70, 77, 83, 116, 117, 118, 128, 134, 137, 140, 144,
                          149, 158, 170, 179, 187, 189, 196, 203, 206, 209, 210, 216, 227, 238, 263]
kf = KFold(n_splits=total_folds, shuffle=True, random_state=seed)  # for future reproduction

tr_vd_pt, ts_pt = split_dir_pats(mypath, ts_id, label_excel)  # numpy array, shape (N,) which store the full path of patient directory

kf_list = list(kf.split(tr_vd_pt))
tr_pt_idx, vd_pt_idx = kf_list[fold - 1]
tr_pt = tr_vd_pt[tr_pt_idx]
vd_pt = tr_vd_pt[vd_pt_idx]

train_dt, valid_dt, test_dt = OrderedDict(), OrderedDict(), OrderedDict()
for dt, dir_pats in zip([train_dt, valid_dt, test_dt], [tr_pt, vd_pt, ts_pt]):
    for dir_pat in dir_pats:
        idx = int(dir_pat.split('/')[-1].split('Pat_')[-1])
        for level in [1, 2, 3, 4, 5]:
            y_disext = label_excel.at[idx, 'L' + str(level) + '_disext']
            y_gg = label_excel.at[idx, 'L' + str(level) + '_gg']
            y_ret = label_excel.at[idx, 'L' + str(level) + '_retp']
            dt[f"patient_{idx}_level_{level}"] = np.array([y_disext, y_gg, y_ret])
# Convert dict to dataframe with dict keys as the index.
train_df, valid_df, test_df = map(lambda tmp_dt: pd.DataFrame.from_dict(tmp_dt, orient='index'), [train_dt, valid_dt, test_dt])


def count_df(df):
    out = pd.DataFrame(index=np.arange(0, 101, 5))

    scores = df.to_numpy()
    for pattern_idx, pattern_name in enumerate(['TOT', 'GG', 'RET']):
        score_1patten = scores[:, pattern_idx]
        values, counts = np.unique(score_1patten, return_counts=True)
        values, counts = map(list, [values, counts])
        for i in np.arange(0, 101, 5):
            if i not in values:
                values.append(i)
                counts.append(0)
        values, counts  = zip(*sorted(zip(values, counts)))
        out[pattern_name] = counts
    return out

for mode, df in zip(['training', 'validation', 'testing'], [train_df, valid_df, test_df]):
    data = count_df(df)
    max_count = data.max().max()  # the the max of each pattern, then get their max
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig = plt.figure(figsize=(15, 3), dpi=600)
    plt.subplots_adjust(left=0, bottom=None, right=1, top=None, wspace=0, hspace=0)

    for plot_id, column in enumerate(data.columns):
        ax = fig.add_subplot(1, 3, plot_id + 1)
        x_ = data.index.to_numpy().reshape(-1, )
        label = data[column].to_numpy().reshape(-1, )
        scatter_kwds = {'facecolor': colors[plot_id], 'width':4}
        ax.bar(x_, label, **scatter_kwds)
        for i, v in zip(x_, label):
            ax.text(i, v+20, str(v), ha='center', color='black')

        ax.set_xlabel("Score (%)", fontsize=15)
        ax.set_xticks([i * 5 for i in range(21)])

        ax.set_xticklabels([i * 5 for i in range(21)])
        y_upper_limit = (max_count//100 + 1) * 100
        ax.set_ylim(0, y_upper_limit )
        ax.set_xlim(-4, 104)

        ax.text(0.5, 0.8,  column, transform=ax.transAxes,ha='center', fontsize=15,color='black')
        # ax.set_title(column, fontsize=15, pad=100)
        if plot_id == 0:
            ax.set_ylabel(f"Count ({mode})", fontsize=15)
        #     for i in range(21):
        #         plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
        else:
            ax.axes.yaxis.set_visible(False)
    fig.tight_layout()
    #         ax.set_axis_off()

    plt.savefig(f"{mode}_score_distribution_fold{fold}.png", bbox_inches = "tight")