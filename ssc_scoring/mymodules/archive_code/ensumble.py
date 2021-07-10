# -*- coding: utf-8 -*-
# @Time    : 3/22/21 10:10 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import glob
import os
import pandas as pd
import numpy as np
from ssc_scoring.run import round_to_5
import csv


if __name__ == "__main__":
    ids = [354,
353,
355,
362,
361
]
    columns = ['disext', 'gg', 'rept']
    mode = "test"
    df_all = []
    for id in ids:
        id_dir = "/data/jjia/ssc_scoring/models/" + str(id)
        pred_file = sorted(glob.glob(os.path.join(id_dir, "*", mode + "_batch_preds.csv")))[0]
        df_pred = pd.read_csv(pred_file, names=columns).to_numpy()
        df_all.append(df_pred)
    df_ave = (df_all[0] + df_all[1] + df_all[2] + df_all[3] + df_all[4]) / 5
    df_ave_int = df_ave.astype(int)
    df_ave_en5 = round_to_5(df_ave_int)

    id_dir = "/data/jjia/ssc_scoring/models/" + str(ids[-1])
    sub_dir = sorted(glob.glob(os.path.join(id_dir, "fold*")))[0]

    name = str(ids[0]) + '_' + str(ids[1]) + '_' + str(ids[2]) + '_' + str(ids[3]) + '_' + str(ids[4]) + 'ensemble.csv'
    file_fpath = sub_dir + '/' + name
    with open(file_fpath, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        if len(df_ave_en5.shape) == 1:  # when data.shape==(batch_size,) in classification task
            df_ave_en5 = df_ave_en5.reshape(-1, 1)
        writer.writerows(df_ave_en5)
    print('successfully save file to ', file_fpath)

    label_file = sorted(glob.glob(os.path.join(id_dir, "*", mode + "_batch_label.csv")))[0]
    labels = pd.read_csv(label_file, names=columns).to_numpy()
    abs_err = np.abs(labels - df_ave_en5)
    mae = np.mean(abs_err)
    mae_disext, mae_gg, mae_rept = np.mean(abs_err, axis=0)
    mae_fpath = file_fpath.split('.csv')[0] + "_mae.csv"
    with open(mae_fpath, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['all', 'disext', 'gg', 'rept'])
        writer.writerow([mae, mae_disext, mae_gg, mae_rept])
    print('all', 'disext', 'gg', 'rept')
    print(mae, mae_disext, mae_gg, mae_rept)
    print('successfully save file to ', mae_fpath)





