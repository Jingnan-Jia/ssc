# merge the results from 4 folds
import numpy as np
import pandas as pd
import os
from mymodules.confusion_test import confusion
import myutil.myutil as futil

# pred_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/LKT2_16patients.csv"
#
# label_1 = "/data/jjia/ssc_scoring/ground_truth_17_patients.csv" ground_truth_16patients

# pred_1 = "/data/jjia/ssc_scoring/1405_16pats_pred.csv"
#
# label_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/ground_truth_16patients.csv"

# run_pos
run_pos = 0

if run_pos:
    from mymodules.path import PathPos as Path
    label_postfix = "_label"
    pred_postfix = "_pred"
    # ex_ls = [51, 52, 275, 274]
    # ex_ls = [133, 130, 132, 131]
    # ex_ls = [155, 156, 157, 158]
    ex_ls = [193, 194, 276, 277]
    # ex_ls = [279, 280, 205, 278]
    # ex_ls = [286, 285, 283, 284]

else:
    from mymodules.path import PathScore as Path
    label_postfix = "_label"
    pred_postfix = "_pred_int_end5"
    ex_ls = [1585, 1586, 1587, 1588]

    # score prediction
    # ex_ls = [1405, 1404, 1411, 1410]
    # ex_ls = [1205, 1202, 1203, 1204]
    # ex_ls = [1136, 1132, 1135, 1134]
    # ex_ls = [1127, 1128, 1129, 1130]

    # ex_ls = [1405, 1404, 1411, 1410]
    # ex_ls = [1123, 1118, 1124, 1120]
    # ex_ls = [1481, 1114, 1115, 1116]


label_all_dir = Path().model_dir + '/' + '_'.join([str(i) for i in ex_ls])

# Collect and re-save.
for mode in ['train', 'validaug', 'valid', 'test']:
    if not os.path.isdir(label_all_dir):
        os.makedirs(label_all_dir)
    label_all_path =  label_all_dir+ '/' + mode + label_postfix + ".csv"
    pred_all_path =  label_all_dir+ '/' + mode + pred_postfix + ".csv"
    #
    df_label_all = pd.DataFrame()
    df_pred_all = pd.DataFrame()
    for ex_id in ex_ls:
        if run_pos:
            pred_1 = Path(id=ex_id).pred(mode)
        else:
            pred_1 = Path(id=ex_id).pred_end5(mode)
        label_1 = Path(id=ex_id).label(mode)

        if os.path.isfile(pred_1):
            df_label = pd.read_csv(label_1)
            df_pred = pd.read_csv(pred_1)
            print('read', label_1, ' sucessfully !')

            df_label_all = pd.concat([df_label_all, df_label])
            df_pred_all = pd.concat([df_pred_all, df_pred])
    df_label_all.to_csv(label_all_path, index=False)
    df_pred_all.to_csv(pred_all_path, index=False)
    print('finish')


for mode in [ 'train', 'validaug', 'valid', 'test']:
    label_all_path =  label_all_dir+ '/' + mode + label_postfix + ".csv"
    pred_all_path =  label_all_dir+ '/' + mode + pred_postfix + ".csv"
    pred_1 = pred_all_path
    label_1 = label_all_path
    if os.path.isfile(pred_1):
        if run_pos:
            try:
                df_label = pd.read_csv(label_1)
                df_pred = pd.read_csv(pred_1)
                print('read ', label_1, 'successfully !')

                for df in [df_label, df_pred]:
                    if df.columns[0] == "ID":
                        del df["ID"]
                        del df["Level"]

                if df_label.columns[0] not in ['L1_pos', 'L1', 'disext']:
                    df_label = pd.read_csv(label_1, header=None)
                    if len(df_label.columns) == 5:
                        columns = ['L1', 'L2', 'L3', 'L4', 'L5']
                    elif len(df_label.columns) == 3:
                        columns = ['disext', 'gg', 'retp']
                    else:
                        columns = ['unknown']
                    df_label.columns = columns

                if df_pred.columns[0] not in ['L1_pos', 'L1', 'disext']:
                    df_pred = pd.read_csv(pred_1, header=None)
                    if len(df_pred.columns) == 5:
                        columns = ['L1', 'L2', 'L3', 'L4', 'L5']
                    elif len(df_pred.columns) == 3:
                        columns = ['disext', 'gg', 'retp']
                    else:
                        columns = ['unknown']
                    df_pred.columns = columns

                label_np = df_label.to_numpy()

                pred_np = df_pred.to_numpy()
                diff = pred_np - label_np

                mean = np.mean(diff)
                std = np.std(diff)

                bland_in_1_mean_std = {"mean": mean, "std": std}
                adap_markersize = 0
            except FileNotFoundError:
                print('Cannot find this file, pass it', mode)
                continue


        else:

            bland_in_1_mean_std = None
            adap_markersize = 1
        confusion(label_1, pred_1, bland_in_1_mean_std=bland_in_1_mean_std, adap_markersize=adap_markersize)
        icc = futil.icc(label_1, pred_1)
        print('icc: ', icc)

    print("finish")
