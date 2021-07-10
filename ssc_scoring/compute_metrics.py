
import numpy as np
import pandas as pd

from mymodules.confusion_test import confusion


pred_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/LKT2_16patients.csv"

# pred_1 = "/data/jjia/ssc_scoring/1405_16pats_pred.csv"
#
label_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/ground_truth_16patients.csv"


for mode in ['train', 'validaug', 'valid', 'test']:
    df_label = pd.read_csv(label_1)
    df_pred = pd.read_csv(pred_1)

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

    bland_in_1_mean_std = {"mean": mean, "std":std}
    confusion(label_1, pred_1, bland_in_1_mean_std=None, adap_markersize=1)


print("finish")
    # except:
    #     print('error')
    #     pass