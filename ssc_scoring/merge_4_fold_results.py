
import numpy as np
import pandas as pd
import os
from ssc_scoring.confusion_test import confusion

# pred_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/LKT2_16patients.csv"
#
# label_1 = "/data/jjia/ssc_scoring/ground_truth_17_patients.csv" ground_truth_16patients

# pred_1 = "/data/jjia/ssc_scoring/1405_16pats_pred.csv"
#
# label_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/ground_truth_16patients.csv"

# run_pos
run_pos = 1
if run_pos:
    models_name = "models_pos"
    label_postfix = "_label"
    pred_postfix = "_pred"
    # ex_ls = [51, 52, 275, 274]
    # ex_ls = [133, 130, 132, 131]
    # ex_ls = [155, 156, 157, 158]
    ex_ls = [193, 194, 276, 277]
    # ex_ls = [279, 280, 205, 278]
    # ex_ls = [286, 285, 283, 284]
else:
    models_name = "models"
    label_postfix = "_label"
    pred_postfix = "_pred_int_end5"

    # score prediction
    # ex_ls = [1405, 1404, 1411, 1410]
    # ex_ls = [1205, 1202, 1203, 1204]
    # ex_ls = [1136, 1132, 1135, 1134]
    # ex_ls = [1127, 1128, 1129, 1130]

    ex_ls = [1405, 1404, 1411, 1410]
    # ex_ls = [1123, 1118, 1124, 1120]
    # ex_ls = [1481, 1114, 1115, 1116]



label_all_dir = "/data/jjia/ssc_scoring/" + models_name + '/' + '_'.join([str(i) for i in ex_ls])


# for mode in ['train', 'validaug', 'valid', 'test']:
#     if not os.path.isdir(label_all_dir):
#         os.makedirs(label_all_dir)
#     label_all_path =  label_all_dir+ '/' + mode + label_postfix + ".csv"
#     pred_all_path =  label_all_dir+ '/' + mode + pred_postfix + ".csv"
#     #
#     df_label_all = pd.DataFrame()
#     df_pred_all = pd.DataFrame()
#     for ex_id in ex_ls:
#         pred_1 = "/data/jjia/ssc_scoring/" + models_name + '/' + str(ex_id) + '/' + mode + pred_postfix + ".csv"
#         label_1 = "/data/jjia/ssc_scoring/" + models_name + '/' + str(ex_id) + '/' + mode + label_postfix + ".csv"
#
#         if os.path.isfile(pred_1):
#             df_label = pd.read_csv(label_1)
#             df_pred = pd.read_csv(pred_1)
#             print('read', label_1, ' sucessfully !')
#
#             df_label_all = pd.concat([df_label_all, df_label])
#             df_pred_all = pd.concat([df_pred_all, df_pred])
#     if os.path.isfile(pred_1):
#         df_label_all.to_csv(label_all_path, index=False)
#         df_pred_all.to_csv(pred_all_path, index=False)
#     print('finish')



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
            except:
                print('pass ', mode)
                pass

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
        else:

            bland_in_1_mean_std = None
            adap_markersize = 1
        confusion(label_1, pred_1, bland_in_1_mean_std=bland_in_1_mean_std, adap_markersize=adap_markersize)


    print("finish")
