# merge the results from 4 folds
import sys
sys.path.append("..")

from typing import Sequence

import numpy as np
import pandas as pd
import os
from ssc_scoring.mymodules.confusion_test import confusion
from medutils.medutils import icc
from ssc_scoring.compute_metrics import metrics


# pred_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/LKT2_16patients.csv"
#
# label_1 = "/data/jjia/ssc_scoring/ground_truth_17_patients.csv" ground_truth_16patients

# pred_1 = "/data/jjia/ssc_scoring/1405_16pats_pred.csv"
#
# label_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/ground_truth_16patients.csv"

# run_pos
def merge(run_pos: bool, ex_ls: Sequence) -> None:
    """Merge the validation results from 4 folds (4 experiments) and evaluate them totally. The obtained
    evaluation metrics is **not equal** to the average of 4-fold validation metrics.

    :param run_pos: If the project name is run_pos.
    :param ex_ls: experiments list ordered by fold number 1,2,3,4.
    :return: None. The merged results will be saved to disk.

    Examples:

        >>> run_pos = True
        >>> ex_ls = [193, 194, 276, 277]
        >>> main(run_pos, ex_ls)

    """
    if run_pos:
        from ssc_scoring.mymodules.path import PathPos as Path
        label_postfix = "_label"
        pred_postfix = "_pred"
        # ex_ls = [51, 52, 275, 274]
        # ex_ls = [133, 130, 132, 131]
        # ex_ls = [155, 156, 157, 158]
         # [193, 194, 276, 277]
        # ex_ls = [279, 280, 205, 278]
        # ex_ls = [286, 285, 283, 284]
        bland_in_1 = True
        adap_markersize = False

    else:
        from ssc_scoring.mymodules.path import PathScore as Path
        label_postfix = "_label"
        pred_postfix = "_pred_int_end5"
        # ex_ls = [1585, 1586, 1587, 1588]

        # score prediction
        # ex_ls = [1405, 1404, 1411, 1410]
        # ex_ls = [1205, 1202, 1203, 1204]
        # ex_ls = [1136, 1132, 1135, 1134]
        # ex_ls = [1127, 1128, 1129, 1130]

        # ex_ls = [1405, 1404, 1411, 1410]
        # ex_ls = [1123, 1118, 1124, 1120]
        # ex_ls = [1481, 1114, 1115, 1116]
        bland_in_1 = False
        adap_markersize = True


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
        print('finish merge 4 fold results')


    # Evaluate the merged results
    for mode in [ 'train', 'validaug', 'valid', 'test']:
        label_all_path =  label_all_dir+ '/' + mode + label_postfix + ".csv"
        pred_all_path =  label_all_dir+ '/' + mode + pred_postfix + ".csv"

        metrics(pred_all_path, label_all_path, bland_in_1, adap_markersize)
        icc_value = icc(label_all_path, pred_all_path)
        print('icc: ', icc_value)


if __name__ == "__main__":
    run_pos = True
    ex_ls = [193, 194, 276, 277]

    merge(run_pos, ex_ls)
