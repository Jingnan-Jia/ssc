# Collect the 16 patients' results from 4-folds' validation
# results and test results from 1st fold for convenience.

import csv
import os
from typing import Dict, List

import pandas as pd

from ssc_scoring.mymodules.path import PathScore as Path

def data_fpath(ex_id_dict: Dict[int, int], pat_id_dict: Dict[int, List[str]]) -> Dict[str, str]:
    """Get path of valid_pred_end5 for each experiment and the patient IDs included in the validation dataset of
    this experiment (which can be used to extract the prediction of these specific patients).

    :param ex_id_dict: key is fold number, value is experiment id.
    :param pat_id_dict: key is fold numbr, value is patient id as string type.
    :return: key is path string, , value is patient id as string type.
    """
    test_path = Path(ex_id_dict[1]).pred_end5('test')
    path_dict = {test_path: ['028', '049', '066', '070', '210', '238']}
    for fold, ex_id in ex_id_dict.items():
        valid_path = Path(ex_id).pred_end5('valid')
        path = {valid_path: pat_id_dict[fold]}
        path_dict.update(path)

    return path_dict


def collect(ex_id_dict) -> None:
    """Collect validation/testing results of 16 patients.
     10 patients are from the validation results of 4 folds from 4 different experiments, 6 patients are from testing
     dataset.

     The ID of 16 patients are the same as those scored twice by human observers. The goal of this function is
     to get a comparison between AI models and human observers.

    :param ex_id_dict: key is fold, value is experiment ID.
    :return: None. The results will be saved to disk.

    Example:

    >>> ex_id_dict = {1: 1405, 2: 1404, 3: 1411, 4: 1410}
    >>> collect(ex_id_dict)

    """

    # ex_1044_fold1: valid:179, 227, 263, test: 28, 49, 66, 70, 210, 238
    # ex_1043_fold2: valid:140,
    # ex_1045_fold3: valid:203, 209,
    # ex_1046_fold4: valid:26, 29, 77, 170


    label_all_dir = Path().model_dir + '/' + '_'.join([str(i) for _, i in ex_id_dict.items()])
    if not os.path.isdir(label_all_dir):
        os.makedirs(label_all_dir)



    pat_id_dict = {1: ['179', '227', '263'],
                   2: ['140'],
                   3: ['026', '029', '077', '170'],
                   4: ['203', '209']}
    # 10 patients are in the validation datasets of 4 folds, other 6 patients are in common testing dataset.

    pred_fpath_dt: dict = data_fpath(ex_id_dict, pat_id_dict)  # len = 5, test, 1,2,3,4 fold
    id_ls_, pred_ls_, level_ls_ = [], [], []
    for level in [1, 2, 3, 4, 5]:
        pred_dt = {}
        for pred_path, pat_list in pred_fpath_dt.items():
            if 'test' in os.path.basename(pred_path):
                data_name_path = os.path.dirname(pred_path) + '/test_data.csv'
            else:
                data_name_path = os.path.dirname(pred_path) + '/valid_data.csv'
            df = pd.read_csv(data_name_path, header=None)
            # df.insert(0, 'row_num', range(0, len(df)))  # here you insert the row count
            df_first_c = df.loc[:, 0]
            index_ls = []
            for pat_id in pat_list:
                for index, row in df_first_c.iteritems():
                    pat_name = 'Pat_' + pat_id
                    if pat_name in row:
                        if 'Level' + str(level) + "_middle" in row:
                            index_ls.append(index)
                            break
            df_pred = pd.read_csv(pred_path)
            pred_ls = []
            for idx in index_ls:
                pred = df_pred.iloc[idx].to_numpy()  # np.array shape: [3,]
                pred_ls.append(pred)
            for pat_id, pred in zip(pat_list, pred_ls):
                pred_dt.update({pat_id: pred})

        print(pred_dt)
        id_ls = []
        pred_ls = []
        level_ls = []
        for id, pred in pred_dt.items():
            pred_ls.append(pred)
            id_ls.append(int(id))
            level_ls.append(int(level))

        id_ls, pred_ls = zip(*sorted(zip(id_ls, pred_ls)))
        id_ls_.extend(id_ls)
        pred_ls_.extend(pred_ls)
        level_ls_.extend(level_ls)

    saved_path = label_all_dir + "/16pats_pred.csv"
    # saved_path = str(ex_id_dict[1]) + '_16pats_pred.csv'
    if not os.path.isfile(saved_path):  # write head
        with open(saved_path, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            head = ['ID', 'Level', 'TOT', 'GG', 'RET']
            writer.writerow(head)

    with open(saved_path, 'a') as f:
        csvwriter = csv.writer(f)
        for pred, id, level in zip(pred_ls_, id_ls_, level_ls_):
            row = [id, level, *pred]
            # row.extend(pred)
            csvwriter.writerow(row)

    # saved_id_path = label_all_dir + "/16pats_id.csv"
    # with open(saved_id_path, 'w') as f:
    #     csvwriter = csv.writer(f)
    #     for pred in id_ls_:
    #         csvwriter.writerow([pred])

    print(f"The collected results are saved at {saved_path}")


def collect_16_pats(ex_id: int) -> None:
    """Collect testing results of 16 patients. All 16 patients are in testing dataset.

       :param ex_id: experiment ID.
       :return: None. The results will be saved to disk.

       Example:

       >>> ex_id = 32
       >>> collect(ex_id)

       """

    label_all_dir = Path(ex_id).id_dir
    pat_list = ['026', '028',  '049', '066', '070', '077', '140', # '029' should be removed !
                '170', '179', '203', '209', '210', '227', '238', '263']

    pred_path = Path(ex_id).pred_end5('test')
    data_name_path = os.path.dirname(pred_path) + '/test_data.csv'
    df = pd.read_csv(data_name_path, header=None)
    df_first_c = df.loc[:, 0]
    index_ls = []
    new_pat_list = []
    level_ls_ = []
    for level in [1, 2, 3, 4, 5]:
        for pat_id in pat_list:
            for index, row in df_first_c.iteritems():
                pat_name = 'Pat_' + pat_id
                if pat_name in row:
                    if 'Level' + str(level) + "_middle" in row:
                        index_ls.append(index)
                        new_pat_list.append(pat_id)
                        level_ls_.append(int(level))
                        break

    df_pred = pd.read_csv(pred_path)

    id_ls_, pred_ls_  = [], []
    for pat_id, idx in zip(new_pat_list, index_ls):
        pred = df_pred.iloc[idx].to_numpy()  # np.array shape: [3,]
        pred_ls_.append(pred)
        id_ls_.append(int(pat_id))

        #
        # id_ls = []
        # pred_ls = []
        # level_ls = []
        # for id, pred in pred_dt.items():
        #     pred_ls.append(pred)
        #     id_ls.append(int(id))
        #     level_ls.append(int(level))

    # id_ls, pred_ls, level_ls = zip(*sorted(zip(id_ls_, pred_ls_, level_ls_)))
        # id_ls_.extend(id_ls)
        # pred_ls_.extend(pred_ls)
        # level_ls_.extend(level_ls)

    saved_path = label_all_dir + "/16pats_pred.csv"
    # saved_path = str(ex_id_dict[1]) + '_16pats_pred.csv'
    if not os.path.isfile(saved_path):  # write head
        with open(saved_path, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            head = ['ID', 'Level', 'TOT', 'GG', 'RET']
            writer.writerow(head)

    with open(saved_path, 'a') as f:
        csvwriter = csv.writer(f)
        for pred, id, level in zip(pred_ls_, id_ls_, level_ls_):
            row = [id, level, *pred]
            # row.extend(pred)
            csvwriter.writerow(row)

    # saved_id_path = label_all_dir + "/16pats_id.csv"
    # with open(saved_id_path, 'w') as f:
    #     csvwriter = csv.writer(f)
    #     for pred in id_ls_:
    #         csvwriter.writerow([pred])

    print(f"The collected results are saved at {saved_path}")


if __name__ == "__main__":
    # different fold corresponds to different experiment.
    all_16_pats_in_ts_data = True
    if all_16_pats_in_ts_data:
        ex_id = 1826
        collect_16_pats(ex_id)
    else:
        ex_id_dict = {1: 1405,
                      2: 1404,
                      3: 1411,
                      4: 1410}
        collect(ex_id_dict)
