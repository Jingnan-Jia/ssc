# -*- coding: utf-8 -*-
# @Time    : 3/19/21 11:22 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import cohen_kappa_score

import ssc_scoring.mymodules.my_bland as sm

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_acc_np(diag_np: np.ndarray, total_np: np.ndarray) -> np.ndarray:
    """Return accuracy for each category.

    Args:
        diag_np: Accurate number of each category
        total_np: Total number of each category

    Examples:
        :func:`ssc_scoring.mymodules.confusion_test.confusion`

    """
    acc_ls = []
    for diag, total in zip(diag_np, total_np):
        if diag == 0:
            acc_ls.append(0)
        else:
            acc_ls.append(diag / total)
    acc = np.array(acc_ls)
    return acc


def mae(df):
    """

    Args:
        df: A DataFrame

    Returns:
        mae, totnp

    Examples:
        :func:`ssc_scoring.mymodules.confusion_test.confusion`.

    """
    # arr = df.to_numpy()
    # labels = df.index.to_numpy().astype(int)
    preds = df.columns.to_numpy().astype(int)
    mae_ls = []
    total_ls = []
    for label, row in df.iterrows():
        row = row.to_numpy()
        abs_err = np.abs(preds - label)
        weighted_err = abs_err * row
        weighted_err = np.sum(weighted_err)
        weighted_err = weighted_err / np.sum(row)
        mae_ls.append(weighted_err)
        total_ls.append(np.sum(row))
    # mae_np = np.nanmean(, axis=axis, keepdims=keepdims)
    mae_np = np.array(mae_ls)
    total_np = np.array(total_ls)
    return mae_np, total_np


def read_check(file_fpath=None) -> pd.DataFrame:
    """Check the head of the loaded csv file. If no head, add proper head.

    Args:
        file_fpath: csv file full path

    Returns:
        Updated df

    Examples:
        :func:`ssc_scoring.mymodules.confusion_test.confusion`

    """
    df = pd.read_csv(file_fpath)
    if df.columns[0] == "ID":
        del df["ID"]
        del df["Level"]

    if df.columns[0] not in ['L1_pos', 'L1', 'L2', 'L3', 'L4', 'L5', 'disext', 'TOT']:
        df = pd.read_csv(file_fpath, header=None)
        if len(df.columns) == 5:
            columns = ['L1', 'L2', 'L3', 'L4', 'L5']
        elif len(df.columns) == 3:
            columns = ['TOT', 'GG', 'RET']
        else:
            columns = ['unknown']
        df.columns = columns

    return df


def confusion(label_file: str, pred_file: str, bland_in_1_mean_std=None, adap_markersize=1) -> dict:
    """

    Args:
        label_file: Full path for label
        pred_file: Full path for pred
        bland_in_1_mean_std:
        adap_markersize:

    Returns:
        A dict include all metrics

    Examples:
        :func:`ssc_scoring.mymodules.tool.compute_metrics` and :func:`ssc_scoring.compute_metrics.metrics`

    """
    print(f"Start the save of confsion matrix plot and csv for label: {label_file} and pred: {pred_file}")

    df_label = read_check(file_fpath=label_file)
    df_pred = read_check(file_fpath=pred_file)
    print('len_df_label', len(df_label))

    # if len(df_label.columns) == 5:
    #     df_pred -= 32
    # df_label = df_label.head(18) # pred_1 = "/data/jjia/ssc_scoring/LK_time2_18patients.csv"
    # df_pred = df_pred.head(18) #label_1 = "/data/jjia/ssc_scoring/ground_truth_18_patients.csv"
    out_dt = {}
    lower_y_ls, upper_y_ls = [], []
    lower_x_ls, upper_x_ls = [], []

    if bland_in_1_mean_std is not None:
        fig = plt.figure(figsize=(6, 4))
        fig_2 = plt.figure(figsize=(5, 4))
        row_nb, col_nb = 1, 1
    else:
        fig = plt.figure(figsize=(15, 4))
        fig_2 = plt.figure(figsize=(15, 5))

        if len(df_label.columns) == 3:
            row_nb, col_nb = 1, 3
        elif len(df_label.columns) == 5:
            row_nb, col_nb = 2, 3
        elif len(df_label.columns) == 1:
            row_nb, col_nb = 1, 1
        else:
            raise Exception(f'the columns number is not 3 or 5, it is {len(df_label.columns)} ', df_label.columns)

    basename = os.path.dirname(pred_file)
    prefix = pred_file.split("/")[-1].split("_")[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    for plot_id, column in enumerate(df_label.columns):
        label = df_label[column].to_numpy().reshape(-1, 1)
        pred = df_pred[column].to_numpy().reshape(-1, 1)

        # bland-altman plot
        if bland_in_1_mean_std is not None:
            ax = fig.add_subplot(row_nb, col_nb, 1)
            ax_2 = fig_2.add_subplot(row_nb, col_nb, 1)

        else:
            ax = fig.add_subplot(row_nb, col_nb, plot_id + 1)
            ax_2 = fig_2.add_subplot(row_nb, col_nb, plot_id + 1)

        # f, ax = plt.subplots(1, figsize=(8, 5))
        scatter_kwds = {'c': colors[plot_id], 'label': column}

        if bland_in_1_mean_std is None or plot_id == len(df_label.columns) - 1:
            if bland_in_1_mean_std is not None:
                label_all = df_label.to_numpy().flatten().reshape(-1, 1)
                pred_all = df_pred.to_numpy().flatten().reshape(-1, 1)
                diffs_all = pred_all - label_all
                mean_diff = np.mean(diffs_all)
                std_diff = np.std(diffs_all)
                bland_in_1_mean_std = {"mean": mean_diff, "std": std_diff}
            f = sm.mean_diff_plot(pred, label, ax=ax, scatter_kwds=scatter_kwds,
                                  bland_in_1_mean_std=bland_in_1_mean_std,
                                  adap_markersize=adap_markersize)
            f_2 = sm.mean_diff_plot(pred, label, ax=ax_2, sd_limit=0, scatter_kwds=scatter_kwds,
                                    bland_in_1_mean_std=bland_in_1_mean_std,
                                    adap_markersize=adap_markersize, ynotdiff=True)

            if bland_in_1_mean_std is None:
                ax.set_title(column, fontsize=15)
                ax_2.set_title(column, fontsize=15)

            # else:
            #     ax.set_title("All", fontsize=15)
            #     ax_2.set_title("All", fontsize=15)




        else:

            f = sm.mean_diff_plot(pred, label, ax=ax, sd_limit=0, scatter_kwds=scatter_kwds,
                                  bland_in_1_mean_std=bland_in_1_mean_std, adap_markersize=adap_markersize)

            f_2 = sm.mean_diff_plot(pred, label, ax=ax_2, sd_limit=0, scatter_kwds=scatter_kwds,
                                    bland_in_1_mean_std=bland_in_1_mean_std,
                                    adap_markersize=adap_markersize, ynotdiff=True)

            if bland_in_1_mean_std is None:
                ax.set_title(column, fontsize=15)
                ax_2.set_title(column, fontsize=15)
        lower_y, upper_y = ax.get_ybound()  # set these plots as the same scale for comparison
        lower_x, upper_x = ax.get_xbound()
        lower_y_ls.append(lower_y)
        upper_y_ls.append(upper_y)
        lower_x_ls.append(lower_x)
        upper_x_ls.append(upper_x)

        diff = pred.astype(int) - label.astype(int)
        abs_diff = np.abs(diff)
        ave_mae = np.mean(abs_diff)
        std_mae = np.std(abs_diff)
        mean = np.mean(diff)
        std = np.std(diff)

        print(f"ave_mae for {column} is {ave_mae}")
        print(f"std_mae for {column} is {std_mae}")

        print(f"mean for {column} is {mean}")
        print(f"std for {column} is {std}")

        if len(df_label.columns) == 3:
            kappa = cohen_kappa_score(label.astype(int), pred.astype(int), weights='linear',
                                      labels=np.array(list(range(100))))
            print(f"weighted kappa for {column} is {kappa}")

            pred[pred < 0] = 0
            pred[pred > 100] = 100

            index_label = list(range(0, 101, 5))
            index_pred = list(range(0, 101, 5))

            df = pd.DataFrame(0, index=index_label, columns=index_pred)
            scores = np.concatenate([label, pred], axis=1)

            for idx in index_label:
                mask = scores[:, 0] == idx
                rows = scores[mask, :]
                unique, counts = np.unique(rows[:, -1], return_counts=True)
                for u, c in zip(unique, counts):
                    df.at[idx, u] = c

            mae_np, total_np = mae(df)

            df.loc[:, 'Total'] = total_np

            diag_np = np.diag(df)

            acc_np = get_acc_np(diag_np, total_np)

            df.loc[:, 'Acc'] = acc_np
            df.loc[:, 'MAE'] = mae_np
            df.loc[0, 'Weighted_kappa'] = kappa
            df.loc[0, 'ave_mae'] = ave_mae
            df.loc[0, 'std_mae'] = std_mae

            df.loc[0, 'mean'] = mean
            df.loc[0, 'std'] = std

            if 'valid' in pred_file:
                out_dt['valid_ave_Acc_' + column] = np.nanmean(acc_np)  # there may be some np.nan
                out_dt['valid_ave_MAE_' + column] = np.nanmean(mae_np)
                out_dt['valid_WKappa_' + column] = kappa
                out_dt['valid_ave_mae' + column] = ave_mae
                out_dt['valid_std_mae' + column] = std_mae

                out_dt['valid_mean' + column] = mean
                out_dt['valid_std' + column] = std

            df.replace(0, np.nan, inplace=True)
            for idx, row in df.iterrows():
                if pd.isna(df.at[idx, 'Acc']) and not pd.isna(df.at[idx, 'Total']):
                    df.at[idx, 'Acc'] = 0

            df.to_csv(basename + '/' + prefix + "_" + column + '_confusion.csv')

        # colormap.set_bad("black")

        print("Finish confsion matrix plot and csv of ", column)

    # plot the bland-altman plot of all data
    if bland_in_1_mean_std:
        label = df_label.to_numpy().flatten().reshape(-1, )
        pred = df_pred.to_numpy().flatten().reshape(-1, )

        # plot linear regression line
        m, b = np.polyfit(label, pred, 1)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(label, pred)
        x_reference = np.array([0, 256])
        print('linear regression m, b:', m, b)
        print('linear regression m, b, r^2, p:', slope, intercept, r_value ** 2, p_value)

        diff = pred.astype(int) - label.astype(int)
        abs_diff = np.abs(diff)
        ave_mae_all = np.mean(abs_diff)
        std_mae_all = np.std(abs_diff)
        print('MAE for all is: ', ave_mae_all)
        print('STD of MAE for all is: ', std_mae_all)

        ax_2.plot(x_reference, m * x_reference + b, '--', color='black', linewidth=1)  # light gray
        ax_2.plot(x_reference, x_reference, '-', color='black', linewidth=1)  # light gray
        # ax_2.text(0.1, 0.7, '---  Regression line',
        #           ha="left", fontsize='large', transform=ax_2.transAxes)
        ax_2.text(0.1, 0.6, f'------\n'
                            f'——\n'
                            f'y\n'
                            f'R\N{SUPERSCRIPT TWO}\n'
                            f'P',
                  ha="left", fontsize='large', transform=ax_2.transAxes)
        if p_value < 0.01:
            ax_2.text(0.16, 0.6, f'    Regression line\n'
                                 f'    Identity line\n'
                                 f'= {m:.2f}x + {b:.2f}\n'
                                 f'= {r_value ** 2:.2f}\n'
                                 f'< 0.01',
                      ha="left", fontsize='large', transform=ax_2.transAxes)
        else:
            ax_2.text(0.16, 0.6, f'Regression line\n'
                                 f'= {m:.2f}x + {b:.2f}\n'
                                 f'= {r_value ** 2:.2f}\n'
                                 f'= {p_value}',
                      ha="left", fontsize='large', transform=ax_2.transAxes)
        # ax_2.text(0.05, 0.9, 'B)', ha="left", fontsize='xx-large', transform=ax_2.transAxes)
    else:
        for plot_id, column in enumerate(df_label.columns):
            label = df_label[column].to_numpy().reshape(-1, )
            pred = df_pred[column].to_numpy().reshape(-1, )

            ax_2 = fig_2.add_subplot(row_nb, col_nb, plot_id + 1)
            # plot linear regression line
            m, b = np.polyfit(label, pred, 1)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(label, pred)
            x_reference = np.array([0, 256])
            print(column, 'linear regression m, b:', m, b)
            print(column, 'linear regression m, b, r^2:', slope, intercept, r_value ** 2)

            ax_2.plot(x_reference, m * x_reference + b, '--', color='gray')  # light gray
            # ax_2.text(0.1, 0.7, '---  Regression line',
            #           ha="left", fontsize='large', transform=ax_2.transAxes)
            ax_2.text(0.1, 0.7, f'y = {m:.2f}x + {b:.2f}\nR\N{SUPERSCRIPT TWO} = {r_value ** 2: .2f}',
                      ha="left", fontsize='large', transform=ax_2.transAxes)

    # ax_2.annotate('{}x + {}'.format(np.round(m, 2), np.round(b, 2)),
    #             xy=(0.1, 0.7),
    #             fontsize=15,
    #             xycoords='axes fraction')

    # ax = fig.add_subplot(row_nb, col_nb, row_nb * col_nb)
    # f, ax = plt.subplots(1, figsize=(8, 5))
    # f = sm.graphics.mean_diff_plot(pred, label, ax=ax)
    # ax.set_title('All', fontsize=16)
    # lower, upper = ax.get_ybound()  # set these plots as the same scale for comparison
    # lower_ls.append(lower)
    # upper_ls.append(upper)
    lower_y, upper_y = min(lower_y_ls), max(upper_y_ls)
    lower_x, upper_x = min(lower_x_ls), max(upper_x_ls)

    print("lower:", lower_y, "upper:", upper_y)
    common_y = max(abs(lower_y), abs(upper_y))
    common_x = max(abs(lower_x), abs(upper_x))

    for i in range(row_nb * col_nb):
        ax = fig.add_subplot(row_nb, col_nb, i + 1)
        ax.set_ylim(-common_y * 1.2, common_y * 1.2)

        ax_2 = fig_2.add_subplot(row_nb, col_nb, i + 1)

        if len(df_label.columns) == 3:
            limitx = 100
            ax.set_xlim(0, limitx)

            ax_2.set_ylim(0, limitx)
            ax_2.set_xlim(0, limitx)


        elif len(df_label.columns) == 5:
            ax.set_xlim(0, 256)

            if all(label) < 256:
                ax_2.set_ylim(0, 255)
                ax_2.set_xlim(0, 255)
            ax.legend(loc="lower left")
            ax_2.legend(loc="lower right")

    # f.suptitle(prefix.capitalize() + " Bland-Altman Plot", fontsize=26)
    f.tight_layout()
    f.savefig(basename + '/' + prefix + '_bland_altman.png')
    plt.close(f)

    # f_2.suptitle(prefix.capitalize() + " Prediction Scatter Plot", fontsize=26)
    f_2.tight_layout()
    f_2.savefig(basename + '/' + prefix + '_label_pred_scatter.png')
    plt.close(f_2)

    if ('valid' in pred_file) and (len(df_label.columns) == 3):
        out_dt['valid_ave_Acc_all'] = 0
        out_dt['valid_ave_MAE_all'] = 0
        out_dt['valid_WKappa_all'] = 0
        for col in df_label.columns:
            out_dt['valid_ave_Acc_all'] += out_dt['valid_ave_Acc_' + col]
            out_dt['valid_ave_MAE_all'] += out_dt['valid_ave_MAE_' + col]
            out_dt['valid_WKappa_all'] += out_dt['valid_WKappa_' + col]
        out_dt['valid_ave_Acc_all'] /= len(df_label.columns)
        out_dt['valid_ave_MAE_all'] /= len(df_label.columns)
        out_dt['valid_WKappa_all'] /= len(df_label.columns)

    return out_dt


if __name__ == "__main__":
    ids = [528]
    ensemble = False

    for id in ids:
        # mypath = Path(id)
        # confusion(mypath.train_batch_label, mypath.train_batch_preds_end5)
        # confusion(mypath.valid_batch_label, mypath.valid_batch_preds_end5)
        # confusion(mypath.test_batch_label, mypath.test_batch_preds_end5)
        abs_dir_path = os.path.dirname(os.path.realpath(__file__))
        id_dir = abs_dir_path + "/models/" + str(id)

        if not ensemble:
            print(' start comfusion')
            for mode in ['train', 'valid', 'test']:
                label_files = sorted(glob.glob(os.path.join(id_dir, mode + "_label.csv")))
                pred_files = sorted(glob.glob(os.path.join(id_dir, mode + "_pred_int_end5.csv")))
                for label_file, pred_file in zip(label_files, pred_files):
                    confusion(label_file, pred_file)
        else:
            label_files = sorted(glob.glob(os.path.join(id_dir, "*", "test_batch_label.csv")))
            pred_files = sorted(glob.glob(os.path.join(id_dir, "*", "*ensemble.csv")))
            for label_file, pred_file in zip(label_files, pred_files):
                confusion(label_file, pred_file)
