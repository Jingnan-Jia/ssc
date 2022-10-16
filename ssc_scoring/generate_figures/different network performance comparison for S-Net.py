"""
This script is used to generate a box plot for the comparison of different networks for S-Net. Then we can decide which
network we should adopt.
"""
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ssc_scoring.mymodules.path import PathScore, PathPos
import csv
from collections import OrderedDict

def snet_figure():

    net_id_dt = {"ConvNeXt": 1883,
                 "SqueezeNet": 1879,
                 "ShuffleNet": 1878,
                 "DenseNet": 1875,
                 "ResNeXt50": 1843,
                 "ResNet50": 1880,
                 "ResNet18": 1844,
                 "VGG19": 1847,
                 "VGG16": 1874,
                 "VGG11": 1846}

    # label_file = mypath.label_excel_fpath # "dataset/SSc_DeepLearning/GohScores.xlsx"  # labels are from here
    # df_excel = pd.read_excel(label_file, engine='openpyxl')
    # df_excel = df_excel.set_index('PatID')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    mypath_init = PathScore(1883)  # initiate the path using a random ex id
    # fig = plt.figure()
    modes = ['valid', 'test', 'train']
    for mode_idx, mode in enumerate(modes):
        # have_file = True
        label_fpath = mypath_init.label(mode)  # this file include 3 columns for tot, gg and ret.
        df_label = pd.read_csv(label_fpath)

        # for net, id in net_id_dt.items():
        #     mypath_init = Path(id)  # initiate the path using a random ex id
        #     label_fpath = mypath_init.label(mode)  # this file include 3 columns for tot, gg and ret.
        #     try:
        #         df_label = pd.read_csv(label_fpath)
        #         break
        #     except:
        #         have_file = False
        # if not have_file:
        #     continue

        # At present, we have, say, 10 different networks.
        # Each network corresponds to one experiment id.
        # Each id corresponds to 3 modes (train, valid, and test).
        # Each mode corresponds to 3 patterns (tot, gg, and ret).
        # So we will generate 3 groups of boxplots comparison for 3 modes.
        # In each group we have 10 boxs (3 patterns are merged to obtain one box), corresponding to 10 nets.
        df_all = pd.DataFrame()
        ae_ls, net_ls, mae_ls = [], [], []

        for net, id in net_id_dt.items():
            mypath = PathScore(id)  # initiate the path using a random ex id

            pred_fpath = mypath.pred(mode)
            df_pred = pd.read_csv(pred_fpath)
            diff = df_label - df_pred
            ae = diff.abs()
            ae_3in1 = ae.to_numpy().reshape(-1,)  # merge 3 patterns mae
            ae_middleslice = ae_3in1[::3]  # exclude the repeated neighboring slices
            df_all[net] = ae_middleslice

            ae_ls.append(ae_middleslice)
            net_ls.append(net)
            mae_ls.append(np.median(ae_middleslice))

        # Sort nets by the decreasing mae values
        mae_ls, net_ls, ae_ls = zip(*sorted(zip(mae_ls, net_ls, ae_ls)))

        fig, ax = plt.subplots()  #figsize=(8, 4)
        # for n, col in enumerate(df_all.columns):
        #     ax.boxplot(df_all[col], positions=[n + 1], notch=True, widths=0.7)
        for n, error in enumerate(ae_ls):
            ax.boxplot(error, positions=[n + 1], notch=True, widths=0.5)
        ax.set_xlabel(f"S-Net architecture")
        ax.set_ylabel(f"Absolute error [%]")
        plt.xticks(list(range(1,len(net_id_dt)+1)), net_ls)
        plt.xticks(rotation=15)  # Rotates X-Axis Ticks by 45-degrees
        plt.show()
        fig.savefig(f"{mode}_different_net_comparison_S-Net_vertal.png", bbox_inches = "tight")
        print(f"saved at: {mode}_different_net_comparison_S-Net_vertal.png")

        # generate the horizental figure
        fig, ax = plt.subplots(figsize=(8, 6))  # figsize=(8, 4)
        for n, error in enumerate(ae_ls):
            ax.boxplot(error, positions=[n + 1], notch=True, widths=0.5, vert=False)
        ax.set_ylabel(f"S-Net architecture")
        ax.set_xlabel(f"Absolute error [%]")
        plt.yticks(list(range(1,len(net_id_dt)+1)), net_ls)
        plt.show()
        fig.savefig(f"{mode}_different_net_comparison_S-Net_horizental.png", bbox_inches = "tight")
        print(f"saved at: {mode}_different_net_comparison_S-Net_horizental.png")


def lnet_figure():
    # fold 1
    net_id_dt = OrderedDict({ "        VGG11  ": 738, "      VGG19  ": 747,"      VGG16  ": 749,
                             # "vgg16": 749, #749
                             # 747
                             })

    # label_file = mypath.label_excel_fpath # "dataset/SSc_DeepLearning/GohScores.xlsx"  # labels are from here
    # df_excel = pd.read_excel(label_file, engine='openpyxl')
    # df_excel = df_excel.set_index('PatID')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    mypath_init = PathPos(753)  # initiate the path using a random ex id
    # fig = plt.figure()
    modes = ['valid', 'test', 'train']
    for mode_idx, mode in enumerate(modes):
        label_fpath = mypath_init.label(mode)  # this file include 3 columns for tot, gg and ret.
        df_label = pd.read_csv(label_fpath)

        # have_file = True
        # for net, id in net_id_dt.items():
        #     mypath_init = Path(id)  # initiate the path using a random ex id
        #     label_fpath = mypath_init.label(mode)  # this file include 3 columns for tot, gg and ret.
        #     try:
        #         df_label = pd.read_csv(label_fpath)
        #         break
        #     except:
        #         have_file = False
        # if not have_file:
        #     continue

        # At present, we have, say, 10 different networks.
        # Each network corresponds to one experiment id.
        # Each id corresponds to 3 modes (train, valid, and test).
        # Each mode corresponds to 3 patterns (tot, gg, and ret).
        # So we will generate 3 groups of boxplots comparison for 3 modes.
        # In each group we have 10 boxs (3 patterns are merged to obtain one box), corresponding to 10 nets.
        df_all = pd.DataFrame()
        ae_ls, net_ls, mae_ls = [], [], []

        for net, id in net_id_dt.items():
            mypath = PathPos(id)  # initiate the path using a random ex id

            pred_fpath = mypath.pred(mode)
            df_pred = pd.read_csv(pred_fpath)
            if id != 193:
                df_pred -= 32  # fix the older bug

            diff = df_label - df_pred
            ae = diff.abs()
            ae_5in1 = ae.to_numpy().reshape(-1,)  # merge 5 levels mae
            ae_middleslice = ae_5in1
            df_all[net] = ae_middleslice

            ae_ls.append(ae_middleslice)
            net_ls.append(net)
            mae_ls.append(np.median(ae_middleslice))

        # Sort nets by the decreasing mae values
        # mae_ls, net_ls, ae_ls = zip(*sorted(zip(mae_ls, net_ls, ae_ls)))

        fig, ax = plt.subplots(figsize=(2, 6))
        # for n, col in enumerate(df_all.columns):
        #     ax.boxplot(df_all[col], positions=[n + 1], notch=True, widths=0.7)
        for n, error in enumerate(ae_ls):
            ax.boxplot(error, positions=[n + 1], notch=True, widths=0.4)
        ax.set_xlabel(f"L-Net architecture")
        ax.set_ylabel(f"Absolute error [slices]")
        plt.xticks(list(range(1,len(net_id_dt)+1)), net_ls)
        ymin, ymax = ax.get_ylim()

        # label_diff(ax, 1, 2, ymax + 5, 'p>0.01')
        ax.set_ylim(ymin, ymax + 10)

        # plt.xticks(rotation=15)  # Rotates X-Axis Ticks by 45-degrees
        plt.show()
        fig.savefig(f"{mode}_different_net_comparison_L-Net_vertal.png", bbox_inches = "tight")
        print(f"saved at: {mode}_different_net_comparison_L-Net_vertal.png")

        # generate the horizental figure
        fig, ax = plt.subplots(figsize=(8, 2))
        for n, error in enumerate(ae_ls):
            ax.boxplot(error, positions=[n + 1], notch=True, widths=0.45, vert=False)
        ax.set_ylabel(f"L-Net architecture")
        ax.set_xlabel(f"Absolute error [slices]")
        plt.yticks(list(range(1,len(net_id_dt)+1)), net_ls)
        plt.show()
        fig.savefig(f"{mode}_different_net_comparison_L-Net_horizental.png", bbox_inches = "tight")
        print(f"saved at: {mode}_different_net_comparison_L-Net_horizental.png")


def lnet_snet_figure():
    fig, (ax_l, ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})


    modes = ['valid', 'test', 'train']
    for mode_idx, mode in enumerate(modes):

        # net_id_dt =
        net_id_dt = OrderedDict({"vgg11": 738,"vgg16": 749,"vgg19": 747,
                     # "vgg16": 749, #749
                       # 747
                     })

        mypath_init = PathPos(753)  # initiate the path using a random ex id

        label_fpath = mypath_init.label(mode)  # this file include 3 columns for tot, gg and ret.
        df_label = pd.read_csv(label_fpath)

        df_all = pd.DataFrame()
        ae_ls, net_ls, mae_ls = [], [], []
        for net, id in net_id_dt.items():
            mypath = PathPos(id)  # initiate the path using a random ex id

            pred_fpath = mypath.pred(mode)
            df_pred = pd.read_csv(pred_fpath)
            if id != 193:
                df_pred -= 32  # fix the older bug

            diff = df_label - df_pred
            ae = diff.abs()
            ae_5in1 = ae.to_numpy().reshape(-1,)  # merge 5 levels mae
            ae_middleslice = ae_5in1
            df_all[net] = ae_middleslice

            ae_ls.append(ae_middleslice)
            net_ls.append(net)
            mae_ls.append(np.median(ae_middleslice))

        # Sort nets by the decreasing mae values
        # mae_ls, net_ls, ae_ls = zip(*sorted(zip(mae_ls, net_ls, ae_ls)))

        for n, error in enumerate(ae_ls):
            ax_l.boxplot(error, positions=[n + 1], notch=True, widths=0.7)
        ax_l.set_xlabel(f"L-Net architecture")
        ax_l.set_ylabel(f"Absolute error [slices]")
        ax_l.set_xticks(list(range(1,len(net_id_dt)+1)), net_ls)
        # plt.xticks(rotation=15)  # Rotates X-Axis Ticks by 45-degrees

        net_id_dt = {"convnext": 1883,
                     "squeezenet": 1879,
                     "shufflenet": 1878,
                     "densenet": 1875,
                     "resnext50": 1843,
                     "resnet50": 1880,
                     "resnet18": 1844,
                     "vgg19": 1847,
                     "vgg16": 1874,
                     "vgg11": 1846}
        mypath_init = PathScore(1883)  # initiate the path using a random ex id

        label_fpath = mypath_init.label(mode)  # this file include 3 columns for tot, gg and ret.
        df_label = pd.read_csv(label_fpath)
        df_all = pd.DataFrame()
        ae_ls, net_ls, mae_ls = [], [], []

        for net, id in net_id_dt.items():
            mypath = PathScore(id)  # initiate the path using a random ex id

            pred_fpath = mypath.pred(mode)
            df_pred = pd.read_csv(pred_fpath)
            diff = df_label - df_pred
            ae = diff.abs()
            ae_3in1 = ae.to_numpy().reshape(-1,)  # merge 3 patterns mae
            ae_middleslice = ae_3in1[::3]  # exclude the repeated neighboring slices
            df_all[net] = ae_middleslice

            ae_ls.append(ae_middleslice)
            net_ls.append(net)
            mae_ls.append(np.median(ae_middleslice))

        # Sort nets by the decreasing mae values
        mae_ls, net_ls, ae_ls = zip(*sorted(zip(mae_ls, net_ls, ae_ls)))

        # generate the horizental figure
        for n, error in enumerate(ae_ls):
            ax.boxplot(error, positions=[n + 1], notch=True, widths=0.5, vert=False)
        ax.set_ylabel(f"S-Net architecture")
        ax.set_xlabel(f"Absolute error [%]")
        ax.set_yticks(list(range(1,len(net_id_dt)+1)), net_ls)

        plt.show()
        fig.savefig(f"{mode}_LS-Net.png", bbox_inches = "tight")
        print(f"{mode}_LS-Net.png")


def label_diff(ax, x1, x2,y,text):
    x = (x1 + x2)/2
    ymin, ymax = ax.get_ylim()
    yshift = (ymax - ymin)/30

    shift = 0.0
    x1 += shift
    x2 -= shift
    # props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':20,'shrinkB':20,'linewidth':2}
    ax.annotate(text, xy=(x, y+1), zorder=10, ha='center')
    ax.plot([x1 ,x2], [y,y], c='k')
    ax.plot([x1, x1], [y, y-yshift], c='k')
    ax.plot([x2, x2], [y, y-yshift], c='k')

    # ax.annotate('', xy=(x1, y), xytext=(x2, y),zorder=20,  arrowprops=props)



if __name__ == "__main__":
    # lnet_snet_figure()

    # snet_figure()
    lnet_figure()
