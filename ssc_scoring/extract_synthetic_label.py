# Used for the synthesis of label distribution

import copy

import matplotlib.pyplot as plt
import numpy as np


def extract_label(file_fpath):
    score_disext, score_gg, score_retp = [], [], []
    score_disext_ori, score_gg_ori, score_retp_ori = [], [], []

    with open(file_fpath) as f:
        for row in f:
            divid_flap = "after systhesis, label is tensor"
            if divid_flap in row:
                score = row.split(divid_flap)[-1]
                score = score[2:-3]  # exclude [( and )]\n
                score = score.split(",")
                try:
                    score_disext.append(int(float(score[0])))
                    score_gg.append(int(float(score[1])))
                    score_retp.append(int(float(score[2])))
                except:
                    pass
            divid_flap2 = "No need for systhesis, label is tensor"
            if divid_flap2 in row:
                score = row.split(divid_flap2)[-1]
                score = score[2:-3]  # exclude [( and )]\n
                score = score.split(",")
                try:
                    score_disext_ori.append(int(float(score[0])))
                    score_gg_ori.append(int(float(score[1])))
                    score_retp_ori.append(int(float(score[2])))
                except:
                    pass
    fig = plt.figure(figsize=(20, 6))
    # x = np.arange(0, 105, 5)
    bar_dt = {label: key for label, key in zip(np.arange(0, 21) * 5, np.zeros((21,)).astype(int))}

    for idx, score in enumerate([score_disext, score_gg, score_retp]):
        score = [s // 5 * 5 for s in score]
        bar_dt_ = copy.deepcopy(bar_dt)
        for s in score:
            bar_dt_[s] += 1
        ax = fig.add_subplot(1, 3, idx+1)
        ax.bar(np.arange(0, 21) * 5, np.array(list(bar_dt_.values())))
        # ax.hist(score, bins=21)
    plt.title("syn")
    plt.show()

    fig = plt.figure(figsize=(20, 6))
    for idx, score in enumerate([score_disext_ori, score_gg_ori, score_retp_ori]):
        ax = fig.add_subplot(1, 3, idx + 1)
        score = [s // 5 * 5 for s in score]
        bar_dt_ = copy.deepcopy(bar_dt)
        for s in score:
            bar_dt_[s] += 1
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.bar(np.arange(0, 21) * 5, np.array(list(bar_dt_.values())))

        # ax.hist(score, bins=21)
    plt.title("ori")
    plt.show()

    fig = plt.figure(figsize=(20, 6))
    score_disext.extend(score_disext_ori)
    score_gg.extend(score_gg_ori)
    score_retp.extend(score_retp_ori)
    for idx, score in enumerate([score_disext,score_gg,score_retp]):
        score = [s // 5 * 5 for s in score]

        ax = fig.add_subplot(1, 3, idx + 1)
        bar_dt_ = copy.deepcopy(bar_dt)
        for s in score:
            bar_dt_[s] += 1
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.bar(np.arange(0, 21) * 5, np.array(list(bar_dt_.values())))

        # ax.hist(score, bins=21)
    plt.title("all")
    plt.show()
    print('finish extracting synthetic labels')



if __name__ == '__main__':
    path = "results/slurmlogs/slurm-96342_0.out"
    extract_label(file_fpath = path)