import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import csv


def main():
    OBSERVER = 'Lucia'
    PIE = True
    SCATTER = True
    if PIE:
        json_fpath = f"/home/jjia/data/ssc_scoring/ssc_scoring/heat_map_performance_{OBSERVER}.json"
        with open(json_fpath) as f:
            results = json.load(f)
        tot_ls = [i['TOT'] for i in results]
        gg_ls = [i['GG'] for i in results]
        ret_ls = [i['RET'] for i in results]

        # fig1, ax1 = plt.subplots()
        labels = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
        fig = plt.figure(figsize=(12, 3))
        title_ls = ['TOT', 'GG', 'RET']
        for idx, pattern_ls in enumerate([tot_ls, gg_ls, ret_ls]):
            values, counts = np.unique(np.array(pattern_ls), return_counts=True)
            values, counts = zip(*sorted(zip(values, counts)))
            ax = fig.add_subplot(1, 4, idx + 2)
            explode = (0, 0, 0, 0, 0)  # only "explode" the 'Agree' and 'Strongly agree'

            patches, texts, autotexts = ax.pie(counts, explode=explode, autopct='%1.1f%%',
                    shadow=False, startangle=0)
            ax.set_title(title_ls[idx], y=0.92, fontdict={'fontsize':12})

            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax = fig.add_subplot(1, 4, 1)
        plt.legend(patches[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.5, 0.5),
                   fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'heatmap_evaluation_{OBSERVER}.png')

    if SCATTER:
        ################################################
        # draw relationship scatter plot
        folder = "/home/jjia/data/ssc_scoring/ssc_scoring/results/models/1903"
        goh_label_fpath = folder + "/valid_label.csv"
        goh_pred_fpath = folder + "/valid_pred.csv"
        goh_data_fpath = folder + "/valid_data.csv"

        goh_label_ls, goh_pred_ls, goh_data_ls = [], [], []
        for fpath, ls in zip([goh_label_fpath, goh_pred_fpath, goh_data_fpath],
                     [goh_label_ls, goh_pred_ls, goh_data_ls]):
            with open(fpath) as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    ls.append(row)

        print(len(goh_label_ls))
        goh_label = np.array(goh_label_ls[1:]).astype(np.float16)  # remove the title text
        goh_pred = np.array(goh_pred_ls[1:]).astype(np.float16)
        goh_data = np.array(goh_data_ls[1:])
        goh_label = goh_label[1::3]
        goh_pred = goh_pred[1::3]
        goh_data = goh_data[1::3]

        error_all = np.abs(goh_label - goh_pred)
        fig = plt.figure(figsize=(12, 4))
        title_ls = ['TOT', 'GG', 'RET']
        y_max = 0
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

        for idx, pattern_ls in enumerate([tot_ls, gg_ls, ret_ls]):
            error = error_all[:, idx]
            goh_ = goh_label[:, idx]
            pattern = np.array(pattern_ls)
            error_sum_ls = [0,0,0,0,0]
            error_enum_ls = [0,0,0,0,0]
            x_label = [1,2,3,4,5]
            for idx_, patt in enumerate(pattern):
                error_sum_ls[int(patt)-1] += goh_[idx_]
                error_enum_ls[int(patt)-1] += 1
            error_ave_ls = [i/j for i, j in zip(error_sum_ls, error_enum_ls)]
            y_max = max(y_max, max(error_ave_ls))
            scatter_kwds = {'c': colors[idx], "s": np.array([error_enum_ls[i-1] *4 for i in x_label])}
            ax = fig.add_subplot(1, 3, idx + 1)
            ax.scatter(x_label, error_ave_ls, **scatter_kwds)
            ax.set_title(title_ls[idx], y=1, fontdict={'fontsize':12})
        print(f"ymax: {y_max}")
        for i in range(3):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_ylim([0, y_max+1])
            ax.set_ylabel('Goh score label', fontdict={'fontsize':12})
            ax.set_xlabel('Likert scale', fontdict={'fontsize':12})

        plt.tight_layout()
        plt.show()
        plt.savefig(f'heatmap_relation2error_{OBSERVER}.png')

        print(len(goh_label))

        print(goh_label)
        print(goh_pred)















if __name__ == "__main__":
    main()