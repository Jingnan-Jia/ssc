import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import csv
import seaborn as sns

def main():
    OBSERVER = 'Lucia'  # 'anne'
    PIE = True
    SCATTER = False
    VIOLIN = False

    json_fpath = f"/home/jjia/data/ssc_scoring/ssc_scoring/heat_map_performance_{OBSERVER}.json"
    with open(json_fpath) as f:
        results = json.load(f)
    tot_ls = [i['TOT'] for i in results]
    gg_ls = [i['GG'] for i in results]
    ret_ls = [i['RET'] for i in results]
    pat_lv_ls = [i['Pat_ID']+'_'+i['Level'] for i in results]



    if PIE:

        # fig1, ax1 = plt.subplots()
        labels = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
        fig = plt.figure(figsize=(12, 3))
        title_ls = ['TOT', 'GG', 'RET']
        for idx, pattern_ls in enumerate([tot_ls, gg_ls, ret_ls]):
            values, counts = np.unique(np.array(pattern_ls), return_counts=True)
            values, counts = zip(*sorted(zip(values, counts)))

            for nb in ('1', '2', '3', '4', '5'):  # in case some values are 0 frequency
                if nb not in values:
                    values = values + (nb,)
                    counts = counts + (0,)

            values, counts = zip(*sorted(zip(values, counts)))
            ax = fig.add_subplot(1, 4, idx + 2)
            explode = (0, 0, 0, 0, 0)  # only "explode" the 'Agree' and 'Strongly agree'

            patches, texts, autotexts = ax.pie(counts, explode=explode, autopct='%1.1f%%',
                    shadow=False, startangle=0, colors=['lightgreen', 'lightskyblue', 'yellow', 'orange', 'red'][::-1],
                    textprops={'fontsize': 11})
            iii = 0
            for patch, txt in zip(patches, autotexts):
                iii += 1
                if iii<5:  # 4,3,2,1
                    angle = (patch.theta2 + patch.theta1) / 2.
                    if iii%2==0: # 2 or 4
                        x = patch.r * 0.5 * np.cos(angle * np.pi / 180)
                        y = patch.r * 0.5 * np.sin(angle * np.pi / 180)
                        # move text to new position
                        txt.set_position((x, y))  
                    else:  # 3 or 1
                        x = patch.r * 0.8 * np.cos(angle * np.pi / 180)
                        y = patch.r * 0.8 * np.sin(angle * np.pi / 180)
                        # move text to new position
                        txt.set_position((x, y))  
                    

                    if angle==0:  # remove it
                        x = patch.r * 1.2 * np.cos(angle * np.pi / 180)
                        y = patch.r * 1.2 * np.sin(angle * np.pi / 180)
                        # move text to new position
                        txt.set_position((x, y))
                        txt.set_color('white')

            # ax.set_title(title_ls[idx], y=0.92, fontdict={'fontsize':12})

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
        goh_data = goh_data[1::3]  # check if the order of pattern_ls is the same as label or pred csv files.

        error_all = np.abs(goh_label - goh_pred)

        def plot_scatter(data, name):

            fig = plt.figure(figsize=(12, 4))
            title_ls = ['TOT', 'GG', 'RET']
            y_max = 0
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

            for idx, pattern_ls in enumerate([tot_ls, gg_ls, ret_ls]):
                goh_ = data[:, idx]
                pattern = np.array(pattern_ls)
                error_sum_ls = [0, 0, 0, 0, 0]
                error_enum_ls = [0, 0, 0, 0, 0]
                x_label = [1, 2, 3, 4, 5]
                for idx_, patt in enumerate(pattern):
                    error_sum_ls[int(patt) - 1] += goh_[idx_]
                    error_enum_ls[int(patt) - 1] += 1
                error_ave_ls = [i / (j + 0.01) for i, j in zip(error_sum_ls, error_enum_ls)]
                y_max = max(y_max, max(error_ave_ls))
                scatter_kwds = {'c': colors[idx], "s": np.array([error_enum_ls[i - 1] * 4 for i in x_label])}
                ax = fig.add_subplot(1, 3, idx + 1)
                ax.scatter(x_label, error_ave_ls, **scatter_kwds)
                ax.set_title(title_ls[idx], y=1, fontdict={'fontsize': 12})
            print(f"ymax: {y_max}")
            for i in range(3):
                ax = fig.add_subplot(1, 3, i + 1)
                ax.set_ylim([0, y_max + 1])
                ax.set_ylabel(f"{name}", fontdict={'fontsize': 12})
                ax.set_xlabel('Likert scale', fontdict={'fontsize': 12})

            plt.tight_layout()
            plt.show()
            plt.savefig(f'heatmap_relation2{name}_{OBSERVER}.png')

        plot_scatter(goh_label, "GohScore")
        plot_scatter(error_all, "MAE")

        print(len(goh_label))

        # print(goh_label)
        # print(goh_pred)
    if VIOLIN:

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
        goh_data = np.array(goh_data_ls[:len(goh_label)])  # sometimes a lot of goh_data will be writen to the file.
        goh_label = goh_label[1::3]
        goh_pred = goh_pred[1::3]
        goh_data = goh_data[1::3]

        pat_lv_ls_ = [str[0].split('Pat_')[-1][:3] + '_' + str[0].split('Level')[-1][:1] for str in goh_data]
        for i, j in zip(pat_lv_ls_, pat_lv_ls):
            assert i == j

        error_all = np.abs(goh_label - goh_pred)

        # Build a DataFrame for the latter seaborn plot.
        all_df = pd.DataFrame(columns=["TOT_likert", "GG_likert", "RET_likert", "TOT_label", "GG_label", "RET_label",
                                       "TOT_pred", "GG_pred", "RET_pred", "TOT_mae", "GG_mae", "RET_mae"])
        all_df["TOT_likert"] = tot_ls
        all_df["GG_likert"] = gg_ls
        all_df["RET_likert"] = ret_ls
        all_df["TOT_label"] = goh_label[:,0]
        all_df["GG_label"] = goh_label[:,1]
        all_df["RET_label"] = goh_label[:,2]
        all_df["TOT_pred"] = goh_pred[:,0]
        all_df["GG_pred"] = goh_pred[:,1]
        all_df["RET_pred"] = goh_pred[:,2]
        all_df["TOT_mae"] = np.abs(goh_label - goh_pred)[:,0]
        all_df["GG_mae"] = np.abs(goh_label - goh_pred)[:,1]
        all_df["RET_mae"] = np.abs(goh_label - goh_pred)[:,2]


        def plot_violin(score_or_mae='score'):
            sns.set_context("paper", font_scale=1.1)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
            y_max = 0
            for idx,pattern in enumerate(['TOT', 'GG', 'RET']):
                plt.figure(figsize=(8, 6))
                sns.violinplot(y=f"{pattern}_{score_or_mae}",
                               x=f"{pattern}_likert",
                               data=all_df, color=colors[idx])
                sns.swarmplot(y=f"{pattern}_{score_or_mae}",
                               x=f"{pattern}_likert",
                               data=all_df,
                              color="white", edgecolor="gray")
                y_max = max(y_max, max(all_df[f"{pattern}_{score_or_mae}"]))

                plt.savefig(f"Seaborn_violinplot_with_points_swarmplot_{pattern}_{score_or_mae}.png",
                            format='png', dpi=150)

            # fig = plt.figure(figsize=(12, 4))
            # title_ls = ['TOT', 'GG', 'RET']
            # y_max = 0
            #
            # for idx, pattern_ls in enumerate([tot_ls, gg_ls, ret_ls]):
            #     goh_ = data[:, idx]
            #     pattern = np.array(pattern_ls)
            #     error_sum_ls = [0, 0, 0, 0, 0]
            #     error_enum_ls = [0, 0, 0, 0, 0]
            #     x_label = [1, 2, 3, 4, 5]
            #     for idx_, patt in enumerate(pattern):
            #         error_sum_ls[int(patt) - 1] += goh_[idx_]
            #         error_enum_ls[int(patt) - 1] += 1
            #     error_ave_ls = [i / (j + 0.01) for i, j in zip(error_sum_ls, error_enum_ls)]
            #     y_max = max(y_max, max(error_ave_ls))
            #     scatter_kwds = {'c': colors[idx], "s": np.array([error_enum_ls[i - 1] * 4 for i in x_label])}
            #     ax = fig.add_subplot(1, 3, idx + 1)
            #     ax.scatter(x_label, error_ave_ls, **scatter_kwds)
            #     ax.set_title(title_ls[idx], y=1, fontdict={'fontsize': 12})
            # print(f"ymax: {y_max}")
            # for i in range(3):
            #     ax = fig.add_subplot(1, 3, i + 1)
            #     ax.set_ylim([0, y_max + 1])
            #     ax.set_ylabel(f"{name}", fontdict={'fontsize': 12})
            #     ax.set_xlabel('Likert scale', fontdict={'fontsize': 12})
            #
            # plt.tight_layout()
            # plt.show()
            # plt.savefig(f'heatmap_relation2{name}_{OBSERVER}.png')

        plot_violin(score_or_mae="pred")
        plot_violin(score_or_mae="mae")

        print(len(goh_label))








if __name__ == "__main__":
    main()