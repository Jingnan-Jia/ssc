# -*- coding: utf-8 -*-
# @Time    : 7/20/21 1:38 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

# Wilcoxon signed-rank test
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_ind
# seed the random number generator
# seed(1)
# generate two independent samples
def main():
	# data1_fpath = 'results/models/1405_1404_1411_1410/valid_pred_int_end5.csv'  # sys + pre_trained
	# data2_fpath = 'results/models/1585_1586_1587_1588/valid_pred_int_end5.csv'  # sys + pre_trained, NormData from -1 to 1

	# data1_fpath = 'results/models/1136_1132_1135_1134/valid_pred_int_end5.csv'  # wo sys, wo sampler, with re_trrained
	# data2_fpath = 'results/models/1481_1114_1115_1116/valid_pred_int_end5.csv'  # wo sys, with sampler, with re_trrained

	# ref_fpath = 'results/models/1405_1404_1411_1410/valid_label.csv'

	# data1_fpath = 'results/models_pos/193_194_276_277/valid_pred.csv'
	# data2_fpath = 'results/models_pos/193_194_276_277/valid_label.csv'
	# ref_fpath = ''
	data1_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/LKT2_16patients.csv"
	data2_fpath = "/data/jjia/ssc_scoring/ssc_scoring/results/models/1826/16pats_pred.csv"

	ref_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/ground_truth_16patients.csv"

	data1 = pd.read_csv(data1_fpath)
	data2 = pd.read_csv(data2_fpath)
	data_ref = pd.read_csv(ref_fpath)

	data1 = data1 - data_ref
	data2 = data2 - data_ref

	for col in data1.columns:
		if col in ['ID', 'Level']:
			continue
		data1_col = data1[col].to_numpy()
		data2_col = data2[col].to_numpy()
		assert len(data1_col) == len(data2_col)
		# compare samples
		print(ttest_ind(data1_col, data2_col))

		stat, p = wilcoxon(data1_col, data2_col)
		print(f'{col}: '
			  f'mean_1: {np.mean(data1_col): .2f}, std_1: {np.std(data1_col): .2f}, '
			  f'mean_2: {np.mean(data2_col): .2f}, std_2: {np.std(data2_col): .2f}, '
			  f'p={p: .3f}')
		# interpret
		alpha = 0.05
		if p > alpha:
			print('Same distribution (fail to reject H0)')
		else:
			print('Different distribution (reject H0)')

if __name__ == "__main__":
	main()