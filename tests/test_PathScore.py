# -*- coding: utf-8 -*-
# @Time    : 7/10/21 11:56 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from ssc_scoring.mymodules.path import PathScore
import unittest


class TestPathScore(unittest.TestCase):
    def test_PathScore(self):
        mypath = PathScore('unittest', check_id_dir=False)
        self.assertEqual(mypath.results_dir, 'results')
        self.assertIn('results/records', mypath.record_file)
        self.assertIn('.csv', mypath.record_file)
        self.assertEqual(mypath.slurmlog_dir, 'results/slurmlogs')
        self.assertEqual(mypath.data_dir, 'dataset')
        self.assertEqual(mypath.label_excel_fpath, 'dataset/GohScores.xlsx')
        self.assertEqual(mypath.id, 'unittest')

        self.assertEqual(mypath.model_dir, 'results/models')
        self.assertEqual(mypath.id_dir, 'results/models/unittest')
        self.assertEqual(mypath.model_fpath, 'results/models/unittest/model.pt')
        self.assertEqual(mypath.model_wt_structure_fpath, 'results/models/unittest/model_wt_structure.pt')
        self.assertEqual(mypath.label('train'), 'results/models/unittest/train_label.csv')
        self.assertEqual(mypath.pred('train'), 'results/models/unittest/train_pred.csv')
        self.assertEqual(mypath.pred_int('valid'), 'results/models/unittest/valid_pred_int.csv')
        self.assertEqual(mypath.pred_end5('valid'), 'results/models/unittest/valid_pred_int_end5.csv')
        self.assertEqual(mypath.loss('train'), 'results/models/unittest/train_loss.csv')
        self.assertEqual(mypath.data('train'), 'results/models/unittest/train_data.csv')


if __name__ == "__main__":
    unittest.main()
