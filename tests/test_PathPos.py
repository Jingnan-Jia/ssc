# -*- coding: utf-8 -*-
# @Time    : 7/10/21 11:56 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from ssc_scoring.mymodules.path import PathPos
import unittest


class TestPathPos(unittest.TestCase):
    def test_PathPos(self):
        mypath = PathPos('unittest', check_id_dir=False)
        self.assertEqual(mypath.results_dir, 'results')
        self.assertIn('results/records_pos', mypath.record_file)
        self.assertIn('.csv', mypath.record_file)
        self.assertEqual(mypath.slurmlog_dir, 'results/slurmlogs')
        self.assertEqual(mypath.data_dir, 'dataset')
        self.assertEqual(mypath.label_excel_fpath, 'dataset/GohScores.xlsx')
        self.assertEqual(mypath.id, 'unittest')

        self.assertEqual(mypath.model_dir, 'results/models_pos')
        self.assertEqual(mypath.id_dir, 'results/models_pos/unittest')
        self.assertEqual(mypath.model_fpath, 'results/models_pos/unittest/model.pt')
        self.assertEqual(mypath.model_wt_structure_fpath, 'results/models_pos/unittest/model_wt_structure.pt')
        self.assertEqual(mypath.label('train'), 'results/models_pos/unittest/train_label.csv')
        self.assertEqual(mypath.pred('train'), 'results/models_pos/unittest/train_pred.csv')
        self.assertEqual(mypath.pred_int('valid'), 'results/models_pos/unittest/valid_pred_int.csv')

        self.assertEqual(mypath.pred_world('valid'), 'results/models_pos/unittest/valid_pred_world.csv')
        self.assertEqual(mypath.world('valid'), 'results/models_pos/unittest/valid_world.csv')

        self.assertEqual(mypath.loss('train'), 'results/models_pos/unittest/train_loss.csv')
        self.assertEqual(mypath.data('train'), 'results/models_pos/unittest/train_data.csv')

        self.assertEqual(mypath.dataset_dir(resample_z=0), 'dataset/SSc_DeepLearning')
        self.assertEqual(mypath.dataset_dir(resample_z=256), 'dataset/LowRes256_256_256')
        self.assertEqual(mypath.dataset_dir(resample_z=512), 'dataset/LowRes512_192_192')
        self.assertEqual(mypath.dataset_dir(resample_z=800), 'dataset/LowRes800_160_160')
        self.assertEqual(mypath.dataset_dir(resample_z=1024), 'dataset/LowRes1024_256_256')




if __name__ == "__main__":
    unittest.main()
