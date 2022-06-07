# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
from parameterized import parameterized
from ssc_scoring.mymodules.mytrans import CropCorseRegiond
import numpy as np
from tests.utils import Compare
import tempfile
import medutils.medutils as futil
import csv
import os


data = ['Pat_045.mha', [500, 600, 700, 800, 9000]]
pred_world = np.array([501, 601, 701, 801, 901])
head = ['L1', 'L2', 'L3', 'L4', 'L5']


temp_dir = tempfile.TemporaryDirectory()
print(temp_dir.name)
data_fpath = os.path.join(temp_dir.name, "train_data.csv")
pred_world_fpath = os.path.join(temp_dir.name, "train_pred_world.csv")
# use temp_dir, and when done:

with tempfile.TemporaryDirectory() as tempdir:
    with open(data_fpath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    futil.appendrows_to(pred_world_fpath,pred_world, head=head)


TEST_CASE_3D_5Label_1 = [
    {"level_node": 0, 
     "train_on_level": 1, 
     "height": 200, 
     "rand_start": False, 
     "start": 400,
     "data_fpath": data_fpath,
     "pred_world_fpath": pred_world_fpath},

    {"image_key": np.ones((1000, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     # world position in mm, keep fixed,  a np.array with shape(-1, )
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     'space_key': np.array([1, 2, 3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100, 200, 30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/Pat_045.mha"},  # full path, a string

    {"image_key": np.ones((200, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": np.array([101]),
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1000]),
     # world position in mm, keep fixed,  a np.array with shape(-1, )
     'space_key': np.array([1, 2, 3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100, 200, 30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/Pat_045.mha",
     'level_key': np.array([1]),

     'corse_pred_in_img_key': np.array([501, 601, 701, 801, 901]),
     'corse_pred_in_img_1_key': np.array([501]),


     }
]

TEST_CASE_3D_5Label_2 = [
    {"level_node": 0, "train_on_level": 2, "height": 400, "rand_start": False, "start": 400,

     "data_fpath": data_fpath,
     "pred_world_fpath": pred_world_fpath
     },

    {"image_key": np.ones((1000, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
'fpath_key': "/data/samples/Pat_045.mha",

     },  # full path, a string

    {"image_key": np.ones((400, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": np.array([201]),  # not sure
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1200]),
     'level_key': np.array([2]),
'fpath_key': "/data/samples/Pat_045.mha",
     'corse_pred_in_img_key': np.array([501, 601, 701, 801, 901]),
     'corse_pred_in_img_1_key': np.array([601]),

     }  # full path, a string
]

TEST_CASE_3D_5Label_3 = [  # "rand_start": True, test shape
    {"level_node": 0, "train_on_level": 2, "height": 400, "rand_start": True, "start": 400,

     "data_fpath": data_fpath,
     "pred_world_fpath": pred_world_fpath
     },

    {"image_key": np.ones((1000, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
'fpath_key': "/data/samples/Pat_045.mha"
     },  # full path, a string

    {"image_key": np.ones((400, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": np.array([101]),  # not sure
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1200]),
     'level_key': np.array([2]),
'fpath_key': "/data/samples/Pat_045.mha",
     'corse_pred_in_img_key': np.array([501, 601, 701, 801, 901]),
     'corse_pred_in_img_1_key': np.array([601]),

     }  # full path, a string
]

TEST_CASE_3D_5Label_4 = [  # "rand_start": True, test shape
    {"level_node": 1, "train_on_level": 2, "height": 400, "rand_start": True, "start": 400,

     "data_fpath": data_fpath,
     "pred_world_fpath": pred_world_fpath
     },

    {"image_key": np.ones((1000, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
'fpath_key': "/data/samples/Pat_045.mha",
     },  # full path, a string

    {"image_key": np.ones((400, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": np.array([101]),  # not sure
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1200]),
     'level_key': np.array([2]),
'fpath_key': "/data/samples/Pat_045.mha",
     'corse_pred_in_img_key': np.array([501, 601, 701, 801, 901]),
     'corse_pred_in_img_1_key': np.array([601]),

     }  # full path, a string
]

TEST_CASE_3D_5Label_5 = [  # doesnot 'level_key' (Error)
    {"level_node": 1, "train_on_level": 0, "height": 400, "rand_start": True, "start": 400,

     "data_fpath": data_fpath,
     "pred_world_fpath": pred_world_fpath
     },

    {"image_key": np.ones((1000, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
'fpath_key': "/data/samples/Pat_045.mha",
     },  # full path, a string

    {"image_key": np.ones((400, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": np.array([101]),  # not sure
     "ori_label_in_img_key": np.array([500, 600, 700, 800, 900]),
     'ori_world_key': np.array([1000, 1200, 1400, 1600, 1800]),

     'world_key': np.array([1200]),
     'level_key': np.array([3]).reshape(-1, ),  # a random number
'fpath_key': "/data/samples/Pat_045.mha",
     'corse_pred_in_img_key': np.array([501, 601, 701, 801, 901]),
     'corse_pred_in_img_1_key': np.array([501]),

     }  # full path, a string
]


class TestCropCorseRegiond(unittest.TestCase):
    @parameterized.expand([TEST_CASE_3D_5Label_1, TEST_CASE_3D_5Label_2])
    def test_CropCorseRegiond(self, input_param, input_data, expected_out):
        result = CropCorseRegiond(**input_param)(input_data)
        self.assertEqual(set(list(result.keys())), set(list(expected_out.keys())))
        Compare().go(result, expected_out)

    @parameterized.expand([TEST_CASE_3D_5Label_3, TEST_CASE_3D_5Label_4])
    def test_CropCorseRegiond_shape(self, input_param, input_data, expected_out):
        result = CropCorseRegiond(**input_param)(input_data)
        self.assertEqual(set(list(result.keys())), set(list(expected_out.keys())))
        for k in result.keys():
            if type(expected_out[k]) is np.ndarray:
                self.assertEqual(result[k].shape, expected_out[k].shape)
            else:
                self.assertEqual(result[k], expected_out[k])

    @parameterized.expand([TEST_CASE_3D_5Label_4, TEST_CASE_3D_5Label_5])
    def test_CropCorseRegiond_key(self, input_param, input_data, expected_out):
        result = CropCorseRegiond(**input_param)(input_data)
        self.assertEqual(set(list(result.keys())), set(list(expected_out.keys())))
        self.assertTrue('level_key' in set(list(result.keys())))


if __name__ == "__main__":
    unittest.main()
    temp_dir.cleanup()
