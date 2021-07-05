# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
from parameterized import parameterized
from ssc_scoring.mytrans import CropLevelRegiond
import numpy as np
from tests.utils import Compare

TEST_CASE_3D_5Label_1 = [
    {"level_node": 1, "train_on_level":1, "height": 200, "rand_start": False, "start": 400},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     # world position in mm, keep fixed,  a np.array with shape(-1, )
     'space_key': np.array([1, 2, 3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100, 200, 30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/abcd.mhd"},  # full path, a string

    {"image_key": np.ones((200, 256, 256)),
     "label_in_img_key": np.array([500]),
     "label_in_patch_key": np.array([100]),
     'world_key': np.array([1000]),
     # world position in mm, keep fixed,  a np.array with shape(-1, )
     'space_key': np.array([1, 2, 3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100, 200, 30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/abcd.mhd",
     'level_key': np.array([1]),
     }
]


TEST_CASE_3D_5Label_2 = [
    {"level_node": 0, "train_on_level":2, "height": 400, "rand_start": False, "start": 400},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     },  # full path, a string

    {"image_key": np.ones((400, 256, 256)),
     "label_in_img_key": np.array([600]),
     "label_in_patch_key": np.array([200]),  # not sure
     'world_key': np.array([1200]),
     'level_key': np.array([2]),
     }  # full path, a string
]

TEST_CASE_3D_5Label_3 = [  # rand start
    {"level_node": 0, "train_on_level":2, "height": 400, "rand_start": True, "start": 400},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     },  # full path, a string

    {"image_key": np.ones((400, 256, 256)),
     "label_in_img_key": np.array([600]),
     "label_in_patch_key": np.array([100]),  # not sure
     'world_key': np.array([1200]),
     'level_key': np.array([2]),
     }  # full path, a string
]


TEST_CASE_3D_5Label_4 = [  # level:0, output 'level_key'
    {"level": 0, "height": 400, "rand_start": True, "start": 400},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     },  # full path, a string

    {"image_key": np.ones((400, 256, 256)),
     "label_in_img_key": np.array([600]),
     "label_in_patch_key": np.array([100]),  # not sure
     'world_key': np.array([1200]),
     'level_key': np.array([2]),
     }  # full path, a string
]


TEST_CASE_3D_5Label_5 = [  # level:0, doesnot 'level_key' (Error)
    {"level": 0, "height": 400, "rand_start": True, "start": 400},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     },  # full path, a string

    {"image_key": np.ones((400, 256, 256)),
     "label_in_img_key": np.array([600]),
     "label_in_patch_key": np.array([100]),  # not sure
     'world_key': np.array([1200]),
     }  # full path, a string
]

class TestCropLevelRegiond(unittest.TestCase):
    @parameterized.expand([TEST_CASE_3D_5Label_1, TEST_CASE_3D_5Label_2])
    def test_CropLevelRegiond(self, input_param, input_data, expected_out):
        result = CropLevelRegiond(**input_param)(input_data)
        self.assertEqual(set(list(result.keys())), set(list(expected_out.keys())))
        Compare().go(result, expected_out)

    @parameterized.expand([TEST_CASE_3D_5Label_3, TEST_CASE_3D_5Label_4])
    def test_CropLevelRegiond_shape(self, input_param, input_data, expected_out):
        result = CropLevelRegiond(**input_param)(input_data)
        self.assertEqual(set(list(result.keys())), set(list(expected_out.keys())))
        for k in result.keys():
            if type(expected_out[k]) is np.ndarray:
                self.assertEqual(result[k].shape, expected_out[k].shape)
            else:
                self.assertEqual(result[k], expected_out[k])

    @parameterized.expand([TEST_CASE_3D_5Label_4, TEST_CASE_3D_5Label_5])
    def test_CropLevelRegiond_key(self, input_param, input_data, expected_out):
        result = CropLevelRegiond(**input_param)(input_data)
        self.assertEqual(set(list(result.keys())), set(list(expected_out.keys())))
        self.assertTrue('level_key' in set(list(result.keys())))



if __name__ == "__main__":
    unittest.main()
