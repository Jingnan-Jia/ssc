# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
from parameterized import parameterized
from ssc_scoring.run_pos import CropLevelRegiond
import numpy as np

TEST_CASE_3D_5Label_1 = [
    {"level": 1, "height": 200, "rand_start": False, "start": 400},

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
     'fpath_key': "/data/samples/abcd.mhd"}  # full path, a string
]


TEST_CASE_3D_5Label_2 = [
    {"level": 2, "height": 400, "rand_start": False, "start": 400},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     },  # full path, a string

    {"image_key": np.ones((400, 256, 256)),
     "label_in_img_key": np.array([600]),
     "label_in_patch_key": np.array([200]),  # not sure
     'world_key': np.array([1200]),
     }  # full path, a string
]

TEST_CASE_3D_5Label_3 = [  # rand start
    {"level": 2, "height": 400, "rand_start": True, "start": 400},

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
        for k1, k2 in zip(result, expected_out):
            if type(expected_out[k2]) is np.ndarray:
                self.assertIsNone(np.testing.assert_array_equal(result[k1], expected_out[k2]))
            else:
                self.assertEqual(result[k1], expected_out[k2])

    @parameterized.expand([TEST_CASE_3D_5Label_3])
    def test_CropLevelRegiond_shape(self, input_param, input_data, expected_out):
        result = CropLevelRegiond(**input_param)(input_data)
        for k1, k2 in zip(result, expected_out):
            if type(expected_out[k2]) is np.ndarray:
                self.assertEqual(result[k1].shape, expected_out[k2].shape)
            else:
                self.assertEqual(result[k1], expected_out[k2])


if __name__ == "__main__":
    unittest.main()
