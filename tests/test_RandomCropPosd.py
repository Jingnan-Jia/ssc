# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
from parameterized import parameterized
import datetime
from ssc_scoring.mymodules.mytrans import RandomCropPosd
import numpy as np
from tests.utils import Compare

TEST_CASE_3D_5Label_Upper = [
    {"z_size": 400, "y_size": 20, "x_size": 20},

    {"image_key": np.ones((1000, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),  # world position in mm, keep fixed,  a np.array with shape(-1, )
     'space_key': np.array([1,2,3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100,200,30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/abcd.mhd"},  # full path, a string

    {"image_key": np.ones((400, 20, 20)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": np.array([200, 300, 400, 400, 400]),
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),  # world position in mm, keep fixed,  a np.array with shape(-1, )
     'space_key': np.array([1, 2, 3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100, 200, 30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/abcd.mhd"}  # full path, a string
]


class TestRandomCropPosd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_3D_5Label_Upper])
    def test_RandomCropPosd_shape(self, input_param, input_data, expected_out):
        result = RandomCropPosd(**input_param)(input_data)
        self.assertEqual(set(result.keys()), set(expected_out.keys()))
        self.assertTrue(result['image_key'].shape, expected_out['image_key'].shape)

if __name__ == "__main__":
    unittest.main()