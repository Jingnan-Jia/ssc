# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
from parameterized import parameterized
from ssc_scoring.run_pos import LoadDatad
import numpy as np
import myutil.myutil as futil

TEST_CASE_Error = [
    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([500, 600, 700, 800, 900]),
     "label_in_patch_key": None,
     'world_key': np.array([1000, 1200, 1400, 1600, 1800]),
     # world position in mm, keep fixed,  a np.array with shape(-1, )
     'space_key': np.array([1, 2, 3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100, 200, 30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/abcd.mhd"},  # full path, a string

]


TEST_CASE_1 = [
    {'world_key': np.array([1400]),
     'fpath_key': "./tests/data/abcd.mhd"},

    {"image_key": np.ones((30, 40, 50)),
     "label_in_img_key": np.array([800]),
     "label_in_patch_key": np.array([800]),  # not sure
     'world_key': np.array([1400]),
     'space_key': np.array([0.5, 0.3, 0.3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-1000,2,3]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "./tests/data/abcd.mhd"}  # full path, a string
]


class TestLoadDatad(unittest.TestCase):
    @parameterized.expand([TEST_CASE_Error])
    def test_LoadDatad_InputError(self, input_data):
        self.assertNotEqual(set(input_data.keys()), {'fpath_key', 'world_key'})


    @parameterized.expand([TEST_CASE_1])
    def test_LoadDatad(self, input_data, expected_out):
        ts_data = np.ones((30, 40, 50))
        futil.save_itk(filename=input_data["fpath_key"], scan=ts_data, origin=(-1000,2,3), spacing=(0.5, 0.3, 0.3))
        result = LoadDatad()(input_data)
        for k1, k2 in zip(result, expected_out):
            if type(expected_out[k2]) is np.ndarray:
                self.assertEqual(result[k1].shape, expected_out[k2].shape)
            else:
                self.assertEqual(result[k1], expected_out[k2])


if __name__ == "__main__":
    unittest.main()
