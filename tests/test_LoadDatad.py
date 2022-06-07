# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
import tempfile
import os

from parameterized import parameterized
from ssc_scoring.mymodules.mytrans import LoadDatad
import numpy as np
import medutils.medutils as futil
from tests.utils import Compare

TEST_CASE_Error = [
    {"image_key": np.ones((100, 20, 20)),
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

    {"image_key": np.ones((30, 40, 50)).astype(np.float32),
     "label_in_img_key": np.array([4800.]),
     "label_in_patch_key": np.array([4800.]),  # not sure
     'ori_label_in_img_key': np.array([4800.]),
     'world_key': np.array([1400.]),
     'ori_world_key': np.array([1400.]),
     'space_key': np.array([0.5, 0.3, 0.3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-1000., 2., 3.]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "./tests/data/abcd.mhd"}  # full path, a string

]


class TestLoadDatad(unittest.TestCase):
    @parameterized.expand([TEST_CASE_Error])
    def test_LoadDatad_InputError(self, input_data):
        self.assertNotEqual(set(input_data.keys()), {'fpath_key', 'world_key'})

    @parameterized.expand([TEST_CASE_1])
    def test_LoadDatad(self, input_data, expected_out):
        ts_data = np.ones((30, 40, 50))  # z, y, z
        with tempfile.TemporaryDirectory() as tempdir:
            input_data['fpath_key'] = os.path.join(tempdir, os.path.basename(input_data["fpath_key"]))
            futil.save_itk(filename=input_data["fpath_key"],
                           scan=ts_data,
                           origin=(-1000, 2, 3),
                           spacing=(0.5, 0.3, 0.3))
            result = LoadDatad()(input_data)
            Compare().go(result, expected_out)


if __name__ == "__main__":
    unittest.main()
