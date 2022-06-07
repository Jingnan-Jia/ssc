# -*- coding: utf-8 -*-
# @Time    : 7/10/21 1:21 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from ssc_scoring.mymodules.mytrans import RandGaussianNoised

import unittest
import tempfile
import os

from parameterized import parameterized
import numpy as np


TEST_CASE_1 = [
    {"image_key": np.array([[0, 1], [1, 0]]).astype(np.float32)},  # this one will be changed

     {"image_key": np.array([[0, 1], [1, 0]]).astype(np.float32)}, # this one will be changed
]


class TestRandGaussianNoisePosd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_RandGaussianNoisePosd_shape(self, input_data, expected_out):
        result = RandGaussianNoised()(input_data)
        self.assertEqual(set(result.keys()), set(expected_out.keys()))
        self.assertTrue(result['image_key'].shape, expected_out['image_key'].shape)

if __name__ == "__main__":
    unittest.main()
