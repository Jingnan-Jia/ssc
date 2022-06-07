# -*- coding: utf-8 -*-
# @Time    : 7/10/21 1:21 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from ssc_scoring.mymodules.mytrans import NormImgPosd

import unittest
import tempfile
import os

from parameterized import parameterized
import numpy as np
from tests.utils import Compare


TEST_CASE_1 = [
    {"image_key": np.array([[0, 1], [1, 0]]).astype(np.float32)},  # this one will be changed

     {"image_key": np.array([[-1, 1], [1, -1]]).astype(np.float32)}, # this one will be changed
]


class TestNormImgPosd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_NormImgPosd(self, input_data, expected_out):
        result = NormImgPosd()(input_data)
        Compare().go(result, expected_out)


if __name__ == "__main__":
    unittest.main()
