# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
import tempfile
import os

from parameterized import parameterized
from ssc_scoring.mymodules.mytrans import AddChanneld
import numpy as np
from tests.utils import Compare


TEST_CASE_1 = [
    {"image_key": np.ones((30, 40, 50)).astype(np.float32)},  # this one will be changed

     {"image_key": np.ones((1, 30, 40, 50)).astype(np.float32)}, # this one will be changed
]


class TestAddChannelPosd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_AddChannelPosd(self, input_data, expected_out):
        with tempfile.TemporaryDirectory() as tempdir:
            result = AddChanneld()(input_data)
            Compare().go(result, expected_out)


if __name__ == "__main__":
    unittest.main()
