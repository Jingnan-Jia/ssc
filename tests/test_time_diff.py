# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
from parameterized import parameterized
import datetime
from ssc_scoring.mymodules.tool import time_diff

TEST_CASE_1 = ['1:02:34']


class TestTimeDiff(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_time_diff(self, expected_out):
        f = "%Y-%m-%d %H:%M:%S"
        t1 = datetime.datetime.strptime('2021-05-03 12:34:01', f)
        t2 = datetime.datetime.strptime('2021-05-03 13:36:35', f)
        elapsed_time = time_diff(t1, t2)
        self.assertEqual(elapsed_time, expected_out)


if __name__ == "__main__":
    unittest.main()