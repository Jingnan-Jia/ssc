# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
from parameterized import parameterized
import datetime
from ssc_scoring.run_pos import CenterCropPosd
import numpy as np


TEST_CASE_3D_1Label_Normal = [  # label_in_patch 500 -> 200, others keep unchanged
    {"z_size": 400, "y_size": 256, "x_size": 256},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([500]),
     "label_in_patch_key": None,
     'world_key': np.array([6789.234]),  # world position in mm, keep fixed,  a np.array with shape(-1, )
     'space_key': np.array([1,2,3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100,200,30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/abcd.mhd"},  # full path, a string

    {"image_key": np.ones((400, 256, 256)),
     "label_in_img_key": np.array([500]),
     "label_in_patch_key": np.array([200]),
     'world_key': np.array([6789.234]),  # world position in mm, keep fixed,  a np.array with shape(-1, )
     'space_key': np.array([1, 2, 3]),  # space,  a np.array with shape(-1, )
     'origin_key': np.array([-100, 200, 30]),  # origin,  a np.array with shape(-1, )
     'fpath_key': "/data/samples/abcd.mhd"}  # full path, a string
]


TEST_CASE_3D_1Label_Upper = [  # label_in_patch 500 -> 0, others keep unchanged
    {"z_size": 400, "y_size": 256, "x_size": 256},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([800]),
     "label_in_patch_key": None,
     },  # full path, a string

    {"image_key": np.ones((400, 256, 256)),
     "label_in_img_key": np.array([800]),
     "label_in_patch_key": np.array([400]),
     }  # full path, a string
]


TEST_CASE_3D_1Label_Lower = [  # label_in_patch 500 -> 0, others keep unchanged
    {"z_size": 100, "y_size": 256, "x_size": 256},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([200]),
     "label_in_patch_key": None,
     },  # full path, a string

    {"image_key": np.ones((100, 256, 256)),
     "label_in_img_key": np.array([200]),
     "label_in_patch_key": np.array([0]),
     }  # full path, a string
]

TEST_CASE_3D_1Label_Error = [  #
    {"z_size": 100, "y_size": 256, "x_size": 256},

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([200]),
     "label_in_patch_key": None,
     },  # full path, a string

    {"image_key": np.ones((1000, 256, 256)),
     "label_in_img_key": np.array([200]),
     "label_in_patch_key": np.array([399]),
     }  # full path, a string
]

class TestCenterCropPosd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_3D_1Label_Normal, TEST_CASE_3D_1Label_Upper])
    def test_CenterCropPosd(self, input_param, input_data, expected_out):
        result = CenterCropPosd(**input_param)(input_data)
        # self.assertEqual(result, expected_out)
        for k1, k2 in zip(result, expected_out):
            if type(expected_out[k2]) is np.ndarray:
                self.assertIsNone(np.testing.assert_array_equal(result[k1], expected_out[k2]))
            else:
                self.assertEqual(result[k1], expected_out[k2])

    @parameterized.expand([TEST_CASE_3D_1Label_Error])
    def test_CenterCropPosdError(self, input_param, input_data, expected_out):
        result = CenterCropPosd(**input_param)(input_data)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 result["label_in_patch_key"], expected_out["label_in_patch_key"])

        # self.assertIsNotNone(np.testing.assert_array_equal(result["label_in_patch_key"], expected_out["label_in_patch_key"]))



if __name__ == "__main__":
    unittest.main()