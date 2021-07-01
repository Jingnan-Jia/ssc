# -*- coding: utf-8 -*-
# @Time    : 7/1/21 2:49 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
from monai.networks import eval_mode
from tests.utils import test_script_save
import unittest
from parameterized import parameterized
from ssc_scoring.run_pos import Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_1 = [
    {"num_classes": 5, "base": 8},
    (4, 1, 192, 256, 256),
    (4, 5),
]

TEST_CASE_2 = [
    {"num_classes": 1, "base": 8},
    (4, 1, 192, 256, 256),
    (4, 1),
]

TEST_CASE_3 = [
    {"num_classes": 9, "base": 8},
    (4, 1, 192, 256, 256),
    (4, 9),
]

CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]

class TestNets(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        for Net in [Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d]:
            net = Net(**input_param).to(device)
            with eval_mode(net):
                result = net.forward(torch.randn(input_shape).to(device))
                self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        test_data = torch.randn(2, 1, 512, 256, 256)
        for Net in [Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d]:
            net = Net(num_classes=8).to(device)
            test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
