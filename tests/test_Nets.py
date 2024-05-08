# -*- coding: utf-8 -*-
# @Time    : 7/1/21 2:49 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
from monai.networks import eval_mode
from tests.utils import test_script_save
import unittest
from parameterized import parameterized
from ssc_scoring.mymodules.networks.cnn_fc3d import Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_1 = [
    {"num_classes": 5, "base": 8, "level_node": 0},
    (2, 1, 192, 256, 256),
    (2, 5),
]

TEST_CASE_2 = [
    {"num_classes": 1, "base": 8, "level_node": 0},
    (2, 1, 192, 256, 256),
    (2, 1),
]

TEST_CASE_3 = [
    {"num_classes": 9, "base": 8, "level_node": 0},
    (2, 1, 192, 256, 256),
    (2, 9),
]

CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]

CASE_InLevel = {"input_param": {"num_classes": 9, "base": 8, "level_node": 3},
                "input_data": [torch.randn((2, 1, 192, 256, 256)).to(device), torch.tensor([[3], [3]]).to(device)],
                "expected_shape": (2, 9)}

class TestNets(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        for Net in [Cnn3fc1, Cnn3fc2, Cnn4fc2, Cnn5fc2, Cnn6fc2, Vgg11_3d]:
            net = Net(**input_param).to(device)
            with eval_mode(net):
                result = net.forward(torch.randn(input_shape).to(device))
                print(f'result.shape: {result.size()}')
                self.assertEqual(result.shape, expected_shape)



class TestVgg11_3d_InLevel(unittest.TestCase):
    def test_shape(self):
        net = Vgg11_3d(num_classes=9, base=8, level_node=3).to(device)
        with eval_mode(net):
            data = [torch.randn((2, 1, 192, 256, 256)).to(device), torch.tensor([[3], [3]]).to(device)]
            result = net.forward(data)
            self.assertEqual(result.shape, (2,9))



if __name__ == "__main__":
    unittest.main()
