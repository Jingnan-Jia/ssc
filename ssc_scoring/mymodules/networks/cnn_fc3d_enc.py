# -*- coding: utf-8 -*-
# @Time    : 7/4/21 9:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import torch.nn as nn

class Cnn3fc1Enc(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8, level_node = 0):
        super().__init__()
        self.level_node = level_node
        self.features = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),

        )

    def forward(self, input):
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        return x


class Cnn3fc2Enc(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8, level_node = 0):
        super().__init__()
        self.level_node = level_node
        self.features = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )


    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]

        x = self.features(x)
        return x


class Cnn4fc2Enc(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8, level_node = 0):
        super().__init__()
        self.level_node = level_node
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        return x


class Cnn5fc2Enc(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8, level_node = 0):
        super().__init__()
        self.level_node = level_node
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        return x


class Cnn6fc2Enc(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8, level_node = 0):
        super().__init__()
        self.level_node = level_node
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 16, base * 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 32),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        return x


class Vgg11_3dEnc(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8, level_node = 0):
        super().__init__()
        self.num_classes = num_classes
        self.base = base
        self.level_node = level_node

        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.Conv3d(base * 16, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),)

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        return x


