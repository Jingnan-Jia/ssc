# -*- coding: utf-8 -*-
# @Time    : 7/4/21 9:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import torch.nn as nn

class Cnn3fc1(nn.Module):
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
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 4 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, num_classes),
        )

    def forward(self, input):
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn3fc2(nn.Module):
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
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 4 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc2_nodes, num_classes),
        )


    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn4fc2(nn.Module):
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
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 8 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc2_nodes, num_classes),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn5fc2(nn.Module):
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
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 16 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc2_nodes, num_classes),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn6fc2(nn.Module):
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
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 32 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc2_nodes, num_classes),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Vgg11_3d(nn.Module):
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

        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.ft = nn.Flatten()
        self.dp1 = nn.Dropout()

        if self.level_node != 0:
            nb_fc0 = base * 16 * 6 * 6 * 6 + 1
        else:
            nb_fc0 = base * 16 * 6 * 6 * 6

        self.ln1 = nn.Linear(nb_fc0, fc1_nodes)
        self.rl1 = nn.ReLU(inplace=True)

        self.dp2 = nn.Dropout()
        self.ln2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.rl2 = nn.ReLU(inplace=True)

        self.dp3 = nn.Dropout()
        self.ln3 = nn.Linear(fc2_nodes, self.num_classes)


    def _fc_first(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.ft(x)
        x = self.dp1(x)
        return x

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
            # print(f'x.shape', x.size())

            x = self._fc_first(x)
        else:
            x, level = input[0], input[1]
            print(f'x.shape', x.size())
            print(f'level.shape', level.size())

            x = self._fc_first(x)
            x = torch.cat((x, level), 1)

        x = self.ln1(x)
        x = self.rl1(x)

        x = self.ln2(x)
        x = self.dp2(x)
        x = self.rl2(x)

        x = self.dp3(x)
        x = self.ln3(x)

        return x


