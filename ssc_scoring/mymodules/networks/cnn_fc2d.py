# -*- coding: utf-8 -*-
# @Time    : 7/11/21 1:59 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import torch.nn as nn
import torchvision.models as models
from ssc_scoring.mymodules.summary import summary
from ssc_scoring.mymodules.networks.myconvnext import convnext_tiny, convnext_large, convnext_base
from ssc_scoring.mymodules.networks.myinception import inception_v3

import copy
from functools import partial
from torch.nn import functional as F
from torchvision.ops.misc import ConvNormActivation

class ReconNet(nn.Module):
    def __init__(self, reg_net, input_size=512):
        super().__init__()
        self.reg_net = reg_net
        self.features = copy.deepcopy(reg_net.features)  # encoder
        self.enc_dec = self._build_dec_from_enc()

    def _last_channels(self):
        last_chn = None
        for layer in self.features[::-1]:
            if isinstance(layer, torch.nn.Conv2d):
                last_chn = layer.out_channels
                break
        if last_chn is None:
            raise Exception("No convolution layers at all in regression network")
        return last_chn

    def _build_dec_from_enc(self):
        decoder_ls = []
        in_channels = None  # statement for convtransposed
        last_chns = self._last_channels()

        layer_shapes = summary(self.features, (1, 512, 512))  # ordered dict saving shapes of each layer
        transit_chn = 0
        for layer, (layer_name, layer_shape) in zip(self.features[::-1], reversed(layer_shapes.items())):
            if transit_chn == 0:
                transit_chn = layer_shape['input_shape'][1]

            if isinstance(layer, torch.nn.Conv2d):

                enc_in_channels = layer_shape['input_shape'][1]
                # enc_out_channels = layer_shape['output_shape'][1]

                enc_kernel_size: int = layer.kernel_size[0]  # square kernel, get one of the sizes
                enc_stride: int = layer.stride[0]

                if enc_stride > 1:  # shape is reduced
                    decoder_ls.append(nn.Upsample(scale_factor=enc_stride, mode='bilinear'))
                    decoder_ls.append(nn.Conv2d(transit_chn, enc_in_channels, enc_kernel_size,
                                                padding=enc_kernel_size - enc_kernel_size // 2 - 1))
                else:
                    decoder_ls.append(nn.Conv2d(transit_chn, enc_in_channels, enc_kernel_size,
                                                padding=enc_kernel_size - enc_kernel_size // 2 - 1))

                decoder_ls.extend([nn.BatchNorm2d(enc_in_channels),
                                   nn.ReLU(inplace=True)])

                transit_chn = enc_in_channels  # new value

            elif isinstance(layer, torch.nn.MaxPool2d):
                decoder_ls.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                decoder_ls.append(nn.Conv2d(transit_chn, transit_chn, 3, padding=1))
                decoder_ls.extend([nn.BatchNorm2d(transit_chn),
                                   nn.ReLU(inplace=True)])

            else:
                pass
        # correct the shape of the final output
        while (isinstance(decoder_ls[-1], nn.ReLU)) or (isinstance(decoder_ls[-1], nn.BatchNorm2d)):
            decoder_ls.pop()
        decoder = nn.Sequential(*decoder_ls)

        # class EncDoc(nn.Module):
        #     def __init__(self, enc, dec):
        #         super().__init__()
        #         self.enc = enc
        #         self.dec = dec
        #
        #     def forward(self, x):
        #         out = self.enc(x)
        #         out = self.dec(out)
        #         return out
        #
        # enc_dec = EncDoc(self.features, decoder)
        enc_dec = nn.Sequential(self.features, decoder)
        enc_dec_layer_shapes = summary(enc_dec, (1, 512, 512), device='cpu')
        input_sz = list(iter(enc_dec_layer_shapes.items()))[0][-1]['input_shape'][-1]
        output_sz = list(iter(enc_dec_layer_shapes.items()))[-1][-1]['input_shape'][-1]
        dif: int = input_sz - output_sz
        if dif > 0:  # the last output of decoder is less than the first output of encoder, need pad
            enc_dec = nn.Sequential(enc_dec,
                                    nn.Upsample(size=input_sz, mode="bilinear"),
                                    nn.Conv2d(1, 1, 3, padding=1))

        # decoder_dict = OrderedDict([(key, value) for key, value in zip(range(len(decoder_ls)), decoder_ls)])
        tmp = summary(enc_dec, (1, 512, 512), device='cpu')
        return enc_dec

    def forward(self, x):
        out = self.enc_dec(x)

        return out


class Cnn3fc1(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn2fc1_old(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class vgg11_HR(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class Cnn2fc1(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)

        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_net(name: str, nb_cls: int, args):
    if name == 'vgg11_HR':
        net = vgg11_HR(num_classes=nb_cls)
    elif 'vgg' in name:
        if name == 'vgg11_bn':
            net = models.vgg11_bn(pretrained=args.pretrained, progress=True)
        elif name == 'vgg16':
            net = models.vgg16(pretrained=args.pretrained, progress=True)
        elif name == 'vgg19':
            net = models.vgg19(pretrained=args.pretrained, progress=True)
        else:
            raise Exception("Wrong vgg net name specified ", name)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # change in_features to 1
        net.classifier[0] = torch.nn.Linear(in_features=512 * 7 * 7, out_features=args.fc1_nodes)
        net.classifier[3] = torch.nn.Linear(in_features=args.fc1_nodes, out_features=args.fc2_nodes)
        net.classifier[6] = torch.nn.Linear(in_features=args.fc2_nodes, out_features=3)
    elif name == 'alex':
        net = models.alexnet(pretrained=args.pretrained, progress=True)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        net.classifier[1] = torch.nn.Linear(in_features=256 * 6 * 6, out_features=args.fc1_nodes)
        net.classifier[4] = torch.nn.Linear(in_features=args.fc1_nodes, out_features=args.fc2_nodes)
        net.classifier[6] = torch.nn.Linear(in_features=args.fc2_nodes, out_features=3)
    elif name == 'cnn3fc1':
        net = Cnn3fc1(num_classes=nb_cls)
    elif name == 'cnn2fc1':
        net = Cnn2fc1(num_classes=nb_cls)
    elif name == 'squeezenet':
        net = models.squeezenet1_0(pretrained=args.pretrained)
        net.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2)
        final_conv = nn.Conv2d(512, nb_cls, kernel_size=1)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.3), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
    elif name == 'densenet161':
        net = models.densenet161(pretrained=args.pretrained)
        net.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
        net.classifier = nn.Linear(2208, nb_cls)
    elif name == 'inception_v3':
        net = models.inception_v3()
    elif name == 'mnasnet1_0':
        net = models.mnasnet1_0()
    elif name == 'shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0(pretrained=args.pretrained)
        net.conv1[0] = nn.Conv2d(1, 24, 3, 2, 1, bias=False)
        net.fc = nn.Linear(1024, nb_cls)
    elif 'convnext' in name:
        if name == 'convnext_tiny':
            net = convnext_tiny(pretrained=True)
            net.classifier[-1] = nn.Linear(768, nb_cls)
        if name == 'convnext_large':
            net = convnext_large(pretrained=True)
            net.classifier[-1] = nn.Linear(1536, nb_cls)
    elif name == 'inception':
        net = inception_v3(pretrained=args.pretrained)

    elif 'res' in name:
        if name == 'resnext50_32x4d':
            net = models.resnext50_32x4d(pretrained=args.pretrained, progress=True)
            net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, nb_cls)
        elif name == 'resnet18':
            net = models.resnet18(pretrained=args.pretrained, progress=True)
            net.fc = nn.Linear(512, nb_cls)
        elif name == 'wide_resnet50_2':
            net = models.wide_resnet50_2(pretrained=args.pretrained, progress=True)
            net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, nb_cls)

        elif name == 'resnext101_32x8d':
            net = models.resnext101_32x8d(pretrained=args.pretrained, progress=True)
            net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, nb_cls)
        else:
            raise Exception('Net name is not correct')
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise Exception("net name is wrong")

    return net
