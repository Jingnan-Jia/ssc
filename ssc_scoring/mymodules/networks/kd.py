# -*- coding: utf-8 -*-
# @Time    : 7/26/21 10:49 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import torch.nn as nn
from ssc_scoring.mymodules.networks.pre_trained_net import generate_model
from ssc_scoring.mymodules.networks.unet3D import UNet3D
import os

class Option:
    def __init__(self):
        raise NotImplementedError


def download_weights(weights_dir):
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    # download weights and unzip weights
    package = "googledrivedownloader"
    try:
        __import__(package)
    except:
        os.system("pip install " + package)

    from google_drive_downloader import GoogleDriveDownloader as gdd
    gdd.download_file_from_google_drive(file_id='1vYvIamI4zWUTD7TGgatxwkMAwqTCpd_B',
                                        dest_path='./' + weights_dir + '/resnet_18_23dataset.pth')
    gdd.download_file_from_google_drive(file_id='1estlxOizcJmNTa85zOeyzjQ7-0rzsdK8',
                                        dest_path='./' + weights_dir + '/resnet_34_23dataset.pth')
    gdd.download_file_from_google_drive(file_id='18FQtxbbvGWlBXk8bGE7MphV7K0LMMhD0',
                                        dest_path='./' + weights_dir + '/resnet_50_23dataset.pth')
    gdd.download_file_from_google_drive(file_id='1izVHemAsKPGu3RvtyzmBJE46C2Qo-OF3',
                                        dest_path='./' + weights_dir + '/resnet_101.pth')
    gdd.download_file_from_google_drive(file_id='1yqIIM_PNin8lNeWlxGH_Qs5sjLkPqjSb',
                                        dest_path='./' + weights_dir + '/resnet_152.pth')
    gdd.download_file_from_google_drive(file_id='1FWtQqyuOUjf4zk-WbACrj5g4AnuSb_8Z',
                                        dest_path='./' + weights_dir + '/resnet_200.pth')
    gdd.download_file_from_google_drive(file_id='192k-iKbvDpnPc1j5Da88Ogrt11JsikVp',
                                        dest_path='./' + weights_dir + '/Genesis_Chest_CT.pt')
    return None



def get_enc_t(net_name):
    """
    Availabel net_name:
    Med3D (https://arxiv.org/pdf/1904.00625.pdf):
        resnet3d_10, resnet3d_18, resnet3d_34, resnet3d_50, resnet3d_101, resnet3d_152, resnet3d_200
    Model Genesis (https://arxiv.org/pdf/1908.06912.pdf):
        unet3d
        resnet2d_18  # I do not have its pre-trained weights yet, because it is not released.

opt: require following values: model, model_depth, input_W, input_H, input_D,
    resnet_shortcut, no_cuda, n_seg_classes, phase, pretrain_path

    :param net_name:
    :return:
    """
    weights_dir = 'pretrained_weights'
    download_weights(weights_dir)

    opt = Option()
    if 'resnet3d' in net_name:
        opt.model = 'resnet'  # do not change it
        opt.model_depth = int(net_name.split('_')[-1])
        if opt.model_depth in [18, 34]:  # https://github.com/Tencent/MedicalNet
            opt.resnet_shortcut = 'A'
        else:
            opt.resnet_shortcut = 'B'
        opt.no_cuda = True  # I think it does not matter
        opt.n_seg_classes = 2  # I think it does not matter
        opt.pretrain_path = weights_dir + "/resnet_" + str(opt.model_depth) + ".pth"  # reset it if needed

        enc = generate_model(opt)  # generate resnet encoder
        enc.load_state_dict(torch.load(opt.pretrain_path))

    elif 'resnet2d' in net_name:
        raise NotImplementedError
    elif 'unet3d' == net_name:
        weights_fpath = weights_dir + '/Genesis_Chest_CT.pt'
        base_model = UNet3D()

        # Load pre-trained weights
        checkpoint = torch.load(weights_fpath)
        state_dict = checkpoint['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        base_model.load_state_dict(unParalled_state_dict)
        enc = base_model.out512  # encoder
    else:
        raise Exception(f'wrong net name {net_name}')
    return enc
