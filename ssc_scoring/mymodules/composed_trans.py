# -*- coding: utf-8 -*-
# @Time    : 7/11/21 2:36 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

from torchvision import transforms
import monai

from ssc_scoring.mymodules.mytrans import RandomAffined, RandomHorizontalFlipd, RandomVerticalFlipd, \
    RandGaussianNoised, LoadDatad, NormImgPosd, AddChanneld, RandomCropPosd,\
    CenterCropPosd, CropLevelRegiond,CoresPos, SliceFromCorsePos, NormNeg1To1d, RescaleToNeg1500Pos1500
from monai.transforms import CastToTyped

from ssc_scoring.mymodules.data_synthesis import SysthesisNewSampled

def xformd_score(mode='train', synthesis=False, args=None):
    """
    The input image data is from 0 to 1.
    :param mode:
    :return:
    """
    keys = "image_key"
    rotation = 90
    image_size = 512
    vertflip = 0.5
    horiflip = 0.5
    shift = 10 / 512
    scale = 0.05

    xforms = []
    if mode in ['train', 'validaug']:
        if synthesis:
            xforms.append(SysthesisNewSampled(keys=keys,
                                              retp_fpath="/data/jjia/ssc_scoring/ssc_scoring/dataset/special_samples/retp.mha",
                                              gg_fpath="/data/jjia/ssc_scoring/ssc_scoring/dataset/special_samples/gg.mha",
                                              mode=mode, sys_pro_in_0=args.sys_pro_in_0,
                 retp_blur=args.retp_blur,
                 gg_blur=args.gg_blur,
                 sampler=args.sampler,
                 gen_gg_as_retp=args.gen_gg_as_retp,
                 gg_increase=args.gg_increase))
        xforms.extend([
            AddChanneld(),
            RandomAffined(keys=keys, degrees=rotation, translate=(shift, shift), scale=(1 - scale, 1 + scale)),
            # CenterCropd(image_size),
            RandomHorizontalFlipd(keys, p=horiflip),
            RandomVerticalFlipd(keys, p=vertflip),
            RandGaussianNoised()
        ])
    else:
        xforms.extend([AddChanneld()])

    # xforms.append(NormImgPosd())
    # xforms.append(NormNeg1To1d())
    xforms.append(RescaleToNeg1500Pos1500())




    transform = transforms.Compose(xforms)

    return transform



def xformd_pos2score(mode, mypath):
    xforms = [LoadDatad(),
              CoresPos(corse_fpath=mypath.pred_int(mode), data_fpath=mypath.data(mode)),
              SliceFromCorsePos(),
              AddChanneld()]

    transform = monai.transforms.Compose(xforms)

    return transform


def recon_transformd(mode='train'):
    keys = "image_key"  # only transform image
    xforms = [AddChanneld()]
    if mode=='train':
        xforms.extend([RandomHorizontalFlipd(keys), RandomVerticalFlipd(keys)])
    xforms.append(NormImgPosd())
    xforms = transforms.Compose(xforms)
    return xforms


def xformd_pos(mode=None, level_node=0, train_on_level=0, z_size=192, y_size=256, x_size=256):
    xforms = [LoadDatad()]
    if level_node or train_on_level:
        xforms.append(CropLevelRegiond(level_node, train_on_level, height=z_size, rand_start=True))
    else:
        if mode == 'train':
            # xforms.extend([RandomCropPosd(), RandGaussianNoised()])
            xforms.extend([RandomCropPosd(z_size=z_size, y_size=y_size, x_size=x_size)])

        else:
            xforms.extend([CenterCropPosd(z_size=z_size, y_size=y_size, x_size=x_size)])
        xforms.append(NormImgPosd())

    xforms.extend([AddChanneld()])
    # xforms.extend([CastToTyped(keys = ('image_key'), dtype=('np.float32'))])
    transform = monai.transforms.Compose(xforms)

    return transform


