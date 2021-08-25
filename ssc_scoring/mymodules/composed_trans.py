# -*- coding: utf-8 -*-
# @Time    : 7/11/21 2:36 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import monai
from torchvision import transforms
from argparse import Namespace
from ssc_scoring.mymodules.data_synthesis import SysthesisNewSampled
from ssc_scoring.mymodules.mytrans import RandomAffined, RandomHorizontalFlipd, RandomVerticalFlipd, \
    RandGaussianNoised, LoadDatad, NormImgPosd, AddChanneld, RandomCropPosd, \
    CenterCropPosd, RandCropLevelRegiond, CoresPosd, SliceFromCorsePosd, RescaleToNeg1500Pos1500d
from ssc_scoring.mymodules.path import PathScore


def xformd_score(mode: str = 'train', synthesis: bool = False, args: Namespace = None) -> monai.transforms.Compose:

    """ Transforms for Goh score  prediction.
    The input image voxel values have to be from 0 to 1 for `SysthesisNewSampled`.

    :param mode: Selected from 'train', 'valid', 'validaug', and 'test'.
    :param synthesis: If using snthesis data.
    :param args:
    :return:

    Example:

    >>> args = ssc_scoring.mymodules.set_args()
    >>> xformd_score(mode='train', synthesis=False, args=args)

    One use case is :meth:`ssc_scoring.mymodules.mydata.LoadScore.xformd`.

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
                                              retp_fpath="/data/jjia/ssc_scoring/ssc_scoring/dataset/special_samples/ret/ret.mha",
                                              gg_fpath="/data/jjia/ssc_scoring/ssc_scoring/dataset/special_samples/gg/gg.mha",
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
    xforms.append(RescaleToNeg1500Pos1500d())

    transform = transforms.Compose(xforms)

    return transform


def xformd_pos2score(mode: str, mypath: PathScore) -> monai.transforms.Compose:
    """Composed transforms to obtain 2D slices given 3D image and slice number.
    It is used to evaluate the cascaded network (PositionPredictionNet + GohScorePredictionNet)s.

    Detailed steps:

    #. Load data
    #. Extract the predicted slice number from the results of PositionPredictionNet.
    #. Get the 2D slices given 3D image (step 1) and slice number (step 2).
    #. Add batch channel.

    :param mode: Selected from 'train', 'valid', 'validaug', and 'test'.
    :param mypath: Instance of PathScore.
    :return:

    Example:

    >>> xformd_pos2score(mode = "valid", mypath = PathScore(id=1405))

    Its use case is :meth:`ssc_scoring.mymodules.mydata.LoadPos2Score.xformd`.

    """
    xforms = [LoadDatad(),
              CoresPosd(corse_fpath=mypath.pred_int(mode), data_fpath=mypath.data(mode)),
              SliceFromCorsePosd(),
              AddChanneld()]

    transform = monai.transforms.Compose(xforms)

    return transform


def recon_transformd(mode: str = 'train'):
    """Transforms for reconstruction network.

    .. Warning::
        This function is not complete. Please double check its source code before using it.

    :param mode: 'train', 'valid', 'validaug', 'test'.
    :return: Composed transforms.

    """

    keys = "image_key"  # only transform image
    xforms = [AddChanneld()]
    if mode == 'train':
        xforms.extend([RandomHorizontalFlipd(keys), RandomVerticalFlipd(keys)])
    xforms.append(NormImgPosd())
    xforms = transforms.Compose(xforms)
    return xforms


def xformd_pos(mode: str = 'train', level_node: int = 0, train_on_level: int = 0,
               z_size: int = 192, y_size: int = 256, x_size: int = 256) -> monai.transforms.Compose():
    """ Transforms for position prediction.
    Detailed steps:

    #. Load 3D CT images.
    #. If `train_on_level` or `level_node`, crop to fixed size (level_posiion - patch_size, level_posiion + patch_size).
    #. Crop (random or center) to fixed patch size.
    #. Normalization.
    #. Add batch channel.

    .. note::
        The position prediction results from networks based on 256, 256, 256 patches may be more accurate if we have
        another network to refine these results. I hope using high-resolution patches can lead to better results. But
        if we still use the field of view which includes 5 levels, the patch size would be bigger than the GPU can fit.
        So I proposed to input a high-resolution patch which include only one level. In order to make the networks know
        which level the input patch is, I proposed 2 methods:

            #. 5 seperate networks are trained independently on the patches which include 5 different level positions.
            The user is responsible to ensure the input patches are from the same level as the networks.
            #. 1 network with an extra node to receive the level information (1,2,3,4,5) are trained. The user is
            responsible to ensure the level information is the same as the input patches.


        `level_node` is specified when your network has extra input node for level information apart the normal input
        node for images.

        `train_on_level` is specified when you want your network to output only one level. Then the transform will
         crop a 3D region in which this level must be visible.


    :param mode: 'train', 'valid', 'validaug', 'test'.
    :param level_node: which level is received by the network if it is required.
    :param train_on_level: in which level the network want to be trained if it is required.
    :param z_size: patch size along z axial
    :param y_size: patch size along y axial
    :param x_size: patch size along x axial
    :return: Composed transforms.
    """
    xforms = [LoadDatad()]
    if level_node or train_on_level:
        xforms.append(RandCropLevelRegiond(level_node, train_on_level, height=z_size, rand_start=True))
    else:
        # pass
        if mode == 'train':
            # xforms.extend([RandomCropPosd(), RandGaussianNoised()])
            # pass
            xforms.extend([RandomCropPosd(z_size=z_size, y_size=y_size, x_size=x_size)])

        else:
            xforms.extend([CenterCropPosd(z_size=z_size, y_size=y_size, x_size=x_size)])
        xforms.append(NormImgPosd())

    xforms.extend([AddChanneld()])
    # xforms.extend([CastToTyped(keys = ('image_key'), dtype=('np.float32'))])
    transform = monai.transforms.Compose(xforms)

    return transform


