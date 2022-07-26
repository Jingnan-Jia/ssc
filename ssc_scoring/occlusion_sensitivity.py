# -*- coding: utf-8 -*-
# @Time    : 3/28/21 2:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import sys
sys.path.append("..")
from monai.transforms import ScaleIntensityRange
from tqdm import tqdm
import copy
import math
import time
from ssc_scoring.mymodules.mydata import LoadScore
from ssc_scoring.mymodules.set_args import get_args
from ssc_scoring.mymodules.data_synthesis import savefig
from ssc_scoring.mymodules.colormap import get_continuous_cmap

from scipy.ndimage import morphology
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from ssc_scoring.run import get_net, Path
from torch.utils.data import Dataset, DataLoader
from medutils.medutils import load_itk

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

grad_block = []
fmap_block = []
# 图片预处理
def img_preprocess(img):
    img_out = (img - np.mean(img)) / np.std(img)  # normalize
    img_out = img_out[None]
    img_out = torch.as_tensor(img_out)

    return img_out


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


def apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):

    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)


def cam_show_img(img, feature_map, grads, out_dir, idx):
    img = img.cpu().numpy()
    _, __, H, W = img.shape  # (1,1,512,512)
    img = np.resize(img, (H, W, 1))
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_jpg = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

    cam_img = 0.3 * heatmap + 0.7 * img_jpg
    heatmap_img = os.path.join(out_dir, str(idx) + "_cam.jpg")
    path_cam_img = os.path.join(out_dir, str(idx) + "_img_cam.jpg")
    path_img = os.path.join(out_dir, str(idx) + "_img.jpg")

    cv2.imwrite(heatmap_img, heatmap)
    cv2.imwrite(path_cam_img, cam_img)
    cv2.imwrite(path_img, img_jpg)


def grad_cam(x, y, net, nb_img):
    # fmap_block = []
    # grad_block = []
    x = x[None]
    y = y[None]
    print(y.cpu().numpy())

    # 注册hook, vgg has features and classifiers
    net.features[3].register_forward_hook(farward_hook)
    net.features[3].register_backward_hook(backward_hook)

    # forward
    output = net(x)

    # backward
    net.zero_grad()
    loss_fun = torch.nn.MSELoss()
    class_loss = loss_fun(output, y)
    class_loss.backward()

    # 生成cam
    print(len(grad_block))
    grads_val = grad_block[0].cpu().data.cpu().numpy().squeeze()
    fmap = fmap_block[0].cpu().data.cpu().numpy().squeeze()
    cam_show_img(x, fmap, grads_val, mypath.id_dir, nb_img)


def generate_candidate(fpath: str, image_size: int = 512):
    """

    Args:
        fpath: full path of seed patch

    Returns:
        Filled image, shape: [512, 512]
    """
    ori_image_fpath = fpath.split('.mha')[0] + '_ori.mha'
    egg = load_itk(fpath)
    # ori = load_itk(ori_image_fpath)
    normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
    egg = normalize0to1(egg)
    # egg[egg > 1500] = 1500
    # egg[egg < -1500] = -1500
    # ori[ori > 1500] = 1500
    # ori[ori < -1500] = -1500
    # # normalize the egg using the original image information
    # egg = (egg - np.min(ori)) / (np.max(ori) - np.min(ori))  # rescale to [0, 1]

    minnorv = np.vstack((np.flip(egg), np.flip(egg, 0)))
    minnorh = np.hstack((minnorv, np.flip(minnorv, 1)))

    cell_size = minnorh.shape
    nb_row, nb_col = image_size // cell_size[0] * 2, image_size // cell_size[
        1] * 2  # big mask for crop
    temp = np.hstack(([minnorh] * nb_col))
    temp = np.vstack(([temp] * nb_row))
    temp = temp[:image_size, :image_size]
    return temp


def occlusion_map(patch_size, x, y, net, lung_mask=None, occlusion_dir=None, save_occ_x=False, stride=None,occ_status='healthy',
                  map_2_w=None):  # for one image
    """Save occlusion map to disk.

    Args:
        patch_size: patch side lenth
        x: image to be predicted, shape [channel, w, h]
        y: predicted scores, shape [1, 3]
        net: network
        lung_mask: lung mask to ensure the occlusion occurs in lung area, shape [channel, w, h]
        occlusion_dir: directory to save occlusion maps

    Returns:
        None
    """
    if not os.path.isdir(occlusion_dir):
        os.makedirs(occlusion_dir)


    # lung_mask = morphology.binary_erosion(lung_mask.numpy(), np.ones((6, 6))).astype(int)
    lung_mask = lung_mask.numpy()
    lung_mask[lung_mask > 0] = 1
    lung_mask[lung_mask <= 0] = 0
    np.save(os.path.join(occlusion_dir, f"lung_mask.npy"), lung_mask)
    x = x.to(device)  # shape [channel, w, h]
    net.to(device)
    x_ = x.unsqueeze(0)
    out_ori = net(x_)
    _, w, h = x.shape
    # print(x.shape)  # [1, 512, 512] [channel, w, h]
    # print(out_ori.shape)  # [1, 3]

    # Three-pattern scores
    out_ori = out_ori.detach().cpu().numpy()
    out_ori_1 = out_ori[0, 0]  # tot score
    out_ori_2 = out_ori[0, 1]  # gg score
    out_ori_3 = out_ori[0, 2]  # ret score

    # out_ori_2 = out_ori[0][1].item()
    # out_ori_3 = out_ori[0][2].item()
    # print(out_ori_2)

    x_np = x.clone().detach().cpu().numpy()  # shape [channel, w, h]

    # y_np = y.clone().detach()..cpu().numpy()  # (1,3)

    map_1 = np.zeros((w, h))
    map_2 = np.zeros((w, h))
    map_3 = np.zeros((w, h))

    map_1_w = np.zeros((w, h))
    map_2_w = np.zeros((w, h))
    map_3_w = np.zeros((w, h))

    # # why do we need the following code?
    # x_mean = np.mean(x_np, axis=(1, 2, 3))  # (10, 1)
    # x_std = np.std(x_np, axis=(1, 2, 3))

    if occ_status=='healthy':
        occ_seed = "/home/jjia/data/ssc_scoring/ssc_scoring/dataset/special_samples/healthy/healthy.mha"
    elif 'diseased' in occ_status:
        if occ_status=='diseased_gg':
            occ_seed = "/home/jjia/data/ssc_scoring/ssc_scoring/dataset/special_samples/diseased/diseased_gg.mha"
        elif occ_status=='diseased_ret':
            occ_seed = "/home/jjia/data/ssc_scoring/ssc_scoring/dataset/special_samples/diseased/diseased_ret.mha"
        else:
            occ_seed = "/home/jjia/data/ssc_scoring/ssc_scoring/dataset/special_samples/diseased/diseased.mha"
    occ_patch = generate_candidate(occ_seed)  # the healthy image is filled by healthy patches

    # print(f'x_np.mean: {x_mean}, x_np.std: {x_std}, ')
    # print(f"sum of lung mask: {np.sum(lung_mask.numpy())}")
    savefig(True, lung_mask, 'lung_mask.png', occlusion_dir)
    savefig(True, x_np[0], f"ori_image_tot_{int(out_ori_1)}_gg_{int(out_ori_2)}_ret_{int(out_ori_3)}.png", occlusion_dir)

    # nb_ptchs = math.ceil(512 / patch_size)
    ptch = patch_size
    i, j = 0, 0  # row index, column index
    while i < 512:
        while j < 512:
            # print(f'i, {i}, j, {j}')

            mask_ori = np.zeros((w, h))
            mask_ori[i : i + ptch, j: j + ptch] = 1
            # print(f"before lung mask, the patch mask sum is {np.sum(mask)}")
            mask_ori = mask_ori * lung_mask # exclude area outside lung
            # print(f"after lung mask, the patch mask sum is {np.sum(mask)}")

            # mask[nb_, 0, i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = 1
            mask = cv2.blur(mask_ori, (5, 5))
            # new_x = copy.deepcopy(x_np)  # shape [0, 512, 512], avoid changing the original x-np
            tmp = x_np[0] * (1-mask) + occ_patch * mask

            # import matplotlib.pyplot as plt
            # plt.figure()
            # fig, ax = plt.subplots()
            # ax.imshow(tmp, cmap='gray')
            # plt.show()
            # ax.axis('off')
            # fig.savefig(str(nb_) + '_sliding.png')
            # plt.close()
            # new_x = torch.tensor(new_x)
            # new_x[nb_, 0, i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = torch.tensor(x_health[:ptch, :ptch])
            new_x = torch.tensor(tmp).float()
            # new_x = new_x.double()
            new_x = new_x.unsqueeze(0).unsqueeze(0)
            # print(f"nex_x.shape: {new_x.shape}")
            #
            # new_x = new_x.unsqueeze(0)  # add channel and batch dims
            # print(f"nex_x.shape: {new_x.shape}")
            new_x = new_x.to(device)
            out = net(new_x)
            # print(f'out: {out}')
            # t3 = time.time()

            # print('t3-t2', t3-t2)

            out_np = out.clone().detach().cpu().numpy()
            out_1 = out_np[0, 0]  # scalar
            out_2 = out_np[0, 1]
            out_3 = out_np[0, 2]
            # print(f'out1: {out_1}')

            dif_1 = out_1 - out_ori_1  # use np.rint to ignore the random noise
            dif_2 = out_2 - out_ori_2
            dif_3 = out_3 - out_ori_3

            if save_occ_x:
                if i%patch_size==0 and j%patch_size==0:  # do not save all steps
                    savefig(True, tmp, f"{i}_{j}_x_tot_{int(out_1)}_gg_{int(out_2)}_ret_{int(out_3)}.png", occlusion_dir)
                    save_x_countor = False
                    if save_x_countor:
                        tmp2 = copy.deepcopy(tmp)
                        edge = 5
                        tmp2[i : i + edge, j : j + ptch] = 1
                        tmp2[i + ptch - edge: i + ptch, j: j + ptch] = 1
                        tmp2[i : i + ptch, j : j + edge] = 1
                        tmp2[i : i + ptch, j+ ptch - edge: j+ ptch] = 1

                        savefig(True, tmp2, f"{i}_{j}_occlusion_x_tot_{int(out_1)}_gg_{int(out_2)}_ret_{int(out_3)}.png", occlusion_dir)

            # print(f"out_1: {np.rint(out_1)}, out_2: {np.rint(out_2)}, out_3: {np.rint(out_3)}, "
            #       f"dif_1: {dif_1}, dif_2: {dif_2}, dif_3: {dif_3}")
            # threshold = 3  # mae difference greater than 3 is regarded valuable
            # if abs(dif_1) < threshold:
            #     dif_1 = 0
            # if abs(dif_2) < threshold:
            #     dif_2 = 0
            # if abs(dif_3) < threshold:
            #     dif_3 = 0
            # mask[mask>0]=1
            map_1[mask_ori > 0] += dif_1
            map_2[mask_ori > 0] += dif_2
            map_3[mask_ori > 0] += dif_3
            # print(f'sum of dif_1: {np.sum(dif_1)}')
            # print(f'sum of map1: {np.sum(map_1)}')
            map_1_w[mask_ori > 0] += 1
            map_2_w[mask_ori > 0] += 1
            map_3_w[mask_ori > 0] += 1
            j += stride
        i += stride
        j = 0
        # print(f'i: {i}')
            # map_1[i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = dif_1
            # map_2[i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = dif_2
            # map_3[i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = dif_3
            # t4 = time.time()

            # print('t4-t3', t4 - t3)
    # print(f'sum of map1: {sum(map_1)}')

    map_1_w[map_1_w==0] = 1
    map_2_w[map_2_w==0] = 1
    map_3_w[map_3_w==0] = 1

    map_1 = map_1 / map_1_w
    map_2 = map_2 / map_2_w
    map_3 = map_3 / map_3_w


    x_min = np.min(x_np)
    x_max = np.max(x_np)
    # print(f"x_min: {x_min}, x_max: {x_max}")

    x_np = x_np[0]
    x_np = (x_np - x_min) / (x_max - x_min) * 255
    cv2.imwrite(occlusion_dir + "/ori_img.jpg", x_np)
    # print(f"ori image saved at {occlusion_dir}")
    y_ls = list(y.numpy())
    pred_ls = list(out_np.reshape(-1,))
    # print(y_ls,  '----')

    # for nb, mp_1, mp_2, mp_3 in zip(range(nb_img), map_1, map_2, map_3):  # per CT
    save_higher = True
    save_lower = True
    save_diff = True
    for map, lb, score, pred in zip([map_1, map_2, map_3], ['disext', 'gg', 'rept'], y_ls, pred_ls):  # per label

        # map_mae = copy.deepcopy(map)
        map_ = copy.deepcopy(map)
        np.save(os.path.join(occlusion_dir, f"{lb}_ori_label_{score}_pred_{pred}_mae_diff.npy"), map)
        if save_diff:
            map_mae = normalize_255(-map, max_dif=4, min_dif=-4)
            # print(f'map_mae: {map_mae}')

            # map_mae[map_mae < 0] = 0
            # map_mae[map_mae == 0] = 127.5
            # map_mae[map_mae > 0] = 255
            # ['#4D08ED', '270477', '#000000','EDD607', '#ED0707']
            # []
            cmp = get_continuous_cmap(hex_list = ['#220FEC', '40C2BF', '#000000','E1D015', '#ED0707'])
            # plt.get_cmap('twilight')
            hm = apply_custom_colormap(np.uint8(map_mae), cmap=cmp)
            # print(f'before, map_hm: {hm}')

            # hm[map==0] = np.array([0,0,0])
            # hm = cv2.applyColorMap(np.uint8(map_mae), cv2.COLORMAP_JET)
            # print(f'after, map_hm: {hm}')

            map_mask = copy.deepcopy(map).reshape(w, h, 1)
            map_mask[map!=0] = 1
            img_with_map_mae_dif = 0.3 * hm + 0.7 * x_np.reshape(w, h, 1)
            temp1 = img_with_map_mae_dif * map_mask
            temp2 = x_np.reshape(w, h, 1) * (1-map_mask)
            saved_map = temp1 + temp2
            cv2.imwrite(os.path.join(occlusion_dir, f"{lb}_ori_label_{score}_pred_{pred}_mae_diff.jpg"), saved_map)

        if save_higher:
            map = normalize_255(map, max_dif=5, min_dif=-5)  # values range from 0 to 255
            map_mae_higher = copy.deepcopy(map)
            map_mae_higher[map_mae_higher <= 127.5] = 0  # make these pixels black
            # map_mae_higher[map_mae_higher > 127.5] = 255
            # map_mae_higher = normalize_255(map_mae_higher)
            map_mae_higher = cv2.applyColorMap(np.uint8(map_mae_higher), cv2.COLORMAP_JET)
            img_with_map_mae_higher = 0.3 * map_mae_higher + 0.7 * x_np.reshape(w, h, 1)

            temp1 = img_with_map_mae_higher * map_mask
            temp2 = x_np.reshape(w, h, 1) * (1 - map_mask)
            img_with_map_mae_higher = temp1 + temp2

            cv2.imwrite(os.path.join(occlusion_dir, lb+"_ori_label_"+str(score)+"_pred_" + str(pred) + "_mae_higher.jpg"),
                        img_with_map_mae_higher)

        if save_lower:
            map_mae_lower = copy.deepcopy(map)
            # map_mae_lower[map_mae_lower >= 0 ] = 0
            map_mae_lower[map_mae_lower >= 127.5] = 255
            map_mae_lower = 255 - map_mae_lower
            # map_mae_lower = normalize_255(map_mae_lower)
            map_mae_lower = cv2.applyColorMap(np.uint8(map_mae_lower), cv2.COLORMAP_JET)
            img_with_map_mae_lower = 0.3 * map_mae_lower + 0.7 * x_np.reshape(w, h, 1)

            temp1 = img_with_map_mae_lower * map_mask
            temp2 = x_np.reshape(w, h, 1) * (1 - map_mask)
            img_with_map_mae_lower = temp1 + temp2

            cv2.imwrite(os.path.join(occlusion_dir, lb+"_ori_label_"+str(score)+ "_pred_" + str(pred) + "_mae_lower.jpg"),
                        img_with_map_mae_lower)
        # print('finish {} of 3')


def normalize_255(occlusion_map: np.ndarray, max_dif = None, min_dif = None):
    """occlusion_map is the difference between two times of predictions.
    Rescale the map to [0, 255] for show

    Args:
        occlusion_map: numpy array

    Returns:

    """
    if max_dif is None:
        min_dif = np.min(occlusion_map)
    if min_dif is None:
        max_dif = np.max(occlusion_map)

    abs_min_dif = abs(min_dif)
    abs_max_dif = abs(max_dif)
    gap = max(abs_min_dif, abs_max_dif)
    min_dif = -gap
    max_dif = gap

    occlusion_map = (occlusion_map - min_dif) / (max_dif - min_dif) * 255
    occlusion_map[occlusion_map>255] = 255
    occlusion_map[occlusion_map<0] = 0

    return occlusion_map


def get_pat_dir(img_fpath: str) -> str:
    dir_ls = img_fpath.split('/')
    for path in dir_ls:
        if 'Pat_' in path:  # "Pat_023"
            return path

def get_level_dir(img_fpath: str) -> str:
    file_name = os.path.basename(img_fpath)
    return file_name[:6]


def batch_occlusion(net_id: int, patch_size: int, max_img_nb: int, occ_status='healthy'):

    args = get_args()  # get argument
    args.batch_size=15  # 15/3=5, all 5 levels in the same patient will be loaded in one batch
    mypath = Path(net_id)  # get path
    print(f"current dir: {os.path.abspath('.')}")  # make sure the current path is 'ssc_scoring'
    label_file = mypath.label_excel_fpath  # "dataset/GohScores.xlsx"  # labels are from here
    seed = 49  # for split of  cross-validation
    all_loader = LoadScore(mypath, label_file, seed, args, nb_img=None, require_lung_mask=True)  # data loader
    train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load()

    test_dataloader = iter(test_dataloader)  # only show visualization maps for testing dataset

    net = get_net('convnext_tiny', 3, args)  # get network architecture
    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # load trained weights
    net.eval()  # 8
    # print(net)

    nb_img= 0
    for data in tqdm(test_dataloader):
        nb_img += 1
        # print(f"nb_img, {nb_img}")

        if nb_img> max_img_nb:
            break
        xs, ys, lung_masks, img_fpaths = data['image_key'], data['label_key'], data['lung_mask_key'], data['fpath_key']
        # [batch, channel, w, h, d]
        idx = 0
        for x_, y_, lung_mask, img_fpath in zip(xs, ys, lung_masks, img_fpaths):  # xs, ys shape: [channel, w, h, d]
            idx += 1
            # print(f'idx, {idx}')
            if idx % 3 == 0:  # skip next 2 images because the neighboring 3 images are similar (up, middl, down)
                # x, y, lung_mask = x_[None], y_[None], lung_mask[None]  # x, y shape: [batch, channel, w, h, d]
                # grad_cam(x, y, net, nb_img)
                pat_dir = get_pat_dir(img_fpath)
                level_dir = get_level_dir(img_fpath)
                # if 'Pat_135' not in pat_dir:
                #     continue
                occlusion_map_dir = os.path.join(mypath.id_dir, 'test_data_occlusion_maps_occ_by_' + occ_status, pat_dir, level_dir)
                occlusion_map(patch_size, x_, y_, net, lung_mask, occlusion_map_dir, save_occ_x=True, stride=patch_size//4, occ_status=occ_status)


if __name__ == '__main__':
    occ_status = 'healthy' # 'diseased' or 'healthy' or 'diseased_ret'
    id = 1903
    patch_size = 64
    # grid_nb = 10
    batch_occlusion(id, patch_size, max_img_nb=1000, occ_status=occ_status)
    print('finish!')
