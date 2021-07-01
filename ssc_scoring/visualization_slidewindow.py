# -*- coding: utf-8 -*-
# @Time    : 3/28/21 2:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import copy
import time

import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from ssc_scoring.run import get_net, Path, prepare_data, SysDataset, ssc_transformd
from torch.utils.data import Dataset, DataLoader
import myutil.myutil as futil

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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


def saliency_map(x, y, net):  # for one image
    x = x.to(device)
    net.to(device)
    out_ori = net(x) # (10, 1, 3)
    nb_img, _, w, h = x.shape  # (10, 1, 512, 512)
    print(x.shape)  # torch.Size([101, 512, 512])
    print(out_ori.shape)

    # print('nb_img:', nb_img)
    out_ori = out_ori.detach().cpu().numpy()
    out_ori_1 = out_ori[:, 0]
    out_ori_2 = out_ori[:, 1]
    out_ori_3 = out_ori[:, 2]

    # out_ori_2 = out_ori[0][1].item()
    # out_ori_3 = out_ori[0][2].item()
    # print(out_ori_2)

    x_np = x.clone().detach().cpu().numpy()

    # y_np = y.clone().detach()..cpu().numpy()  # (1,3)

    map_1 = np.zeros_like(x_np).reshape(nb_img, w, h)
    map_2 = np.zeros_like(x_np).reshape(nb_img, w, h)
    map_3 = np.zeros_like(x_np).reshape(nb_img, w, h)

    x_mean = np.mean(x_np, axis=(1, 2, 3))  # (10, 1)
    x_std = np.std(x_np, axis=(1, 2, 3))
    x_min = np.min(x_np, axis=(1, 2, 3))
    x_max = np.max(x_np, axis=(1, 2, 3))
    def health_img():

        def generate_candidate(fpath):
            image_size = 512
            ori_image_fpath = fpath.split('.mha')[0] + '_ori.mha'
            egg = futil.load_itk(fpath)
            ori = futil.load_itk(ori_image_fpath)
            egg[egg>1500] = 1500
            egg[egg<-1500] = -1500
            ori[ori > 1500] = 1500
            ori[ori < -1500] = -1500
            # normalize the egg using the original image information
            egg = (egg - np.min(ori)) / (np.max(ori) - np.min(ori))

            minnorv = np.vstack((np.flip(egg), np.flip(egg, 0)))
            minnorh = np.hstack((minnorv, np.flip(minnorv, 1)))

            cell_size = minnorh.shape
            nb_row, nb_col = image_size // cell_size[0] * 2, image_size // cell_size[
                1] * 2  # big mask for crop
            temp = np.hstack(([minnorh] * nb_col))
            temp = np.vstack(([temp] * nb_row))
            temp = temp[:image_size, :image_size]
            return temp

        health_fpath = "/data/jjia/ssc_scoring/dataset/special_samples/healthy.mha"
        health_temp = generate_candidate(health_fpath)
        return health_temp

    x_health = health_img()
    print(f'x_np.mean: {x_mean}, x_np.std: {x_std}, ')

    nb_ptchs = 4
    ptch = int(w / nb_ptchs)
    for i in range(nb_ptchs):
        for j in range(nb_ptchs):
            print(f'i, {i}, j, {j}')
            new_x = x.clone().detach()

            for nb_ in range(nb_img):
                mask = np.zeros((512, 512))
                mask[i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = 1
                # mask[nb_, 0, i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = 1
                mask = cv2.blur(mask, (20, 20))
                tmp = new_x.numpy()[nb_, 0] * (1-mask) + x_health * mask
                new_x[nb_, 0] = torch.tensor(tmp)
                if nb_ < 1:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    fig, ax = plt.subplots()
                    ax.imshow(tmp, cmap='gray')
                    plt.show()
                    ax.axis('off')
                    fig.savefig(str(nb_) + '_sliding.png')
                    plt.close()
                # new_x = torch.tensor(new_x)
                # new_x[nb_, 0, i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = torch.tensor(x_health[:ptch, :ptch])
            new_x = new_x.to(device)
            out = net(new_x)
            # t3 = time.time()

            # print('t3-t2', t3-t2)

            out_np = out.clone().detach().cpu().numpy()
            out_1 = out_np[:, 0]  # scalar
            out_2 = out_np[:, 1]
            out_3 = out_np[:, 2]

            chg_1 = out_ori_1- out_1
            chg_2 = out_ori_2- out_2
            chg_3 = out_ori_3- out_3
            # print(f"out_1: {out_1}, out_2: {out_2}, out_3: {out_3}")
            # print(f"chg1, {chg_1}, chg2, {chg_2}, chg3, {chg_3}")
            for nb in range(nb_img):
                map_1[nb, i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = chg_1[nb]
                map_2[nb, i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = chg_2[nb]
                map_3[nb, i * ptch: (i + 1) * ptch, j * ptch: (j + 1) * ptch] = chg_3[nb]
            # t4 = time.time()

            # print('t4-t3', t4 - t3)

    x_np = x_np.reshape(nb_img, w, h)
    for nb in range(nb_img):
        x_np[nb] = (x_np[nb] - x_min[nb]) / (x_max[nb] - x_min[nb]) * 255
    for i, data in zip(range(nb_img), x_np):
        cv2.imwrite(mypath.id_dir + '/' + str(i) + "img.jpg", data)
    for nb, mp_1, mp_2, mp_3 in zip(range(nb_img), map_1, map_2, map_3):  # per CT
        for map, lb, lb_idx in zip([mp_1, mp_2, mp_3], ['disext', 'gg', 'rept'], [0,1,2]):  # per label

            map_po = copy.deepcopy(map)
            map_po[map_po < 0] = 0
            map_po = normalize_255(map_po)
            hm_po = cv2.applyColorMap(np.uint8(map_po), cv2.COLORMAP_JET)
            slsy_img_po = 0.3 * hm_po + 0.7 * x_np[nb].reshape(w, h, 1)
            cv2.imwrite(mypath.id_dir + '/' + str(nb) + lb + "slsy_"+str(y[nb,lb_idx].item())+"po.jpg", slsy_img_po)

            map_ng = copy.deepcopy(map)
            map_ng[map_ng > 0] = 0
            map_ng = normalize_255(map_ng)
            hm_ng = cv2.applyColorMap(np.uint8(map_ng), cv2.COLORMAP_JET)
            slsy_img_ng = 0.3 * hm_ng + 0.7 * x_np[nb].reshape(w, h, 1)
            cv2.imwrite(mypath.id_dir + '/' + str(nb) + lb + "slsy_" + str(y[nb, lb_idx].item()) + "ng.jpg",
                        slsy_img_ng)

            map = normalize_255(map)
            hm = cv2.applyColorMap(np.uint8(map), cv2.COLORMAP_JET)
            slsy_img = 0.3 * hm + 0.7 * x_np[nb].reshape(w, h, 1)
            cv2.imwrite(mypath.id_dir + '/' + str(nb) + lb + "slsy_"+str(y[nb,lb_idx].item())+".jpg", slsy_img)

            print('save image to ', mypath.id_dir + '/' + str(nb) + lb + "slsy_"+str(y[nb,lb_idx].item())+".jpg")



def normalize_255(map_1_po):
    map_1_po = (map_1_po - np.min(map_1_po)) / (np.max(map_1_po) - np.min(map_1_po)) * 255
    return map_1_po

if __name__ == '__main__':
    id = 1405
    mypath = Path(id)

    tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data(mypath)
    ts_dataset = SysDataset(ts_x[:10], ts_y[:10], transform=ssc_transformd())
    test_dataloader = DataLoader(ts_dataset, batch_size=10, shuffle=False, num_workers=6)
    test_dataloader = iter(test_dataloader)

    net = get_net('vgg11_bn', 3)
    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
    net.eval()  # 8
    print(net)



    nb_img = 0
    while nb_img < 10:
        print(f"nb_img, {nb_img}")
        data = next(test_dataloader)
        xs, ys = data['image_key'], data['label_key']
        for x_, y_, idx in zip(xs, ys, range(10)):
            print(f'idx, {idx}')
            if idx % 3 == 0:
                print(f'idx is okay, {idx}')
                x, y = x_[None], y_[None]
                # grad_cam(x, y, net, nb_img)
                saliency_map(x, y, net)

                nb_img += 1
