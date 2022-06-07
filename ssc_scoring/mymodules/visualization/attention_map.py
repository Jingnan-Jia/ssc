# -*- coding: utf-8 -*-
# @Time    : 4/15/21 10:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import numpy as np
from medutils.medutils import load_itk

import cv2
import torchvision.models as models
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm

from ssc_scoring.mymodules.path import PathScore as Path

class GradCAM():
    def __init__(self, eval_id):
        self.eval_id = eval_id
        self.mypath = Path(self.eval_id)


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        self.grad_block = []
        self.fmap_block = []

        self.net = models.vgg11_bn()
        self.net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # change in_features to 1
        self.net.classifier[0] = torch.nn.Linear(in_features=512 * 7 * 7, out_features=1024)
        self.net.classifier[3] = torch.nn.Linear(in_features=1024, out_features=1024)
        self.net.classifier[6] = torch.nn.Linear(in_features=1024, out_features=3)

        self.net.load_state_dict(torch.load(self.mypath.model_fpath, map_location=self.device))
        self.net.to(self.device)
        self.net.features[-1].register_forward_hook(self.farward_hook)
        self.net.features[-1].register_backward_hook(self.backward_hook)
        self.net.eval()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)

        # 定义获取梯度的函数
    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def run(self, pat_id, pat_level):
        pat_id_str = str(pat_id)
        if len(pat_id_str)==2:
            pat_id_str = '0' + pat_id_str

        self.pat_name = "Pat_" + pat_id_str
        self.Level =str(pat_level)
        self.img_fpath = self.mypath.data_dir + "/SSc_DeepLearning/" + self.pat_name + "/Level" + self.Level + "_middle.mha"

        image, origin, space = load_itk(self.img_fpath, require_ori_sp=True)
        image_len = image.shape[0]
        image[image < -1500] = -1500
        image[image > 1500] = 1500
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        w, h = image.shape
        img = image[None][None]
        img = torch.tensor(img)

        img_id = int(self.img_fpath.split('/')[-2].split('_')[-1])
        label_file = self.mypath.label_excel_fpath # "/data/jjia/ssc_scoring/dataset/GohScores.xlsx"
        df_excel = pd.read_excel(label_file, engine='openpyxl')
        df_excel = df_excel.set_index('PatID')

        y_disext = df_excel.at[img_id, 'L' + self.Level + '_disext']
        y_gg = df_excel.at[img_id, 'L' + self.Level + '_gg']
        y_retp = df_excel.at[img_id, 'L' + self.Level + '_retp']


        label_all = torch.tensor(np.array([[y_disext, y_gg, y_retp]]).astype(np.float32))
        label_total = torch.tensor(np.array([[y_disext, 0, 0]]).astype(np.float32))
        label_gg = torch.tensor(np.array([[0, y_gg, 0]]).astype(np.float32))
        label_retp = torch.tensor(np.array([[0, 0, y_retp]]).astype(np.float32))

        loss = nn.MSELoss()
        image = (image - image.min()) / (image.max() - image.min())
        image = np.expand_dims(image, -1)
        output = self.net(img)
        label_str = str(int(label_all[0, 0].item())) + '_' +str(int(label_all[0, 1].item())) + '_' +str(int(label_all[0, 2].item()))
        pred = output.detach().cpu().numpy()
        pred_str = str(int(pred[0, 0])) + "_" + str(int(pred[0, 1])) + "_" + str(int(pred[0, 2]))
        print(f"predict: {pred}")

        for name, label in zip(['all', 'total', 'gg', 'retp'], [label_all, label_total, label_gg,label_retp ]):
            self.grad_block = []
            print(f"predict: {output.detach().cpu().numpy()}, label: {label.detach().cpu().numpy()}")

            l = loss(output,label )
            self.opt.zero_grad()  # clear the gradients
            l.backward(retain_graph=True)
            grad = torch.stack(self.grad_block, 0)
            grads_mean = torch.mean(grad, [1, 2])

            cam = torch.zeros(list(self.fmap_block[0].shape))
            for weight, map in zip(grads_mean, self.fmap_block):
                cam += weight * map
            cam = cam[0]
            cam = torch.mean(cam, 0)
            cam = cam.cpu().detach().numpy()


            cam = np.maximum(cam, 0)
            cam = cam / cam.max()
            cam = cv2.resize(cam, (w, h))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

            cam_img = 0.3 * heatmap + 0.7 * image*255
            if not os.path.isdir(self.mypath.id_dir + '/cam'):
                os.makedirs(self.mypath.id_dir + '/cam')

            cv2.imwrite(self.mypath.id_dir + '/cam/'+self.pat_name+'level_' +self.Level +name+'.png', cam_img)
        cv2.imwrite(self.mypath.id_dir + '/cam/' + self.pat_name + 'level_' + self.Level + 'label_' +  label_str + "pred_"+pred_str + '.png', image*255)

def gassuan_blur2d(total_length, smooth_len):
    small_box = np.zeros((total_length - 2 * smooth_len, total_length - 2 * smooth_len))
    big_box = np.pad(small_box, smooth_len, mode='linear_ramp', end_values=1)
    return big_box


def main():
    cam = GradCAM(1405)
    ts_id = [68, 83, 36, 187, 238, 12, 158, 189, 230, 11, 35, 37, 137, 144, 17, 42, 66, 70, 28, 64, 210, 3, 49, 32,
             236, 206, 194, 196, 7, 9, 16, 19, 20, 21, 40, 46, 47, 57, 58, 59, 60, 62, 116, 117, 118, 128, 134, 216]

    for id in tqdm(ts_id[4:10 ]):
        for level in [1,2,3,4,5]:
            cam.run(id, level)
        print('finish id: ', id)
    print("finish all")


if __name__ == "__main__":
    main()
