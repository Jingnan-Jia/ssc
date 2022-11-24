import os
# sys.path.append("../..")

import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np

from ssc_scoring.mymodules.mydata import LoadScore
from ssc_scoring.mymodules.set_args import get_args
from ssc_scoring.run import Path, get_net



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda")


def get_pat_dir(img_fpath: str) -> str:
    dir_ls = img_fpath.split('/')
    for path in dir_ls:
        if 'Pat_' in path:  # "Pat_023"
            return path


def get_level_dir(img_fpath: str) -> str:
    file_name = os.path.basename(img_fpath)
    return file_name[:6]



if __name__ == '__main__':

    net_id = 1920

    args = get_args()  # get argument
    # 15/3=5, all 5 levels in the same patient will be loaded in one batch
    args.batch_size = 15
    mypath = Path(net_id)  # get path
    # make sure the current path is 'ssc_scoring'
    print(f"current dir: {os.path.abspath('.')}")
    # "dataset/GohScores.xlsx"  # labels are from here
    label_file = mypath.label_excel_fpath
    seed = 49  # for split of  cross-validation
    all_loader = LoadScore(mypath, label_file, seed, args,
                           nb_img=None, require_lung_mask=True)  # data loader
    valid_dataloader_ori = all_loader.load(onlyreturn='valid', nb=5000)[0]

    # only show visualization maps for testing dataset
    valid_dataloader = iter(valid_dataloader_ori)

    net = get_net('convnext_tiny', 3, args)  # get network architecture
    # load trained weights
    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
    target_layer = [net.features[-1][-1].block[0]]
    cam = GradCAMPlusPlus(model=net, target_layers=target_layer, use_cuda=True)

    ma_ls, mi_ls = [], []
    for data in tqdm(valid_dataloader):
        xs, ys, lung_masks, img_fpaths = data['image_key'], data[
            'label_key'], data['lung_mask_key'], data['fpath_key']
        # [batch, channel, w, h, d]
        idx = 0
        # xs, ys shape: [channel, w, h, d]
        for x_, y_, lung_mask, img_fpath in zip(xs, ys, lung_masks, img_fpaths):
            idx += 1
            # print(f'idx, {idx}')
            # skip next 2 images because the neighboring 3 images are similar (up, middl, down)
            if idx % 3 == 0:
                for pattern_idx, pattern in enumerate(['TOT', 'GG', 'RET']):

                    input_tensor = x_[None]
                    targets = [ClassifierOutputTarget(pattern_idx)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                    # In this example grayscale_cam has only one image in the batch:
                    grayscale_cam = grayscale_cam[0, :]
                    x_new = x_[0].cpu().detach().numpy()[:,:,None] * np.ones(3, dtype=int)[None, None, :]

                    ma, mi = np.max(grayscale_cam), np.min(grayscale_cam)
                    ma_ls.append(ma)
                    mi_ls.append(mi)
    truncated_max = sorted(ma_ls)[int(len(ma_ls)*0.95)]
    truncated_min = 0
    print('the truncated_max is: ', truncated_max)
    # plt.figure()
    # plt.scatter(ma_ls)
    valid_dataloader = iter(valid_dataloader_ori)

    for data in tqdm(valid_dataloader):
        xs, ys, lung_masks, img_fpaths = data['image_key'], data[
            'label_key'], data['lung_mask_key'], data['fpath_key']
        # [batch, channel, w, h, d]
        idx = 0
        # xs, ys shape: [channel, w, h, d]
        for x_, y_, lung_mask, img_fpath in zip(xs, ys, lung_masks, img_fpaths):
            idx += 1
            # print(f'idx, {idx}')
            # skip next 2 images because the neighboring 3 images are similar (up, middl, down)
            if idx % 3 == 0:
                for pattern_idx, pattern in enumerate(['TOT', 'GG', 'RET']):

                    input_tensor = x_[None]
                    targets = [ClassifierOutputTarget(pattern_idx)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                    # In this example grayscale_cam has only one image in the batch:
                    grayscale_cam = grayscale_cam[0, :]
                    x_new = x_[0].cpu().detach().numpy()[:,:,None] * np.ones(3, dtype=int)[None, None, :]

                    # re-scale according global maxmin
                    grayscale_cam[grayscale_cam>truncated_max]=truncated_max
                    grayscale_cam[grayscale_cam<truncated_min]=truncated_min
                    grayscale_cam = (grayscale_cam - truncated_min) / (truncated_max - truncated_min)

                    camoverimg = show_cam_on_image(x_new, grayscale_cam, use_rgb=True)

                    pat_dir = get_pat_dir(img_fpath)
                    level_dir = get_level_dir(img_fpath)
                    cam_map_dir = '_'.join([mypath.id_dir+'/valid_data_gradcam_plusplus_rescale2global/'+ pat_dir, level_dir])
                    
                 
                    plt.imshow(camoverimg)
                    plt.axis('off')
                    plt.savefig(f"{cam_map_dir}_{pattern}.png", bbox_inches='tight')
                print(f"{cam_map_dir}_{pattern}.png")

    print('finish!')
