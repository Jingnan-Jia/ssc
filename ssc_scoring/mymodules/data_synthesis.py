# -*- coding: utf-8 -*-
# @Time    : 7/11/21 3:53 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import copy
import random
from multiprocessing import Manager, Lock
from typing import (Union)

import cv2
import myutil.myutil as futil
import numpy as np
import torch
from torchvision.transforms import CenterCrop, RandomAffine

import matplotlib.pyplot as plt
from statistics import mean
from monai.transforms import MapTransform

manager = Manager()
train_label_numbers = manager.dict(
    {label: key for label, key in zip(np.arange(0, 21) * 5, np.zeros((21,)).astype(np.int))})
train_lock = Lock()
validaug_label_numbers = manager.dict(
    {label: key for label, key in zip(np.arange(0, 21) * 5, np.zeros((21,)).astype(np.int))})
validaug_lock = Lock()
ori_nb = manager.Value('ori_nb', 0)
sys_nb = manager.Value('sys_nb', 0)


def savefig(save_flag, img, fpath):
    if save_flag:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        fig.savefig(fpath, bbox_inches='tight')
        plt.close()


def resort_pts_for_convex(pts_ls: list) -> list:
    pts_ls = sorted(pts_ls, key=lambda x: x[0])  # sort list by the x position
    new_pts_ls: list = [pts_ls[0]]  # decide the first point
    del pts_ls[0]  # remove the first point from original list

    # start from left to bottom
    idx_ls_to_del = []
    for idx, pos in enumerate(pts_ls):
        if pos[1] <= new_pts_ls[-1][1]:  # new points need to be lower than
            new_pts_ls.append(pos)
            idx_ls_to_del.append(idx)

    for offset, idx in enumerate(idx_ls_to_del):
        del pts_ls[idx - offset]  # pts_ls has already remove an element for each iteration

    # then start from right to bottom
    try:
        pts_ls = pts_ls[::-1]
        new_pts_ls_r2l = [pts_ls[0]]
        del pts_ls[0]
    except IndexError:
        new_pts_ls_r2l = []

    idx_ls_to_del = []
    for idx, pos in enumerate(pts_ls):
        if pos[1] <= new_pts_ls_r2l[-1][1]:  # new points need to be lower than
            new_pts_ls_r2l.append(pos)
            idx_ls_to_del.append(idx)
    for offset, idx in enumerate(idx_ls_to_del):
        del pts_ls[idx - offset]  # pts_ls has already remove an element for each iteration

    new_pts_ls_l2r = new_pts_ls_r2l[::-1]
    new_pts_ls.extend(new_pts_ls_l2r)  # left to bottom, bottom to right
    new_pts_ls.extend(pts_ls)  # right to left via top
    return new_pts_ls


def gen_pts(nb_points: int, limit: int, radius: int) -> np.ndarray:
    pts_ls: list = []
    for i in range(nb_points):
        if i == 0:
            pos = [random.randint(0, limit), random.randint(0, limit)]
            pts_ls.append(pos)
        else:
            pos = [pts_ls[0][0] + random.randint(-radius, radius),
                   pts_ls[0][1] + random.randint(-radius, radius)]

            pts_ls.append(pos)
    valuable_pts_ls = []
    for po in pts_ls:
        if po[0] < 0:
            po[0] = 0
        if po[0] > limit:
            po[0] = limit
        if po[1] < 0:
            po[1] = 0
        if po[1] > limit:
            po[1] = limit
        valuable_pts_ls.append(po)

    pts_ls = resort_pts_for_convex(valuable_pts_ls)
    pts: np.ndarray = np.array(pts_ls)
    pts: np.ndarray = pts.reshape((-1, 1, 2)).astype(np.int32)

    return pts


class SysthesisNewSampled(MapTransform):
    def __init__(self,
                 keys,
                 retp_fpath,
                 gg_fpath,
                 mode,
                 sys_pro_in_0,
                 retp_blur,
                 gg_blur,
                 sampler,
                 gen_gg_as_retp,
                 gg_increase

                 ):
        super().__init__(keys)
        # self.sys_ratio = sys_ratio
        self.image_size = 512
        self.random_affine = RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(1 - 0.5, 1 + 0.1))
        self.center_crop = CenterCrop(self.image_size)

        self.sys_pro = sys_pro_in_0 if sys_pro_in_0 else 20 / 21

        self.mode = mode
        self.retp_fpath = retp_fpath  # retp will generated from its egg
        self.gg_fpath = gg_fpath
        self.retp_blur = retp_blur
        self.gg_blur = gg_blur
        self.sampler = sampler
        self.gen_gg_as_retp = gen_gg_as_retp
        self.gg_increase = gg_increase

        self.retp_temp = self.generate_candidate(self.retp_fpath)
        self.gg_temp = self.generate_candidate(self.gg_fpath)

        self.retp_candidate = self.rand_affine_crop(self.retp_temp)
        self.gg_candidate = self.rand_affine_crop(self.gg_temp)

        self.counter = 0
        self.systh_y = []
        if self.mode == "train":
            self.label_numbers = train_label_numbers
        elif self.mode == 'validaug':
            self.label_numbers = validaug_label_numbers
        else:
            raise Exception("mode is wrong for synthetic data", self.mode)

    def generate_candidate(self, fpath):
        ori_image_fpath = fpath.split('.mha')[0] + '_ori.mha'
        egg = futil.load_itk(fpath)
        ori = futil.load_itk(ori_image_fpath)
        # normalize the egg using the original image information
        egg = (egg - np.min(ori)) / (np.max(ori) - np.min(ori))

        minnorv = np.vstack((np.flip(egg), np.flip(egg, 0)))
        minnorh = np.hstack((minnorv, np.flip(minnorv, 1)))

        cell_size = minnorh.shape
        nb_row, nb_col = self.image_size // cell_size[0] * 2, self.image_size // cell_size[1] * 2  # big mask for crop
        temp = np.hstack(([minnorh] * nb_col))
        temp = np.vstack(([temp] * nb_row))
        return temp

    def rand_affine_crop(self, retp_temp: np.ndarray):
        retp_temp_tensor = torch.from_numpy(retp_temp[None])
        retp_affina = self.random_affine(retp_temp_tensor)
        retp_candidate = self.center_crop(retp_affina)
        retp_candidate = torch.squeeze(retp_candidate).numpy()
        return retp_candidate

    def balanced(self, label):
        category = label // 5 * 5
        average = mean(list(self.label_numbers.values()))
        min_account = min(list(self.label_numbers.values()))
        max_account = max(list(self.label_numbers.values()))

        print("min account", min_account, "max account", max_account)
        print("generated label", category, 'all:', list(self.label_numbers.values()))

        if self.label_numbers[category] > min_account * 1.5:
            return False
        else:
            return True

    def account_label(self, label):
        category = label // 5 * 5
        # with self.lock:
        if self.mode == "train":
            with train_lock:
                train_label_numbers[category] = train_label_numbers[category] + 1
                print(sum(train_label_numbers.values()))
        else:
            with validaug_lock:
                validaug_label_numbers[category] += 1
                print(sum(validaug_label_numbers.values()))

    def __call__(self, data):
        d = dict(data)
        print("ori label is: " + str(d['label_key']))

        if d['label_key'][0].item() == 0:

            tmp = random.random()
            # print("tmp random is : " + str(tmp) + " self.sys_pro: " + str(self.sys_pro))
            if tmp < self.sys_pro:
                with train_lock:
                    sys_nb.value += 1
                    print("sys_nb: " + str(sys_nb.value))
                for key in self.keys:
                    d[key], d['label_key'] = self.systhesis(d[key], d['lung_mask_key'])
                    # with train_lock:
                    print("after systhesis, label is " + str(d['label_key']) + str("\n"))
            else:
                with train_lock:
                    ori_nb.value += 1
                    print("ori_nb: " + str(ori_nb.value))
                # with train_lock:
                print("No need for systhesis, label is " + str(d['label_key']) + str("\n"))
        else:
            with train_lock:
                ori_nb.value += 1
                print("ori_nb: " + str(ori_nb.value))

            # with train_lock:
            print("No need for systhesis, label is " + str(d['label_key']) + str("\n"))

        self.account_label(d['label_key'][0].item())
        return d

    def random_mask(self, nb_ellipse: int = 3, type: str = "ellipse"):
        fig_: np.ndarray = np.zeros((self.image_size, self.image_size))
        # Blue color in BGR
        color = 1
        # Line thickness of -1 px
        thickness = -1
        nb_shapes: int = random.randint(1, nb_ellipse)

        if type == "ellipse":
            startAngle = 0
            endAngle = 360
            # Using cv2.ellipse() method
            # Draw a ellipse with blue line borders of thickness of -1 px
            for i in range(nb_shapes):
                angle = random.randint(0, 180)
                center_coordinates = (random.randint(0, self.image_size), random.randint(0, self.image_size))
                if random.random() > 0.5:
                    axlen = random.randint(1, 100)
                else:
                    axlen = random.randint(1, 200)
                axesLength = (axlen, int(axlen * (1 + random.random())))

                image = cv2.ellipse(fig_, center_coordinates, axesLength, angle,
                                    startAngle, endAngle, color, thickness)
                fig_ += image
            fig_[fig_ > 0] = 1
        else:
            radius = 200
            for i in range(nb_shapes):
                nb_points: int = random.randint(3, 10)
                pts: np.ndarray = gen_pts(nb_points, limit=self.image_size, radius=radius)
                # image: np.ndarray = cv2.polylines(fig_, [pts], True, color, thickness=1)

                image: np.ndarray = cv2.fillPoly(fig_, [pts], color)
                fig_ += image
            fig_[fig_ > 0] = 1

            fig, ax = plt.subplots()
            ax.imshow(fig_, cmap='gray')
            plt.show()
            ax.axis('off')
            fig.savefig('/data/jjia/ssc_scoring/image_samples/polygonmask' + str(self.counter) + '.png')
            plt.close()

        return fig_

    def systhesis(self, img: torch.Tensor, lung_mask: Union[np.ndarray, torch.Tensor]):
        img = img.numpy()
        if type(lung_mask) == torch.Tensor:
            lung_mask = lung_mask.numpy()

        self.counter += 1
        if self.counter == 100:  # update self.retp_candidate
            self.counter = 0
            self.retp_candidate = self.rand_affine_crop(self.retp_temp)
            self.gg_candidate = self.rand_affine_crop(self.gg_temp)

        save_img: bool = False
        savefig(save_img, img, 'image_samples/0_ori_img_' + str(self.counter) + '.png')
        savefig(save_img, self.retp_candidate, 'image_samples/1_retp_candidate_' + str(self.counter) + '.png')

        while (1):
            rand_retp_mask = self.random_mask(3, type="ellipse")
            rand_gg_mask = self.random_mask(3, type="ellipse")

            savefig(save_img, rand_gg_mask, 'image_samples/2_rand_gg_mask_' + str(self.counter) + '.png')
            savefig(save_img, rand_retp_mask, 'image_samples/3_rand_retp_mask_' + str(self.counter) + '.png')
            savefig(save_img, lung_mask, 'image_samples/4_lung_mask_' + str(self.counter) + '.png')

            rand_retp_mask *= lung_mask
            rand_gg_mask *= lung_mask

            savefig(save_img, rand_gg_mask, 'image_samples/5_gg_mask_lung_' + str(self.counter) + '.png')
            savefig(save_img, rand_retp_mask, 'image_samples/6_retp_mask_lung_' + str(self.counter) + '.png')

            union_mask = rand_gg_mask + rand_retp_mask
            union_mask[union_mask > 0] = 1

            savefig(save_img, union_mask, 'image_samples/7_union_mask_lung_' + str(self.counter) + '.png')

            intersection_mask = rand_gg_mask * rand_retp_mask
            gg_exclude_retp = rand_gg_mask - intersection_mask

            savefig(save_img, intersection_mask, 'image_samples/8_intersection_mask_lung_' + str(self.counter) + '.png')
            savefig(save_img, gg_exclude_retp, 'image_samples/9_gg_exclude_retp_' + str(self.counter) + '.png')

            # lung_area = np.sum(lung_mask)
            # total_dis_area = np.sum(union_mask)
            # gg_area = np.sum(rand_gg_mask)
            # retp_area = np.sum(rand_retp_mask)
            #
            # y_disext = int(total_dis_area/lung_area * 100)
            # y_gg = int(gg_area / lung_area * 100)
            # y_retp = int(retp_area / lung_area * 100)
            # y = np.array([y_disext, y_gg, y_retp])

            smooth_edge = self.retp_blur
            rand_retp_mask = cv2.blur(rand_retp_mask, (smooth_edge, smooth_edge))
            savefig(save_img, rand_retp_mask, 'image_samples/10_retp_mask_blur_' + str(self.counter) + '.png')

            smooth_edge = self.gg_blur
            rand_gg_mask = cv2.blur(rand_gg_mask, (smooth_edge, smooth_edge))
            savefig(save_img, rand_gg_mask, 'image_samples/11_gg_mask_blur_' + str(self.counter) + '.png')

            smooth_edge = self.gg_blur
            intersection_mask = cv2.blur(intersection_mask, (smooth_edge, smooth_edge))
            savefig(save_img, intersection_mask,
                    'image_samples/11_intersection_mask_blur_' + str(self.counter) + '.png')

            rand_retp_mask_cp = copy.deepcopy(rand_retp_mask)  # recalculate scores
            rand_gg_mask_cp = copy.deepcopy(rand_gg_mask)
            rand_retp_mask_cp[rand_retp_mask_cp > 0] = 1
            rand_gg_mask_cp[rand_gg_mask_cp > 0] = 1
            rand_retp_mask_cp = rand_retp_mask_cp * lung_mask
            rand_gg_mask_cp = rand_gg_mask_cp * lung_mask

            union_mask = rand_gg_mask_cp + rand_retp_mask_cp
            union_mask[union_mask > 0] = 1

            lung_area = np.sum(lung_mask)
            total_dis_area = np.sum(union_mask)
            gg_area = np.sum(rand_gg_mask_cp)
            retp_area = np.sum(rand_retp_mask_cp)

            y_disext = int(total_dis_area / lung_area * 100)
            y_gg = int(gg_area / lung_area * 100)
            y_retp = int(retp_area / lung_area * 100)
            # print("old y: ", y)
            y = np.array([y_disext, y_gg, y_retp])
            print("new y: ", y)

            if self.sampler:
                if self.balanced(y_disext):
                    break
                else:
                    print("not balanced, re generate image")
            else:
                break

        if np.sum(y) > 0:
            retp = rand_retp_mask * self.retp_candidate
            savefig(save_img, retp, 'image_samples/12_retp_' + str(self.counter) + '.png')

            img_exclude_retp_mask = (1 - rand_retp_mask) * img
            savefig(save_img, img_exclude_retp_mask,
                    'image_samples/13_img_exclude_retp_mask_' + str(self.counter) + '.png')

            img_wt_retp = retp + img_exclude_retp_mask
            savefig(save_img, img_wt_retp, 'image_samples/14_img_wt_retp_' + str(self.counter) + '.png')

            if self.gen_gg_as_retp:
                gg = rand_gg_mask * self.gg_candidate
                savefig(save_img, gg, 'image_samples/12_gg_' + str(self.counter) + '.png')

                img_exclude_gg_mask = (1 - rand_gg_mask) * img_wt_retp
                savefig(save_img, img_exclude_gg_mask,
                        'image_samples/13_img_exclude_gg_mask_' + str(self.counter) + '.png')

                img_wt_retp_gg = gg + img_exclude_gg_mask
                savefig(save_img, img_wt_retp_gg,
                        'image_samples/14_img_wt_retp_gg_wo_overlap' + str(self.counter) + '.png')

                merge = 0.5 * (intersection_mask * img_wt_retp_gg) + 0.5 * (
                        intersection_mask * img_wt_retp)  # gg + retp
                final = img_wt_retp_gg * (1 - intersection_mask) + merge
                y_name = '_'.join([str(y[0]), str(y[1]), str(y[2])])
                savefig(save_img, final, 'image_samples/15_img_wt_retp_gg_final_' + str(self.counter) + y_name + '.png')

                # retp part



            else:
                smooth_edge = self.gg_blur
                rand_gg_mask = cv2.blur(rand_gg_mask, (smooth_edge, smooth_edge))
                savefig(save_img, rand_gg_mask, 'image_samples/15_gg_mask_blur_' + str(self.counter) + '.png')

                gg = copy.deepcopy(img_wt_retp)
                savefig(save_img, gg, 'image_samples/16_gg_candidate_' + str(self.counter) + '.png')

                lighter_gg = copy.deepcopy(gg)
                lighter_gg += self.gg_increase
                gg = rand_gg_mask * lighter_gg + (1 - rand_gg_mask) * gg
                savefig(save_img, gg, 'image_samples/17_gg_lighter_' + str(self.counter) + '.png')

                gg_blur = 3
                gg = cv2.blur(gg, (gg_blur, gg_blur))
                savefig(save_img, gg, 'image_samples/18_gg_lighter_blur_' + str(self.counter) + '.png')

                gg = rand_gg_mask * gg
                savefig(save_img, gg, 'image_samples/19_gg_lighter_blur_smoothed_' + str(self.counter) + '.png')

                img_exclude_gg_mask = (1 - rand_gg_mask) * img_wt_retp
                img_wt_retp_gg = img_exclude_gg_mask + gg
                savefig(save_img, img_wt_retp_gg, 'image_samples/20_img_wt_retp_gg_' + str(self.counter) + '.png')

            return torch.from_numpy(img_wt_retp_gg.astype(np.float32)), torch.tensor(y.astype(np.float32))
        else:
            return torch.from_numpy(img.astype(np.float32)), torch.tensor(np.array([0, 0, 0]).astype(np.float32))
