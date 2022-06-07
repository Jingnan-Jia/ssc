# -*- coding: utf-8 -*-
# @Time    : 7/11/21 3:53 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import copy
import glob
import random
from multiprocessing import Manager, Lock
from typing import (Union)

import cv2
from medutils.medutils import load_itk
import numpy as np
import torch
from torchvision.transforms import CenterCrop, RandomAffine
import os
import matplotlib.pyplot as plt
from statistics import mean
from monai.transforms import Transform, ScaleIntensityRange
from monai.transforms import RandomizableTransform

manager = Manager()
# Store the numbers of each label during multi-process training/validaug as a monitor of _balanced label distribution.
train_label_numbers = manager.dict(
    {label: key for label, key in zip(np.arange(0, 21) * 5, np.zeros((21,)).astype(np.int))})
train_lock = Lock()

validaug_label_numbers = manager.dict(
    {label: key for label, key in zip(np.arange(0, 21) * 5, np.zeros((21,)).astype(np.int))})
validaug_lock = Lock()

# Create two values to store the numbers of original images and synthetic images during multi-process training.
ori_nb = manager.Value('ori_nb', 0)
sys_nb = manager.Value('sys_nb', 0)


def savefig(save_flag: bool, img: np.ndarray, image_name: str, dir: str = "image_samples") -> None:
    """Save figure.

    Args:
        save_flag: Save or not.
        img: Image numpy array.
        image_name: image name.
        dir: directory

    Returns:
        None. Image will be saved to disk.

    Examples:
        :func:`ssc_scoring.mymodules.data_synthesis.SysthesisNewSampled`

    """

    fpath = os.path.join(dir, image_name)
    # print(f'image save path: {fpath}')

    directory = os.path.dirname(fpath)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    if save_flag:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        fig.savefig(fpath, bbox_inches='tight')
        plt.close()


def resort_pts_for_convex(pts_ls: list) -> list:
    """Re-sort the polygon points to get a convex polygon.

    Connect all of the points: start from the most left one, then left-bottom, gradually go to the right-bottom,
    then right-top, finally left-top.

    Args:
        pts_ls: A list of points, [(x1, y1), (x2, y2), ...]

    Returns:
        A new list of points. Connecting the points in the new list can get a convex.

    Examples:
        :func:`ssc_scoring.mymodules.data_synthesis.gen_pts`

    """
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
    """Get a list of points (a list of (x, y)) for the generation of convex polygon.

    Details:
        #. Randomly generate the first point in `[0, limit]`.
        #. Randomly generate other points whose relative distance to the first point is less than `radious`.
        #. Correct points outside the limit.
        #. re-sort the points so that connection of these points from the first one to the last one lead to a polygon.

    Args:
        nb_points: Number of points.
        limit: The limit/upperbound of x and y
        radius: Radius of the circle which can include all of these points.

    Returns:
        A list of points.

    Example:
        :func:`ssc_scoring.mymodules.data_synthesis.SysthesisNewSampled`

    """
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


class SysthesisNewSampled(RandomizableTransform, Transform):
    def __init__(self,
                 key,
                 retp_fpath,
                 gg_fpath,
                 mode,
                 sys_pro_in_0,
                 retp_blur,
                 gg_blur,
                 sampler,
                 gen_gg_as_retp,
                 gg_increase,
                 tr_x
                 ):
        """Synthesis new image samples.

        Args:
            key: Always be 'image_key'
            retp_fpath: Full path for one retp seed
            gg_fpath: Full path for one gg seed
            mode: Chosed from 'train' or 'validaug'
            sys_pro_in_0: Probability to synthesis image when is is healthy
            retp_blur: Smooth width between retp pattern foreground and healthy background
            gg_blur: Smooth width between gg pattern foreground and healthy background
            sampler: If using _balanced sampler which leads to _balanced label distribution
            gen_gg_as_retp: If generage gg pattern using the same method as it used by retp
            gg_increase: Voxel value increase for gg pattern, because gg part is always brighter

        """
        # self.sys_ratio = sys_ratio
        super(RandomizableTransform, self).__init__()
        self.key = key
        self.image_size = 512
        self.random_affine = RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(1 - 0.5, 1 + 0.1))
        self.center_crop = CenterCrop(self.image_size)

        self.sys_pro_in_0 = sys_pro_in_0
        self.tr_x = tr_x

        self.mode = mode
        self.retp_fpath = retp_fpath  # retp will generated from its egg
        self.gg_fpath = gg_fpath
        self.retp_blur = retp_blur
        self.gg_blur = gg_blur
        self.sampler = sampler
        self.gen_gg_as_retp = gen_gg_as_retp
        self.gg_increase = gg_increase

        self.ret_eggs_fpath = self._filter_egg_fpaths_for_train('ret')
        self.gg_eggs_fpath = self._filter_egg_fpaths_for_train('gg')

        self.retp_temp = self._generate_candidate(self.ret_eggs_fpath)
        self.gg_temp = self._generate_candidate(self.gg_eggs_fpath)

        self.retp_candidate = self._rand_affine_crop(self.retp_temp)
        self.gg_candidate = self._rand_affine_crop(self.gg_temp)

        self.counter = 0  # Count the number of training (or validaug) images
        self.synth_y = []  # Labels of synthetic images
        if self.mode == "train":
            self.label_numbers = train_label_numbers
        elif self.mode == 'validaug':
            self.label_numbers = validaug_label_numbers
        else:
            raise Exception("mode is wrong for synthetic data", self.mode)

    def _filter_egg_fpaths_for_train(self, pattern='gg'):
        if pattern=='gg':
            egg_fpaths = glob.glob(os.path.join(os.path.dirname(self.gg_fpath), 'gg_*from*.mha'))
        elif pattern=='ret':
            egg_fpaths = glob.glob(os.path.join(os.path.dirname(self.retp_fpath), 'ret_*from*.mha'))
        else:
            raise Exception(f'pattern should be ret or gg, but is {pattern}')
        tr_pat_ids = set([x_path.split('Pat_')[-1][:3] for x_path in self.tr_x])

        # train_egg_fpaths = []
        # for egg_fpath in egg_fpaths:
        #     pat_id = egg_fpath.split('pat')[-1][:3]
        #     print(f"tr_pat_ids: {tr_pat_ids}")
        #     if pat_id in tr_pat_ids:
        #         print(f"this pat_id {pat_id} from path {egg_fpath} is from training dataset, use it.")
        #         train_egg_fpaths.append(egg_fpath)
        #     else:
        #         print(f"this pat_id {pat_id} from path {egg_fpath} is not from training dataset, give up it.")
                
        return egg_fpaths


    def _generate_candidate(self, eggs_fpath):
        # ori_image_fpath = fpath.split('.mha')[0] + '_ori.mha'
        egg_fpath = random.choice(eggs_fpath)
        print(f'randomly select this egg: {egg_fpath}')
        egg = load_itk(egg_fpath)
        # ori = load_itk(ori_image_fpath)
        # normalize the egg using the original image information
        normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
        egg = normalize0to1(egg)
        # egg = (egg - np.min(ori)) / (np.max(ori) - np.min(ori))

        minnorv = np.vstack((np.flip(egg), np.flip(egg, 0)))
        minnorh = np.hstack((minnorv, np.flip(minnorv, 1)))

        cell_size = minnorh.shape
        nb_row, nb_col = self.image_size // cell_size[0] * 2, self.image_size // cell_size[1] * 2  # big mask for crop
        temp = np.hstack(([minnorh] * nb_col))
        temp = np.vstack(([temp] * nb_row))
        return temp

    def _rand_affine_crop(self, retp_temp: np.ndarray):
        retp_temp_tensor = torch.from_numpy(retp_temp[None])
        retp_affina = self.random_affine(retp_temp_tensor)
        retp_candidate = self.center_crop(retp_affina)
        retp_candidate = torch.squeeze(retp_candidate).numpy()
        return retp_candidate

    def _balanced(self, label):
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

    def _account_label(self, label):
        category = label // 5 * 5
        # with self.lock:
        if self.mode == "train":
            with train_lock:
                train_label_numbers[category] = train_label_numbers[category] + 1
                print(f'current train label numbers: {sum(train_label_numbers.values())}')
        else:
            with validaug_lock:
                validaug_label_numbers[category] += 1
                print(f'current validaug label numbers: {sum(validaug_label_numbers.values())}')

    def __call__(self, data):
        d = dict(data)
        print("ori label is: " + str(d['label_key']))

        if d['label_key'][0].item() == 0:  # Possible for synthesis
            tmp = random.random()
            if tmp < self.sys_pro_in_0:  # Do synthesis
                with train_lock:
                    sys_nb.value += 1
                    print("sys_nb: " + str(sys_nb.value))
                d[self.key], d['label_key'] = self._systhesis(d[self.key], d['lung_mask_key'])
                # with train_lock:
                print("after synthesis, label is " + str(d['label_key']) + str("\n"))
            else:  # No synthesis, number of original images +1
                with train_lock:
                    ori_nb.value += 1
                    print("ori_nb: " + str(ori_nb.value))
                # with train_lock:
                print("No need for synthesis, label is " + str(d['label_key']) + str("\n"))
        else:  # No synthesis, number of original images +1
            with train_lock:
                ori_nb.value += 1
                print("ori_nb: " + str(ori_nb.value))

            # with train_lock:
            print("No need for synthesis, label is " + str(d['label_key']) + str("\n"))

        self._account_label(d['label_key'][0].item())
        return d

    def _random_mask(self, nb_ellipse: int = 3, type: str = "ellipse"):
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
                # Array of polygons where each polygon is represented as an array of points.
                pts: np.ndarray = gen_pts(nb_points, limit=self.image_size, radius=radius)
                # image: np.ndarray = cv2.polylines(fig_, [pts], True, color, thickness=1)

                image: np.ndarray = cv2.fillPoly(fig_, [pts], color)
                fig_ += image
            fig_[fig_ > 0] = 1

            # savefig(True, fig_, str(self.counter) + 'polygonmask.png')

        return fig_

    def _systhesis(self, img: torch.Tensor, lung_mask: Union[np.ndarray, torch.Tensor]):
        img = img.numpy()
        if type(lung_mask) == torch.Tensor:
            lung_mask = lung_mask.numpy()

        if random.random() < 0.2:  # update affine every 5 images

            # if random.random() < 0.02:  # update pattern egg every 50 images
            self.retp_temp = self._generate_candidate(self.ret_eggs_fpath)
            self.gg_temp = self._generate_candidate(self.gg_eggs_fpath)

            self.retp_candidate = self._rand_affine_crop(self.retp_temp)
            self.gg_candidate = self._rand_affine_crop(self.gg_temp)

        save_img: bool = False  # If save the synthetic images and the intermediate images
        savefig(save_img, img, str(self.counter) + '_0_ori_img_' + self.mode + '.png')
        savefig(save_img, self.retp_candidate, str(self.counter) + '_1_retp_candidate.png')

        while (1):
            rand_retp_mask = self._random_mask(3, type="ellipse")
            rand_gg_mask = self._random_mask(3, type="ellipse")

            savefig(save_img, rand_gg_mask, str(self.counter) + '_2_rand_gg_mask.png')
            savefig(save_img, rand_retp_mask, str(self.counter) + '_3_rand_retp_mask.png')
            savefig(save_img, lung_mask, str(self.counter) + '_4_lung_mask.png')

            rand_retp_mask *= lung_mask
            rand_gg_mask *= lung_mask

            savefig(save_img, rand_gg_mask, str(self.counter) + '_5_gg_mask_lung.png')
            savefig(save_img, rand_retp_mask, str(self.counter) + '_6_retp_mask_lung.png')

            union_mask = rand_gg_mask + rand_retp_mask
            union_mask[union_mask > 0] = 1

            savefig(save_img, union_mask, str(self.counter) + '_7_union_mask_lung.png')

            intersection_mask = rand_gg_mask * rand_retp_mask
            gg_exclude_retp = rand_gg_mask - intersection_mask

            savefig(save_img, intersection_mask, str(self.counter) + '_8_intersection_mask_lung.png')
            savefig(save_img, gg_exclude_retp, str(self.counter) + '_9_gg_exclude_retp.png')

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
            savefig(save_img, rand_retp_mask, str(self.counter) + '_10_retp_mask_blur.png')

            smooth_edge = self.gg_blur
            rand_gg_mask = cv2.blur(rand_gg_mask, (smooth_edge, smooth_edge))
            savefig(save_img, rand_gg_mask, str(self.counter) + '_11_gg_mask_blur.png')

            smooth_edge = self.gg_blur
            intersection_mask = cv2.blur(intersection_mask, (smooth_edge, smooth_edge))
            savefig(save_img, intersection_mask, str(self.counter) + '_11_intersection_mask_blur.png')

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
                if self._balanced(y_disext):
                    break
                else:
                    print("not _balanced, re generate image")
            else:
                break

        if np.sum(y) > 0:
            retp = rand_retp_mask * self.retp_candidate
            savefig(save_img, retp, str(self.counter) + '_12_retp.png')

            img_exclude_retp_mask = (1 - rand_retp_mask) * img
            savefig(save_img, img_exclude_retp_mask,
                    str(self.counter) + '_13_img_exclude_retp_mask.png')

            img_wt_retp = retp + img_exclude_retp_mask
            savefig(save_img, img_wt_retp, str(self.counter) + '_14_img_wt_retp.png')

            if self.gen_gg_as_retp:
                gg = rand_gg_mask * self.gg_candidate
                savefig(save_img, gg, str(self.counter) + '_15_gg.png')

                img_exclude_gg_mask = (1 - rand_gg_mask) * img_wt_retp
                savefig(save_img, img_exclude_gg_mask,
                        str(self.counter) + '_16_img_exclude_gg_mask.png')

                img_wt_retp_gg = gg + img_exclude_gg_mask
                savefig(save_img, img_wt_retp_gg,
                        str(self.counter) + '_17_img_wt_retp_gg_wo_overlap.png')

                merge = 0.5 * (intersection_mask * img_wt_retp_gg) + 0.5 * (
                        intersection_mask * img_wt_retp)  # gg + retp
                final = img_wt_retp_gg * (1 - intersection_mask) + merge
                y_name = '_'.join([str(y[0]), str(y[1]), str(y[2])])
                savefig(save_img, final,
                        str(self.counter) + '_18_img_wt_retp_gg_final_' + y_name + '.png')

                # retp part



            else:
                smooth_edge = self.gg_blur
                rand_gg_mask = cv2.blur(rand_gg_mask, (smooth_edge, smooth_edge))
                savefig(save_img, rand_gg_mask, str(self.counter) + '_15_gg_mask_blur.png')

                gg = copy.deepcopy(img_wt_retp)
                savefig(save_img, gg, str(self.counter) + '_16_gg_candidate.png')

                lighter_gg = copy.deepcopy(gg)
                lighter_gg += self.gg_increase
                gg = rand_gg_mask * lighter_gg + (1 - rand_gg_mask) * gg
                savefig(save_img, gg, str(self.counter) + '_17_gg_lighter.png')

                gg_blur = 3
                gg = cv2.blur(gg, (gg_blur, gg_blur))
                savefig(save_img, gg, str(self.counter) + '_18_gg_lighter_blur.png')

                gg = rand_gg_mask * gg
                savefig(save_img, gg, str(self.counter) + '_19_gg_lighter_blur_smoothed.png')

                img_exclude_gg_mask = (1 - rand_gg_mask) * img_wt_retp
                img_wt_retp_gg = img_exclude_gg_mask + gg
                savefig(save_img, img_wt_retp_gg, str(self.counter) + '_20_img_wt_retp_gg.png')

            self.counter += 1
            return torch.from_numpy(img_wt_retp_gg.astype(np.float32)), torch.tensor(y.astype(np.float32))
        else:
            return torch.from_numpy(img.astype(np.float32)), torch.tensor(np.array([0, 0, 0]).astype(np.float32))
