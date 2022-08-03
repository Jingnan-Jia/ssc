# generate the ssc weight map. the first 3 steps focus the borde, the 4th step will focus the lower lung
# 0. generate a rectangular just covering the whole lung.
# 1. generate a inscribed circle of the rectangular
# 2. make a gaussian distribution (edge is 1, center is 0)
# 3. deform the circle to an ellipse (resample the circle)
# 4. generate a gradient

import numpy as np
import glob
from ssc_scoring.mymodules.path import PathScore
from medutils.medutils import load_itk
import SimpleITK as sitk
import cv2
import os
import matplotlib.pyplot as plt
import math


def bbox2(img):
    """Return the box boundary for valuable voxels."""
    min = np.min(img)
    img = np.max(img, axis=0) # conver 3d lung to 2d lung (project to a axial view)


    rows = np.max(img, axis=1)
    cols = np.max(img, axis=0)
    rmin, rmax = np.where(rows>min)[0][[0, -1]]
    cmin, cmax = np.where(cols>min)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, img


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

# def blur_circle(img_rows, img_cols):
#     mask = np.ones((img_rows, img_cols))
#     for i in range(img_rows):
#         for j in range(img_cols):
#             d = math.sqrt((i - int(img_rows / 2)) * (i - int(img_rows / 2)) + (j - int(img_cols / 2)) * (j - int(img_cols / 2)))
#         #     r = int(np.round(d))
#         # if r <= 5:
#         #     mask[i, j] = 0
#         # elif r >= int(self.img_rows / 2):  # 40:
#         #     mask[i, j] = 0
#     return mask


def main():
    squared = False
    weight_dir = 'results/weight_maps_test'

    folder = PathScore().ori_data_dir
    file_ls = glob.glob(folder+"/Pat_*/CTimage_lung.mha")
    for file in file_ls:
        mask = load_itk(file)  # shape: [1000, 512, 512]
        rmin, rmax, cmin, cmax, mask_axial = bbox2(mask)
        width = rmax - rmin
        length = cmax - cmin
        edge = max(width, length)
        square = np.zeros([edge, edge])  # a square
        square[edge//2, edge//2] = 1

        square = sitk.GetImageFromArray(square.astype(np.int16))
        distance_map = sitk.SignedMaurerDistanceMap(square)
        distance_map = np.abs(sitk.GetArrayViewFromImage(distance_map))
        print(f"max: {np.max(distance_map)}")
        distance_map = distance_map/np.max(distance_map)
        if squared:  #
            distance_map = distance_map **2
        dim = (length, width)
        resized = cv2.resize(distance_map, dim, interpolation = cv2.INTER_LINEAR)

        # fig, ax = plt.subplots()
        # ax.imshow(resized, cmap='gray')
        # plt.show()

        # add the focus to the lower lung
        grad1d = np.linspace(0, 1, width)
        grad2d = np.tile(grad1d, (length, 1)).T
        if squared:
            grad2d = grad2d **2

        # fig, ax = plt.subplots()
        # ax.imshow(grad2d, cmap='gray')
        # plt.show()

        new_w = resized * 0.5 + grad2d * 0.5

        tmp_w = np.zeros(mask_axial.shape)
        tmp_w[rmin:rmax, cmin:cmax] = new_w

        new_w = tmp_w * mask_axial
        weight_dir = os.path.dirname(file)
        file_id = file.split('Pat_')[-1][:3]
        np.save(f'{weight_dir}/weight_map.npy', new_w)
        savefig(True, new_w, f'weight_map.png', weight_dir)






if __name__ == '__main__':
    main()