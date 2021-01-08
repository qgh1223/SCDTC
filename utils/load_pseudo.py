from .load import CellImageLoad
import cv2
import numpy as np
from .augumentation import *
import torch
from scipy.ndimage.interpolation import rotate
import random


class CellImageLoadPseudo(CellImageLoad):
    def __init__(self, ori_path, gt_path, bg_path, dataset, channel, crop_size=(128, 128)):
        super().__init__(ori_path, gt_path, dataset, channel, crop_size)
        self.bg_paths = bg_path

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = self.load_img(img_name)

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        gt = gt / 255

        bg_name = self.bg_paths[data_id]
        if bg_name == 0:
            bg = np.ones_like(img)
        else:
            bg = cv2.imread(str(bg_name), 0)
            bg = bg / 255

        if self.dataset in ["PhC-C2DL-PSC"]:
            img = img[40: img.shape[0] - 40, 60: img.shape[1] - 60]
            gt = gt[40: gt.shape[0] - 40, 60: gt.shape[1] - 60]
            bg = bg[40: bg.shape[0] - 40, 60: bg.shape[1] - 60]

        if self.dataset == "riken":
            img = img[:850]
            gt = gt[:850]
            bg = bg[:850]

        img, gt, bg = self.data_augument(img, gt, bg)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        bg = torch.from_numpy(bg.astype(np.float32))

        if self.channel == 1:
            img = img.unsqueeze(0)
        else:
            img = img.permute(2, 0, 1)

        datas = {"image": img, "gt": gt.unsqueeze(0), "bg": bg.unsqueeze(0)}

        return datas

    def data_augument(self, img, gt, bg):
        if self.dataset not in ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa"]:
            # data augumentation
            top, bottom, left, right = self.random_crop_param(img.shape[:2])

            img = img[top:bottom, left:right]
            gt = gt[top:bottom, left:right]
            bg = bg[top:bottom, left:right]

            rand_value = np.random.randint(0, 4)
            img = rotate(img, 90 * rand_value, mode="nearest")
            gt = rotate(gt, 90 * rand_value)

        # Brightness
        # pix_add = random.uniform(-0.1, 0.1)
        # img = change_brightness(img, pix_add)
        # img = change_brightness(img, pix_add)
        #
        # img = (img - img.min()) / (1 + pix_add - img.min())

        return img, gt, bg
