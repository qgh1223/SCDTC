import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import torch
import cv2
from scipy.ndimage.interpolation import rotate
import math
from .augumentation import *


class CellImageLoad(object):
    def __init__(self, ori_path, gt_path, dataset, channel, crop_size=(128, 128)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.dataset = dataset
        self.channel = channel
        self.crop_size = crop_size

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = self.load_img(img_name)

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        gt = gt / 255

        if self.dataset in ["PhC-C2DL-PSC"]:
            gt = gt[40: gt.shape[0] - 40, 60: gt.shape[1] - 60]

        if self.dataset == "riken":
            img = img[:850]
            gt = gt[:850]

        img, gt = self.data_augument(img, gt)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        if self.channel == 1:
            img = img.unsqueeze(0)
        else:
            img = img.permute(2, 0, 1)

        datas = {"image": img, "gt": gt.unsqueeze(0)}

        return datas

    def load_img(self, img_name):
        if self.dataset in ["C2C12"]:
            img = cv2.imread(str(img_name), -1)
            img = img / 4096
        elif self.dataset in ["hMSC"]:
            img = cv2.imread(str(img_name), -1)
            img = img / img.max()
        elif self.dataset in ["MoNuSeg", "TNBC"]:
            img = cv2.imread(str(img_name))
            img = img / 255
        elif self.dataset in ["GBM", "B23P17", "Elmer", "riken"]:
            img = cv2.imread(str(img_name), 0)
            img = img / 255
        else:
            # img = np.load(str(img_name))
            img = cv2.imread(str(img_name), -1)
            if self.dataset in ["Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa"]:
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = img / 255
            if self.dataset in ["PhC-C2DL-PSC"]:
                img = img[40: img.shape[0] - 40, 60: img.shape[1] - 60]
        return img

    def data_augument(self, img, gt):
        if self.dataset not in ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa"]:
            # data augumentation
            top, bottom, left, right = self.random_crop_param(img.shape[:2])

            img = img[top:bottom, left:right]
            gt = gt[top:bottom, left:right]

        rand_value = np.random.randint(0, 4)
        img = rotate(img, 90 * rand_value, mode="nearest")
        gt = rotate(gt, 90 * rand_value)

        img_height, img_width = img.shape

        # Brightness
        pix_add = random.uniform(-0.1, 0.1)

        img = change_brightness(img, pix_add)

        img = (img - img.min()) / (1 + pix_add - img.min())

        return img, gt

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right


class CellImageLoadTest(CellImageLoad):
    def __init__(self, ori_path, dataset, channel, crop_size=(128, 128)):
        self.ori_paths = ori_path
        self.dataset = dataset
        self.channel = channel
        self.crop_size = crop_size

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = self.load_img(img_name)

        if self.dataset == "riken":
            img = img[:850]

        img = torch.from_numpy(img.astype(np.float32))

        if self.channel == 1:
            img = img.unsqueeze(0)
        else:
            img = img.permute(2, 0, 1)

        datas = {"image": img}

        return datas

