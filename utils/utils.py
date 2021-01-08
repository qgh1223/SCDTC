import numpy as np
import cv2
from pathlib import Path
import random
import os
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def gather_path(train_paths, mode, extension):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob(extension)))
    return ori_paths


def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(
        img, (pad_size, pad_size), "constant"
    )  # zero padding
    img_t = cv2.GaussianBlur(
        img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma
    )  # gaussian filter
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t


def heatmap_gen(shape, cell_positions, g_size, save_path):
    black = np.zeros((shape[0], shape[1]))

    # 1013 - number of frame
    for frame in range(int(cell_positions[:, 0].max()) + 1):
        # likelihood map of one input
        result = black.copy()
        cells = cell_positions[cell_positions[:, 0] == frame]
        for _, x, y in cells:
            img_t = black.copy()  # likelihood map of one cell
            img_t[int(y)][int(x)] = 255  # plot a white dot
            img_t = gaus_filter(img_t, 301, g_size)
            result = np.maximum(result, img_t)  # compare result with gaussian_img
        #  normalization
        result = 255 * result / result.max()
        result = result.astype("uint8")
        cv2.imwrite(str(save_path / Path("%05d.tif" % frame)), result)
        print(frame + 1)
    print("finish")
