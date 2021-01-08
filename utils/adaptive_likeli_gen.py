import numpy as np
from numpy import linalg as LA
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage import filters
from skimage.transform import rescale, rotate, resize


def gaussian(x, sigma, mu):
    # 分散共分散行列の行列式
    det = np.linalg.det(sigma)
    # 分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = x.ndim
    return np.exp(-np.diag((x - mu) @ inv @ (x - mu).T) / 2.0) / (np.sqrt((2 * np.pi) ** n * det))


def tangent_angle(u: np.ndarray, v: np.ndarray):
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n
    return np.rad2deg(np.arccos(c))


# SCALED_DATASET = ["DIC-C2DH-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC"]

CTC_LIST = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC"]
dataset = ["B23P17", "GBM", "hMSC", "riken", "Elmer", "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "PhC-C2DH-U373",
           "PhC-C2DL-PSC", ]
dataset = ["hMSC"]

if __name__ == '__main__':
    for data in dataset:
        if data in CTC_LIST:
            seq_list = ["/01", "/02"]
        else:
            seq_list = [""]
        for seq in seq_list:
            mask_paths = sorted(
                Path(f"/home/kazuya/main/WSISPDR/outputs/graphcut_train/{data}{seq}/0.01/labelresults").glob("*.tif"))
            img_paths = sorted(
                Path(f"/home/kazuya/main/WSISPDR/outputs/guided_train/{data}{seq}").glob("*/original.png"))
            save_path = Path(f"/home/kazuya/dataset/adaptive_traindata/{data}{seq}")
            save_path.joinpath("gau").mkdir(parents=True, exist_ok=True)
            save_path.joinpath("img").mkdir(parents=True, exist_ok=True)

            for idx, img_path in enumerate(img_paths):
                img = cv2.imread(str(img_path), 0)
                cv2.imwrite(str(save_path.joinpath(f"img/{idx:05d}.png")), img)

            basis = np.zeros((101, 101))
            basis[50, 50] = 1
            basis = filters.gaussian(basis, sigma=9)
            basis = basis / basis.max()

            for mask_path in mask_paths:
                mask = np.array(Image.open(mask_path))
                gaus = np.zeros_like(mask, dtype=np.float)
                for mask_idx in np.unique(mask)[1:]:
                    if (mask_idx == mask).sum() < 10:
                        continue
                    gau = np.zeros((mask.shape[0] + 100, mask.shape[1] + 100))
                    y, x = np.where(mask_idx == mask)
                    y_pos = int(y.mean())
                    x_pos = int(x.mean())

                    sigma = np.cov(x, y)

                    X, Y = np.meshgrid(np.arange(101), np.arange(101))
                    try:
                        gau_local = gaussian(np.c_[X.flatten(), Y.flatten()], sigma, mu=np.array([50, 50]))
                    except:
                        continue
                    gau_local = gau_local.reshape((101, 101))
                    gau_local = gau_local / gau_local.max()

                    # w, v = np.linalg.eig(sigma)
                    # scale = w.max() / w.min()
                    # basis_s = rescale(basis, (1, scale))
                    #
                    # theta = tangent_angle(v[int(w.argmax())], np.array([1, 0]))
                    # if v[int(w.argmax())][1] > 0:
                    #     basis_s = rotate(basis_s, theta)
                    # else:
                    #     basis_s = rotate(basis_s, -theta)
                    # basis_s = basis_s[:, round((basis_s.shape[1] - basis_s.shape[0]) / 2):round((basis_s.shape[1] + basis_s.shape[0]) / 2)]
                    # gau[y_pos:y_pos+101, x_pos:x_pos+101] = resize(basis_s, (101, 101))
                    gau[y_pos:y_pos + 101, x_pos:x_pos + 101] = gau_local
                    gaus = np.maximum(gaus, gau[50: -50, 50: -50])

                # plt.imshow(gaus), plt.show()
                cv2.imwrite(str(save_path.joinpath("gau/" + mask_path.name)), (gaus * 255).astype(np.uint8))
