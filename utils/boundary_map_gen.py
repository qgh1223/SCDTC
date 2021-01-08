import shutil
from skimage.segmentation import find_boundaries
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re
import cv2

w0 = 10
sigma = 5

DATASET = {"BF-C2DL-HSC": {1: [], 2: []},
           "BF-C2DL-MuSC": {1: [], 2: []},
           "DIC-C2DH-HeLa": {1: [67], 2: []},
           "Fluo-C2DL-MSC": {1: [9, 28, 30, 31, 36, 46, 47], 2: []},
           "Fluo-N2DH-GOWT1": {
               1: [2, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 27, 28, 29, 32, 34, 37, 40, 41, 44, 45, 46, 47],
               2: [22, 25, 27, 28, 29, 30, 35, 39, 46, 47, 60, 65, 76, 80, 82, 91]},
           "Fluo-N2DH-SIM+": {1: [], 2: []},
           "Fluo-N2DL-HeLa": {
               1: [12, 14, 15, 20, 21, 22, 23, 25, 29, 38, 39, 40, 44, 50, 51, 53, 54, 55, 62, 76, 77, 78, 79, 80, 81,
                   88], 2: [23, 35, 36, 67, 78, 87]},
           "PhC-C2DH-U373": {1: [], 2: []},
           "PhC-C2DL-PSC": {1: [], 2: []}, }


def make_weight_map(masks):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.

    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)

    """
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ


def supervised():
    mask_paths = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}_GT/SEG").glob("*.tif")
    save_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}_GT/WMP")

    save_path.mkdir(parents=True, exist_ok=True)

    for mask_path in mask_paths:
        mask = Image.open(mask_path)
        frame = int(re.search(r'\d+', mask_path.stem).group())
        if frame not in DATASET[dataset][seq]:
            mask = np.array(mask)
            mask_3d = []
            for instance_id in np.unique(mask)[1:]:
                binary_mask = np.zeros_like(mask)
                binary_mask[mask == instance_id] = 1
                mask_3d.append(binary_mask)
            weight_map = make_weight_map(np.array(mask_3d))
            # plt.imshow(weight_map), plt.show()
            np.save(str(save_path.joinpath(mask_path.stem)), weight_map)


def boundary_map_gen(ori_paths, mask_paths, lik_paths, save_path):
    save_path.joinpath("WMP").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("mask").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("like").mkdir(parents=True, exist_ok=True)

    for mask_path, ori_path, lik_path in zip(mask_paths, ori_paths, lik_paths):
        mask = Image.open(mask_path)
        mask = np.array(mask)
        if mask.max() != 0:
            mask_3d = []
            mask_2d = np.zeros_like(mask)
            for instance_id in np.unique(mask)[1:]:
                binary_mask = np.zeros_like(mask)
                binary_mask[mask == instance_id] = 1
                mask_3d.append(binary_mask)
                mask_2d = np.maximum(mask_2d, cv2.erode(binary_mask, np.ones((3, 3)), iterations=1))
            weight_map = make_weight_map(np.array(mask_3d))

            cv2.imwrite(str(save_path.joinpath("mask/" + mask_path.name)), (mask_2d * 255).astype(np.uint8))
            img = cv2.imread(str(ori_path), 0)
            cv2.imwrite(str(save_path.joinpath("img/" + mask_path.name)), img)
            like = cv2.imread(str(lik_path), 0)
            cv2.imwrite(str(save_path.joinpath("like/" + mask_path.name)), like)

            np.save(str(save_path.joinpath("WMP/" + mask_path.stem)), weight_map)

MODES = {
    # 6: "BF-C2DL-HSC",
    # 7: "BF-C2DL-MuSC",
    8: "DIC-C2DH-HeLa",
    # 5: "Fluo-C2DL-MSC",
    # 6: "Fluo-N2DH-GOWT1",
    # 7: "Fluo-N2DH-SIM+",
    # 8: "Fluo-N2DL-HeLa",
    9: "PhC-C2DH-U373",
    # 10: "PhC-C2DL-PSC",
}


if __name__ == '__main__':
    for dataset in MODES.values():
        # for seq in [1, 2]:
        #     mask_paths = sorted(Path(f"../output/graphcut/{dataset}/{seq:02d}/0.01/labelresults").glob("*.tif"))
        #     ori_paths = sorted(Path(f"../output/guided_train/{dataset}/{seq:02d}").glob("*/original.png"))
        #     lik_paths = sorted(
        #         Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}_GT/LIK-S").glob("*.*"))
        #     assert len(ori_paths) == len(mask_paths), print(f"{dataset}: {seq}")
        #     save_path = Path(f"../image/UNet_train/{dataset}/{seq:02d}_train/{seq:02d}")
        #
        #     boundary_map_gen(ori_paths, mask_paths, lik_paths, save_path)

        for seq in ["01-01", "01-02", "02-01", "02-02"]:
            mask_paths = sorted(Path(f"../output/graphcut_unlabeled/{dataset}/{seq}/0.01/labelresults").glob("*.tif"))
            ori_paths = sorted(Path(f"../output/guided_unlabeled/{dataset}/{seq}").glob("*/original.png"))
            lik_paths = sorted(Path(f"../output/guided_unlabeled/{dataset}/{seq}").glob("*/*detection.png"))
            assert len(ori_paths) == len(mask_paths), print(f"{dataset}: {seq}")
            save_path = Path(f"../image/UNet_train/{dataset}/{seq[:2]}_train/{seq}")

            boundary_map_gen(ori_paths, mask_paths, lik_paths, save_path)
