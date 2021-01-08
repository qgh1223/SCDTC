from utils import gaus_filter
from pathlib import Path
import cv2
import numpy as np
from skimage.transform import resize
import re
import shutil


def boundary_gaussian(label_paths, bound_paths, save_path):
    sigma = 5
    kernel_size = 11
    save_path.joinpath("border").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("border_gaus").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("bound_gaus").mkdir(parents=True, exist_ok=True)

    for img_idx, (label_path, bound_path) in enumerate(zip(label_paths, bound_paths)):
        label = cv2.imread(str(label_path), -1)
        bound = cv2.imread(str(bound_path), 0)
        border = np.zeros((label.shape[0], label.shape[1], 3))
        contours, hierarchy = cv2.findContours(label.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        border = cv2.drawContours(border, contours, -1, (255, 255, 255), 3)
        border = cv2.cvtColor(border.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        border_gaus = gaus_filter(bound, kernel_size, sigma)
        cv2.imwrite(str(save_path.joinpath(f"bound_gaus/{img_idx:05d}.png")), border_gaus)

        border = bound - border
        cv2.imwrite(str(save_path.joinpath(f"border/{img_idx:05d}.png")), border)

        dst = gaus_filter(border, kernel_size, sigma)
        cv2.imwrite(str(save_path.joinpath(f"border_gaus/{img_idx:05d}.png")), dst)


def boundary_gen(mask_paths, save_path, ori_paths, lik_paths, thickness):
    save_path.joinpath("ori").mkdir(parents=True, exist_ok=True)
    save_path.joinpath(f"bou-{thickness}").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("mask").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("lik").mkdir(parents=True, exist_ok=True)
    for idx, (ori_path, mask_path, lik_path) in enumerate(zip(ori_paths, mask_paths, lik_paths)):
        ori = cv2.imread(str(ori_path), -1)
        mask = cv2.imread(str(mask_path), -1)
        mask = resize(mask, ori.shape, preserve_range=True, order=0).astype(np.uint8)

        lik = cv2.imread(str(lik_path), -1)
        lik = resize(lik, ori.shape, preserve_range=True, order=0).astype(np.uint8)

        boundary = np.zeros_like(mask)

        if mask.max() > 0:
            for i in range(1, mask.max() + 1):
                mask_tmp = np.zeros_like(mask, dtype=np.uint8)
                mask_tmp[mask == i] = 255
                contours, hierarchy = cv2.findContours(mask_tmp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                boundary = cv2.drawContours(boundary, contours, -1, (255, 255, 255), thickness)
            mask[mask > 0] = 255

        cv2.imwrite(str(save_path.joinpath(f"ori/{idx:05d}.tif")), (ori))
        cv2.imwrite(str(save_path.joinpath(f"bou-{thickness}/{mask_path.name}")), (boundary).astype(np.uint8))
        cv2.imwrite(str(save_path.joinpath(f"mask/{mask_path.name}")), (mask).astype(np.uint8))
        cv2.imwrite(str(save_path.joinpath(f"lik/{mask_path.name}")), (lik).astype(np.uint8))


ctc_datasets = {
    # 1: "BF-C2DL-HSC",
    # 2: "BF-C2DL-MuSC",
    3: "DIC-C2DH-HeLa",
    # 4: "Fluo-C2DL-MSC",
    # 5: "Fluo-N2DH-GOWT1",
    # 6: "Fluo-N2DH-SIM+",
    # 7: "Fluo-N2DL-HeLa",
    # 8: "PhC-C2DH-U373",
    # 9: "PhC-C2DL-PSC",
}

if __name__ == '__main__':

    # for dataset in ctc_datasets.values():
    #     for thickness in [1, 3]:
    #         for seq in [1, 2]:
    #             mask_paths = sorted(Path(f"../output/graphcut/{dataset}/{seq:02d}/0.01/labelresults").glob("*.tif"))
    #             ori_paths = sorted(
    #                 Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}").glob("*.*"))
    #             lik_paths = sorted(
    #                 Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}_GT/LIK-S").glob("*.*"))
    #             assert len(ori_paths) == len(mask_paths), print("different")
    #             save_path = Path(f"../image/Bes_train/{dataset}/{seq:02d}")
    #             save_path.mkdir(parents=True, exist_ok=True)
    #             # shutil.rmtree(save_path)
    #
    #             boundary_gen(mask_paths, save_path, ori_paths, lik_paths, thickness)
    #         for seq in ["01-01", "01-02", "02-01", "02-02"]:
    #             mask_paths = sorted(
    #                 Path(f"../output/graphcut_unlabeled/{dataset}/{seq}/0.01/labelresults").glob("*.tif"))
    #             ori_paths = sorted(
    #                 Path(f"/home/kazuya/dataset/Cell_tracking_challenge/test/{dataset}/{seq[-2:]}").glob("*.*"))
    #             lik_paths = sorted(
    #                 Path(f"/home/kazuya/main/WS/output/guided_unlabeled/{dataset}/{seq}").glob("*/*detection.png"))
    #             assert len(ori_paths) == len(mask_paths), print("different")
    #             save_path = Path(f"../image/Bes_train/{dataset}/{seq}")
    #             # shutil.rmtree(save_path)
    #             boundary_gen(mask_paths, save_path, ori_paths, lik_paths, thickness)

    for dataset in ctc_datasets.values():
        for thickness in [1, 3, 6]:
            for seq in [1, 2]:
                mask_paths = sorted(Path(f"../output/graphcut/{dataset}/{seq:02d}/0.01/labelresults").glob("*.tif"))
                ori_paths = sorted(Path(f"../output/guided_train/{dataset}/{seq:02d}").glob("*/*original.png"))
                lik_paths = sorted(
                    Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}_GT/LIK-S").glob("*.*"))
                assert len(ori_paths) == len(mask_paths), print("different")
                save_path = Path(f"../image/Bes_train/{dataset}/{seq:02d}_train_s/{seq:02d}")
                save_path.mkdir(parents=True, exist_ok=True)
                # shutil.rmtree(save_path)

                boundary_gen(mask_paths, save_path, ori_paths, lik_paths, thickness)

                label_paths = sorted(
                    Path(f"/home/kazuya/main/WS/output/graphcut/{dataset}/{seq:02d}/0.01/labelresults").glob("*.tif"))
                bound_paths = sorted(
                    Path(f"/home/kazuya/main/WS/image/Bes_train/{dataset}/{seq:02d}_train_s/{seq:02d}/bou-3").glob(
                        "*.*"))
                save_path = Path(f"/home/kazuya/main/WS/image/Bes_train/{dataset}/{seq:02d}_train_s/{seq:02d}")
                boundary_gaussian(label_paths, bound_paths, save_path)
            for seq in ["01-01", "01-02", "02-01", "02-02"]:
                mask_paths = sorted(
                    Path(f"../output/graphcut_unlabeled/{dataset}/{seq}/0.01/labelresults").glob("*.tif"))
                ori_paths = sorted(Path(f"../output/guided_unlabeled/{dataset}/{seq}").glob("*/*original.png"))
                lik_paths = sorted(
                    Path(f"../output/guided_unlabeled/{dataset}/{seq}").glob("*/*detection.png"))
                assert len(ori_paths) == len(mask_paths), print("different")
                save_path = Path(f"../image/Bes_train/{dataset}/{seq[:2]}_train_s/{seq}")
                # shutil.rmtree(save_path)

                boundary_gen(mask_paths, save_path, ori_paths, lik_paths, thickness)

                label_paths = sorted(
                    Path(f"/home/kazuya/main/WS/output/graphcut_unlabeled/{dataset}/{seq}/0.01/labelresults").glob(
                        "*.tif"))
                bound_paths = sorted(
                    Path(f"/home/kazuya/main/WS/image/Bes_train/{dataset}/{seq[:2]}_train_s/{seq}/bou-3").glob(
                        "*.*"))
                save_path = Path(f"/home/kazuya/main/WS/image/Bes_train/{dataset}/{seq[:2]}_train_s/{seq}")
                boundary_gaussian(label_paths, bound_paths, save_path)
