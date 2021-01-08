from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import cv2

DATASETS = {
        # 1: "C2C12",
        # 6: "hMSC",
        # 3: "Elmer",
        # 2: "GBM",
        # 5: "B23P17",
        # 3: "riken",
        7: "BF-C2DL-HSC",
        8: "BF-C2DL-MuSC",
        4: "DIC-C2DH-HeLa",
        5: "PhC-C2DH-U373",
        12: "PhC-C2DL-PSC",
        }

SCALED_DATASET = [
        "DIC-C2DH-HeLa",
        "Fluo-C2DL-MSC",
        "Fluo-N2DH-GOWT1",
        "Fluo-N2DH-SIM+",
        "Fluo-N2DL-HeLa",
        "PhC-C2DH-U373",
        "PhC-C2DL-PSC",
    ]

SIG_LIST = {"BF-C2DL-HSC": 6,
            "BF-C2DL-MuSC": 9,
            "DIC-C2DH-HeLa": 9,
            "PhC-C2DH-U373": 9,
            "PhC-C2DL-PSC": 6,
            "C2C12": 9,
            "B23P17": 12,
            "Elmer": 3,
            "GBM": 9,
            "hMSC": 6,
            "riken": 9}
CTC_LIST = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC"]

if __name__ == '__main__':
    for dataset in DATASETS.values():
        if dataset in CTC_LIST:
            seq_list = ["/01", "/02"]
        else:
            seq_list = [""]
        for seq in seq_list:
            mask_paths = sorted(Path(f"/home/kazuya/main/WSISPDR/outputs/graphcut_train2/{dataset}{seq}/0.01/labelresults").glob("*.tif"))
            save_path = Path(f"/home/kazuya/dataset/watershed_segm/{dataset}{seq}")
            save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
            save_path.joinpath("dist").mkdir(parents=True, exist_ok=True)
            save_path.joinpath("seed").mkdir(parents=True, exist_ok=True)

            if dataset in SCALED_DATASET:
                scale = "-S"
            else:
                scale = ""

            # data_dirs = sorted(Path(f"/home/kazuya/dataset/send_ws/{dataset}_cut").iterdir())
            #
            # data_dirs.pop(0)
            # if dataset == "B23P17":
            #     data_dirs.pop(3)
            # else:
            #     data_dirs.pop(0)
            #
            # img_paths = []
            # lik_paths = []
            # for data_dir in data_dirs:
            #     if dataset == "riken":
            #         img_paths.extend(sorted(data_dir.joinpath("ori").glob("*.tif"))[100:300])
            #         lik_paths.extend(sorted(data_dir.joinpath(f"{SIG_LIST[dataset]}").glob("*.tif"))[100:300])
            #     else:
            #         img_paths.extend(sorted(data_dir.joinpath("ori").glob("*.tif")))
            #         lik_paths.extend(sorted(data_dir.joinpath(f"{SIG_LIST[dataset]}").glob("*.tif")))

            data_dir = Path(f"/home/kazuya/dataset/send_ws/Cell_tracking_challenge/{dataset}")
            # data_dir = Path(f"/home/kazuya/dataset/send_ws/C2C12P7/sequ_cut/0303/sequ9")
            #
            # img_paths = sorted(data_dir.joinpath("ori").glob("*.*"))
            # lik_paths = sorted(data_dir.joinpath(f"{SIG_LIST[dataset]}").glob("*.tif"))
            #
            img_paths = sorted(data_dir.joinpath(seq[1:] + scale).glob("*.*"))
            lik_paths = sorted(data_dir.joinpath(f"{seq[1:]}_GT/LIK{scale}").glob("*.tif"))

            assert len(mask_paths) == len(img_paths), "the number of mask and img is not same"
            assert len(lik_paths) == len(img_paths), "the number of lik and img is not same"

            for idx, (mask_path, img_path, lik_path) in enumerate(zip(mask_paths, img_paths, lik_paths)):
                mask = np.array(Image.open(mask_path))
                img = np.array(Image.open(img_path))
                # img = img / img.max() * 255
                lik = np.array(Image.open(lik_path))
                # plt.imshow(mask), plt.show()

                bl_mask = np.zeros_like(mask)
                for mask_idx in np.unique(mask)[1:]:
                    tmp_mask = np.zeros_like(mask)
                    tmp_mask[mask_idx == mask] = 255
                    # plt.imshow(tmp_mask), plt.show()
                    bl_mask_tmp = gaussian(tmp_mask, sigma=3)
                    bl_mask_tmp = bl_mask_tmp / bl_mask_tmp.max() * 255
                    bl_mask = np.maximum(bl_mask, bl_mask_tmp)

                cv2.imwrite(str(save_path.joinpath(f"dist/{idx:05d}.png")), bl_mask.astype(np.uint8))
                cv2.imwrite(str(save_path.joinpath(f"img/{idx:05d}.png")), img.astype(np.uint8))
                cv2.imwrite(str(save_path.joinpath(f"seed/{idx:05d}.png")), lik)
                # plt.imshow(bl_mask), plt.show()




