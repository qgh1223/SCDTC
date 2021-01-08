from pathlib import Path
import cv2
import numpy as np

import cv2
from pathlib import Path
import numpy as np


def monuseg_cut():
    root_path = Path("/home/kazuya/dataset/Cancer/MoNuSegTrainingData")
    ori_paths = sorted(root_path.joinpath("ori").glob("*.png"))
    likeli_paths = sorted(root_path.joinpath("6").glob("*.png"))
    mask_paths = sorted(root_path.joinpath("mask").glob("*.png"))

    save_path = Path("/home/kazuya/dataset/Cancer/MoNuSegTrainCut")
    save_path.joinpath("ori").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("likeli").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("mask").mkdir(parents=True, exist_ok=True)

    for path in zip(ori_paths, likeli_paths, mask_paths):
        img = cv2.imread(str(path[0]))
        likeli = cv2.imread(str(path[1]))
        mask = cv2.imread(str(path[2]), -1)
        for x in [0, 320, 640]:
            for y in [0, 320, 640]:
                img_cut = img[x: x + 320, y:y + 320]
                likeli_cut = likeli[x: x + 320, y: y + 320]
                mask_cut = mask[x: x + 320, y: y + 320]

                cv2.imwrite(str(save_path.joinpath(f"ori/{path[0].stem}_{x:03d}_{y:03d}.png")), img_cut)
                cv2.imwrite(str(save_path.joinpath(f"likeli/{path[1].stem}_{x:03d}_{y:03d}.png")), likeli_cut)
                cv2.imwrite(str(save_path.joinpath(f"mask/{path[1].stem}_{x:03d}_{y:03d}.png")), mask_cut)


def fluo_n2dl_hela():
    for seq in [1, 2]:
        root_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/detection/Fluo-N2DL-HeLa/sequ{seq:02d}")
        ori_paths = sorted(root_path.joinpath("s_scale").glob("*.npy"))
        likeli_paths = sorted(root_path.joinpath("s_6").glob("*.tif"))
        # mask_paths = sorted(root_path.joinpath("mask").glob("*.png"))

        save_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/detection/Fluo-N2DL-HeLaCut/sequ{seq:02d}")
        save_path.joinpath("ori").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("likeli").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("mask").mkdir(parents=True, exist_ok=True)

        for path in zip(ori_paths, likeli_paths):
            img = np.load(str(path[0]))
            likeli = cv2.imread(str(path[1]))
            # mask = cv2.imread(str(path[2]), -1)
            for x in [0, 320, 640]:
                for y in [0, 320, 640, 960, 1280]:
                    img_cut = img[x: x + 320, y:y + 320]
                    likeli_cut = likeli[x: x + 320, y: y + 320]
                    # mask_cut = mask[x: x + 320, y: y + 320]

                    np.save(str(save_path.joinpath(f"ori/{path[0].stem}_{x:03d}_{y:03d}.npy")), img_cut)
                    cv2.imwrite(str(save_path.joinpath(f"likeli/{path[1].stem}_{x:03d}_{y:03d}.png")), likeli_cut)
                    # cv2.imwrite(str(save_path.joinpath(f"mask/{path[1].stem}_{x:03d}_{y:03d}.png")), mask_cut)


def phc_c2dl_psc():
    for seq in [1, 2]:
        root_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/detection/PhC-C2DL-PSC/sequ{seq:02d}")
        ori_paths = sorted(root_path.joinpath("l_scale").glob("*.npy"))
        likeli_paths = sorted(root_path.joinpath("l_6").glob("*.tif"))
        # mask_paths = sorted(root_path.joinpath("mask").glob("*.png"))

        save_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/detection/PhC-C2DL-PSCCut/sequ{seq:02d}")
        save_path.joinpath("ori").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("likeli").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("mask").mkdir(parents=True, exist_ok=True)

        for path in zip(ori_paths, likeli_paths):
            # img = cv2.imread(str(path[0]))
            img = np.load(str(path[0]))
            likeli = cv2.imread(str(path[1]))
            # mask = cv2.imread(str(path[2]), -1)
            for x in [0, 320, 640, 960, 1600]:
                for y in [0, 320, 640, 960, 1600, 1920, 2240]:
                    img_cut = img[x: x + 320, y:y + 320]
                    likeli_cut = likeli[x: x + 320, y: y + 320]
                    # mask_cut = mask[x: x + 320, y: y + 320]

                    np.save(str(save_path.joinpath(f"ori/{path[0].stem}_{x:03d}_{y:03d}.npy")), img_cut)
                    cv2.imwrite(str(save_path.joinpath(f"likeli/{path[1].stem}_{x:03d}_{y:03d}.png")), likeli_cut)
                    # cv2.imwrite(str(save_path.joinpath(f"mask/{path[1].stem}_{x:03d}_{y:03d}.png")), mask_cut)


def c2c2():
    img_paths = sorted(Path("/home/kazuya/dataset/C2C12P7/mask/ori").glob("*.tif"))
    mask_paths = sorted(Path("/home/kazuya/dataset/C2C12P7/mask/mask").glob("*.tif"))
    save_path = Path("/home/kazuya/dataset/C2C12P7/mask")
    save_path.joinpath("mask_cut").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("ori_cut").mkdir(parents=True, exist_ok=True)

    for ori_path, mask_path in zip(img_paths, mask_paths):
        for height in range(2):
            for width in range(2):
                h = height * 512
                w = width * 512
                img = cv2.imread(str(ori_path), -1)[h:h + 512,
                      w: w + 512]
                mask = cv2.imread(str(mask_path), -1)[h:h + 512,
                       w: w + 512]

                cv2.imwrite(str(save_path.joinpath(f"ori_cut/{ori_path.stem}-{h:04d}-{w:04d}.png")), img)
                cv2.imwrite(str(save_path.joinpath(f"mask_cut/{ori_path.stem}-{h:04d}-{w:04d}.png")), mask)


def C2C12_cut():
    # input_path = sorted(Path("/home/kazuya/dataset/C2C12P7/mask/mask").glob("*.tif"))
    input_path = sorted(Path("/home/kazuya/hdd/competitive/result/Yin/C2C12/seg").glob("*.tif"))
    # output_path = Path("/home/kazuya/dataset/C2C12P7/mask/mask_cut")
    output_path = Path("/home/kazuya/hdd/competitive/result/Yin/C2C12/seg_cut")
    output_path.mkdir(parents=True, exist_ok=True)
    for img_i, path in enumerate(input_path):
        # load image
        ori_img = cv2.imread(str(path), -1)

        for height in range(2):
            for width in range(2):
                h = height * 512
                w = width * 512
                output_path_each = output_path.joinpath(f"{path.stem}-{h:04d}-{w:04d}.png")

                cv2.imwrite(str(output_path_each), ori_img[h: h + 512, w: w + 512])


import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(30 * 30 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(4, 30 * 30 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x1 = x.repeat(1, 64 * 64)
        x2 = x.repeat(1, 64 * 64)
        return x

if __name__ == '__main__':
    model = Net()
    x = torch.randn(4, 1, 64, 64)
    y = model(x)
    # fluo_n2dl_hela()
    # phc_c2dl_psc()
    # C2C12_cut()
    for seq in ["01", "02"]:
        img_paths = sorted(Path(f"/home/kazuya/dataset/Cell_tracking_challenge/Fluo-N2DL-HeLa/{seq}").glob("*.tif"))
        save_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/Fluo-N2DL-HeLa/{seq}-reg")
        save_path.mkdir(parents=True, exist_ok=True)
        for index, img_path in enumerate(img_paths):
            img = cv2.imread(str(img_path), -1)
            img = (img - img.min()) / (img.max() - img.min())
            cv2.imwrite(str(save_path.joinpath(f"{index:04d}.tif")), (img * 255).astype(np.uint8))

    # frame1 = 616
    # x1 = 244
    # y1 = 27
    #
    # frame2 = 646
    # x2 = 363
    # y2 = 132
    #
    # seq = 2
    # num = 2
    # for method in ['ours']:
    #     # for method in ['ours']:
    #     paths = sorted(Path(f"/home/kazuya/Downloads/MMPM/F{seq:04d}").glob("*.png"))
    #     save_path = Path(f"/home/kazuya/Downloads/MMPM/F{seq:04d}-cut/{num:02d}")
    #     save_path.mkdir(exist_ok=True, parents=True)
    #     for i in range(frame1, frame2):
    #         ori_path = paths[i]
    #         img = cv2.imread(str(ori_path))
    #         img = img[y1:y2, x1: x2].astype(np.uint8)
    #
    #         cv2.imwrite(str(save_path.joinpath(f"{i:05d}.png")), img)
