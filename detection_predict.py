from datetime import datetime
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import cv2
from networks import UNet
from utils import show_res, optimum, target_peaks_gen, remove_outside_plot, local_maxim
import argparse
import math
import matplotlib.pyplot as plt


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="dataset's path",
        default="./image/test",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./output/detection",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        default="./weight/best.pth",
    )
    parser.add_argument(
        "-c", "--channel", dest="channel", help="img channel 1 or 3", default=1, type=int
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", default=True, action="store_true"
    )

    args = parser.parse_args()
    return args


def call_predict(args):
    net = UNet(n_channels=args.channel, n_classes=1)
    net.load_state_dict(torch.load(args.weight_path, map_location="cpu"))

    if args.gpu:
        net.cuda()
    args.net = net

    if args.fm:
        pred = PredictFmeasure(args)
    else:
        pred = Predict(args)

    pred.main()


def gather_path(train_paths, mode, extension):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob(extension)))
    return ori_paths


class Predict:
    def __init__(self, args):
        self.net = args.net
        self.gpu = args.gpu

        self.dataset = args.dataset
        self.ori_path = args.imgs

        self.save_ori_path = args.output_path / Path("ori")
        self.save_pred_path = args.output_path / Path("pred")

        self.save_ori_path.mkdir(parents=True, exist_ok=True)
        self.save_pred_path.mkdir(parents=True, exist_ok=True)

    def pred(self, img):
        if self.dataset in ["MoNuSeg", "TNBC"]:
            img = img.transpose(2, 0, 1)
        else:
            img = img.reshape(1, img.shape[0], img.shape[1]).astype(np.float32)
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            if self.gpu:
                img = img.cuda()
            mask_pred = self.net.forward(img)
        pre_img = mask_pred.detach().cpu().numpy()[0, 0]
        pre_img = (pre_img * 255).astype(np.uint8)
        return pre_img

    def load_img(self, ori_path, gt_path=None):
        if self.dataset in ["C2C12", "hMSC"]:
            ori = cv2.imread(str(ori_path), -1)
        elif self.dataset in ["MoNuSeg", "TNBC"]:
            ori = cv2.imread(str(ori_path))
        elif self.dataset in ["ushi"]:
            ori = cv2.imread(str(ori_path), 0)
            ori = (ori + (111 - ori.mean()))
        else:
            ori = cv2.imread(str(ori_path), 0)

        if self.dataset in ["MoNuSeg", "TNBC"]:
            img = (ori.astype(np.float32) / 255)
        elif self.dataset == "C2C12":
            img = (ori.astype(np.float32) / 4096)
        elif self.dataset == "hMSC":
            img = (ori.astype(np.float32) / ori.max())
        elif self.dataset in ["GBM", "B23P17", "Elmer", "riken"]:
            img = (ori.astype(np.float32) / 255)
        elif self.dataset in ["Fluo-N2DH-SIM+", "Fluo-N2DL-HeLa"]:
            img = (ori - ori.min()) / (ori.max() - ori.min())
            img = img.astype(np.float32)
        else:
            img = ori / 255

        if gt_path is not None:
            gt = np.array(Image.open(gt_path).convert('L'))
            return img, gt
        return img

    def main(self):
        self.net.eval()

        for i, path in enumerate(self.ori_path):
            ori = self.load_img(path)
            pre_img = self.pred(ori)
            cv2.imwrite(str(self.save_pred_path / Path("%05d.tif" % i)), pre_img)
            cv2.imwrite(str(self.save_ori_path / Path("%05d.tif" % i)), (ori * 255).astype(np.uint8))


class PredictFmeasure(Predict):
    def __init__(self, args):
        super().__init__(args)
        self.ori_path = args.imgs
        self.gt_path = args.likelis

        self.save_gt_path = args.output_path / Path("gt")
        self.save_error_path = args.output_path / Path("error")
        self.save_txt_path = args.output_path / Path("f-measure.txt")

        self.save_gt_path.mkdir(parents=True, exist_ok=True)
        self.save_error_path.mkdir(parents=True, exist_ok=True)

        self.peak_thresh = 100
        self.dist_peak = 2
        if self.dataset == "PhC-C2DH-U373":
            self.dist_threshold = 100
        else:
            self.dist_threshold = 20

        if self.dataset == "PhC-C2DL-PSC":
            self.w_th = 100
        else:
            self.w_th = 10
        self.tps = 0
        self.fps = 0
        self.fns = 0

    def cal_tp_fp_fn(self, ori, gt_img, pre_img, i):
        gt = target_peaks_gen((gt_img).astype(np.uint8))
        res = local_maxim(pre_img, self.peak_thresh, self.dist_peak)
        associate_id = optimum(gt, res, self.dist_threshold)

        gt_final, no_detected_id = remove_outside_plot(gt, associate_id, 0, pre_img.shape, window_thresh=self.w_th)
        res_final, overdetection_id = remove_outside_plot(res, associate_id, 1, pre_img.shape, window_thresh=self.w_th)

        show_res(ori, gt, res, no_detected_id, overdetection_id, path=str(self.save_error_path / Path("%05d.tif" % i)))
        cv2.imwrite(str(self.save_pred_path / Path("%05d.png" % (i))), pre_img)
        cv2.imwrite(str(self.save_ori_path / Path("%05d.png" % (i))), (ori * 255).astype(np.uint8))
        cv2.imwrite(str(self.save_gt_path / Path("%05d.png" % (i))), gt_img)

        tp = associate_id.shape[0]
        fn = gt_final.shape[0] - associate_id.shape[0]
        fp = res_final.shape[0] - associate_id.shape[0]
        self.tps += tp
        self.fns += fn
        self.fps += fp

    def main(self):
        self.net.eval()

        for i, (ori_path, gt_path) in enumerate(zip(self.ori_path, self.gt_path)):
            import gc
            gc.collect()

            ori, gt_img = self.load_img(ori_path, gt_path)
            pre_img = self.pred(ori)

            self.cal_tp_fp_fn(ori, gt_img, pre_img, i)

        if self.tps == 0:
            f_measure = 0
            precision = 0
            recall = 0
        else:
            recall = self.tps / (self.tps + self.fns)
            precision = self.tps / (self.tps + self.fps)
            f_measure = (2 * recall * precision) / (recall + precision)

        print(self.dataset, precision, recall, f_measure)
        with self.save_txt_path.open(mode="a") as f:
            f.write("%f,%f,%f\n" % (precision, recall, f_measure))


def supervised_test(args):
    for seq in [[1, 2], [2, 1]]:
        base_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{args.dataset}")
        if args.dataset in SCALED_DATASET:
            args.imgs = sorted(base_path.joinpath(f"{seq[1]:02d}-S").glob("*.png"))
            args.likelis = sorted(base_path.joinpath(f"{seq[1]:02d}_GT/LIK-S").glob("*.tif"))
        else:
            args.imgs = sorted(base_path.joinpath(f"{seq[1]:02d}").glob("*.tif"))
            args.likelis = sorted(base_path.joinpath(f"{seq[1]:02d}_GT/LIK").glob("*.tif"))

        if args.dataset == "PhC-C2DL-PSC":
            args.imgs = args.imgs[150:250]

        args.weight_path = f"./weight/super/{args.dataset}/{seq[0]:02d}/best.pth"

        args.output_path = Path(f"./output/detection/super_png/{args.dataset}/{seq[1]:02d}")

        args.output_path.mkdir(parents=True, exist_ok=True)

        args.fm = True
        call_predict(args)


def unsupervised_pred(args):
    for seq in [1, 2]:
        base_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{args.dataset}")
        if args.dataset in SCALED_DATASET:
            args.imgs = sorted(base_path.joinpath(f"{seq:02d}-S").glob("*.png"))
            args.likelis = sorted(base_path.joinpath(f"{seq:02d}_GT/LIK-S").glob("*.tif"))
        else:
            args.imgs = sorted(base_path.joinpath(f"{seq:02d}").glob("*.tif"))
            args.likelis = sorted(base_path.joinpath(f"{seq:02d}_GT/LIK").glob("*.tif"))

        args.weight_path = f"weight/{args.mode}/{args.dataset}/{seq:02d}/best.pth"

        args.output_path = Path(f"./output/detection/{args.mode}/{args.dataset}/{seq:02d}")
        args.output_path.mkdir(parents=True, exist_ok=True)

        args.fm = False
        call_predict(args)


DATASETS = {
    # 4: "BF-C2DL-HSC",
    # 5: "BF-C2DL-MuSC",
    6: "DIC-C2DH-HeLa",
    # 7: "Fluo-C2DL-MSC",
    8: "Fluo-N2DH-GOWT1",
    9: "Fluo-N2DH-SIM+",
    10: "Fluo-N2DL-HeLa",
    11: "PhC-C2DH-U373",
    # 12: "PhC-C2DL-PSC",
    # 13: "ushi"
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

CTC_config = {"DIC-C2DH-HeLa": "s_scale_norm",
              "Fluo-C2DL-MSC": "l_scale_norm",
              "Fluo-N2DH-GOWT1": "s_scale_norm",
              "Fluo-N2DH-SIM+": "ori_norm",
              "Fluo-N2DL-HeLa": "s_scale",
              "PhC-C2DH-U373": "l_scale",
              "PhC-C2DL-PSC": "l_scale"}

if __name__ == "__main__":
    args = parse_args()
    # for args.mode in ["pseudo", "pseudo_pre"]:
    #     print(args.mode)

    for args.dataset in DATASETS.values():
        supervised_test(args)

            # unsupervised_pred(args)
