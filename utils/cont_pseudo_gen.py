import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matching import local_maxim, optimum
from utils import heatmap_gen


class ContCellSelect(object):
    def __init__(self, detect_th=125, peak_dist=10, debug=False, trac_len_th=3):
        self.detect_th = detect_th
        self.peak_dist = peak_dist
        self.debug = debug
        self.new_cell_id = 0
        self.trac_len_th = trac_len_th

    def det_cell(self, path):
        pred = cv2.imread(str(path[1]), 0)
        pred_plot = local_maxim(pred, self.detect_th, self.peak_dist)

        if self.debug:
            self.pred_plot = pred_plot
            self.img = cv2.imread(str(path[0]), 0)
            plt.imshow(self.img), plt.plot(pred_plot[:, 0], pred_plot[:, 1], "rx"), plt.show()
        return pred_plot

    def update_track(self, track_res, det, frame):
        # track_res_np = np.array(track_res)
        pre_det = track_res[track_res[:, 3] == frame - 1]
        assoc_ids = optimum(pre_det, det, 10)

        if self.debug:
            non_assoc = list(set(range(det.shape[0])) - set(assoc_ids[:, 1]))
            plt.imshow(self.img), plt.plot(det[non_assoc][:, 0], det[non_assoc][:, 1], "rx")
            plt.plot(det[list(assoc_ids[:, 1].astype(np.int))][:, 0], det[list(assoc_ids[:, 1].astype(np.int))][:, 1],
                     "b2"), plt.show()
        for assoc_id in assoc_ids:
            x = det[int(assoc_id[1])][0]
            y = det[int(assoc_id[1])][1]
            id = pre_det[int(assoc_id[0])][2]
            track_res = np.append(track_res, [[x, y, id, frame]], axis=0)
        for idx in set(range(det.shape[0])) - set(assoc_ids[:, 1]):
            track_res = np.append(track_res, [[det[idx][0], det[idx][1], self.new_cell_id, frame]], axis=0)
            self.new_cell_id += 1
        return track_res

    def select_pseudo(self, track_res):
        track_res_new = np.zeros((0, 4))
        for id in np.unique(track_res[:, 2]):
            if sum(track_res[:, 2] == id) > self.trac_len_th:
                track_res_new = np.append(track_res_new, track_res[track_res[:, 2] == id], axis=0)
        self.track_res = track_res
        return track_res_new

    def save_result(self, path, track_res_final, frame, save_path, save_path_img):
        img = cv2.imread(str(path[0]), 0)
        pred_plot_final = track_res_final[track_res_final[:, 3] == frame]
        pred_plot = self.track_res[self.track_res[:, 3] == frame]

        cv2.imwrite(str(save_path_img), img)

        plt.figure(figsize=(3, 3))
        plt.axis("off")
        plt.imshow(img, cmap="gray"), plt.plot(pred_plot[:, 0], pred_plot[:, 1], "r1", ms=12)
        plt.plot(pred_plot_final[:, 0], pred_plot_final[:, 1], "b2", ms=12)
        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0, trasparent=True)


def cont_pseudo_select(base_path, save_path, trac_len_th=3):
    """
    Select continuously detected pred position longer than a threshold
    :param path: path of a detection result
    :param path: path of save directly
    :trac_len int: threshold for determining how many consecutive counts
    """
    save_path.joinpath("detect_result").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
    img_paths = sorted(base_path.joinpath("ori").glob("*.tif"))
    lik_paths = sorted(base_path.joinpath("pred").glob("*.tif"))

    track_res = np.zeros((0, 4))
    # track_res = []
    Tracker = ContCellSelect(trac_len_th=trac_len_th)
    for frame, path in enumerate(zip(img_paths, lik_paths)):
        det_res = Tracker.det_cell(path)
        track_res = Tracker.update_track(track_res, det_res, frame)
    track_res_final = Tracker.select_pseudo(track_res)

    np.save(str(save_path.joinpath("track_res")), track_res_final)
    for frame, path in enumerate(zip(img_paths, lik_paths)):
        save_path_each = save_path.joinpath(f"detect_result/{frame:05d}.png")
        save_path_each_img = save_path.joinpath(f"img/{frame:05d}.tif")
        Tracker.save_result(path, track_res_final, frame, save_path_each, save_path_each_img)


def fg_pseudo_gen(base_path, track_res_path, gaus_size):
    img = cv2.imread(str(base_path.joinpath("ori/00000.tif")), 0)
    track_res = np.load(track_res_path)

    save_path = track_res_path.parent.joinpath("fg_pseudo")
    save_path.mkdir(parents=True, exist_ok=True)

    heatmap_gen(img.shape, track_res[:, [3, 0, 1]], gaus_size, save_path)


def back_mask_gen(base_path, save_path, local_len=3, bg_th=50, fg_th=50):
    pred_paths = sorted(base_path.joinpath("pred").glob("*.tif"))
    save_path = save_path.joinpath("bg_pseudo")
    fg_path = save_path.parent.joinpath("fg_pseudo")

    save_path.mkdir(parents=True, exist_ok=True)
    for idx in range(len(pred_paths) - local_len + 1):
        imgs = []
        for idx_l in range(local_len):
            img = cv2.imread(str(pred_paths[idx + idx_l]), 0)
            imgs.append(img)
        # img_base = imgs[int(local_len/2)]
        imgs = np.array(imgs)
        img_ave = imgs.mean(axis=0)

        fg = cv2.imread(str(save_path.parent.joinpath((f"fg_pseudo/{int(idx + local_len/2):05d}.tif"))), 0)
        mask = np.zeros_like(img_ave, dtype=np.uint8)
        mask[img_ave < bg_th] = 255
        # mask[np.abs(img_base - img_ave) < 10] = 255
        mask[fg > fg_th] = 255
        # plt.imshow(mask), plt.show()
        cv2.imwrite(str(save_path.joinpath(f"{int(idx + local_len/2):04d}.tif")), mask)


DATASETS = {
    # 1: "BF-C2DL-HSC",
    # 2: "BF-C2DL-MuSC",
    3: "DIC-C2DH-HeLa",
    # 4: "Fluo-C2DL-MSC",
    5: "Fluo-N2DH-GOWT1",
    6: "Fluo-N2DH-SIM+",
    7: "Fluo-N2DL-HeLa",
    8: "PhC-C2DH-U373",
    # 9: "PhC-C2DL-PSC",
}

GAUS_SIZE = {
    "BF-C2DL-HSC": 6,
    "BF-C2DL-MuSC": 9,
    "DIC-C2DH-HeLa": 9,
    "Fluo-C2DL-MSC": 9,
    "Fluo-N2DH-GOWT1": 9,
    "Fluo-N2DH-SIM+": 9,
    "Fluo-N2DL-HeLa": 6,
    "PhC-C2DH-U373": 9,
    "PhC-C2DL-PSC": 6,
}

#%%

if __name__ == '__main__':
    for dataset in DATASETS.values():
        for seq in [1, 2]:
            # for test_seq in [1, 2]:
            len_th = 3
            bg_th = 3
            base_path = Path(f"../output/detection/super/{dataset}/{seq:02d}")
            save_path = Path(f"../output/select_pseudo/test/{dataset}/{len_th}-{bg_th}/{seq:02d}")

            # # select accurate result
            cont_pseudo_select(base_path, save_path, trac_len_th=len_th)
            #
            # # gen fg pseudo mask
            fg_pseudo_gen(base_path, save_path.joinpath("track_res.npy"), GAUS_SIZE[dataset])

            # gen bg pseudo mask
            back_mask_gen(base_path, save_path, bg_th=bg_th)
