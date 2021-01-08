from pathlib import Path
from PIL import Image
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import cv2


if __name__ == '__main__':
    for sam_idx in range(1, 6):
        pred_paths = sorted(Path(f"/home/kazuya/main/WSISPDR/outputs/detection/ushi/test_data/sample{sam_idx}/pred").glob("*.tif"))
        save_path = Path(f"/home/kazuya/main/WSISPDR/outputs/detection/ushi/test_data/sample{sam_idx}/plot")
        save_path.mkdir(parents=True, exist_ok=True)

        pos_list = np.zeros((0, 3))
        for idx, pred_path in enumerate(pred_paths):
            pred = np.array(Image.open(pred_path))
            pred[(pred.max() * 0.8) > pred] = 0
            pos = peak_local_max(pred, min_distance=10)
            peak_img = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.uint8)
            for x, y in pos:
                peak_img[x, y] = 255
            labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
            pos_new = np.zeros((0, 2))
            for j in range(1, labels):
                pos_new = np.append(pos_new, [[center[j, 0], center[j, 1]]], axis=0)
            plt.imshow(pred), plt.plot(pos_new[:, 0], pos_new[:, 1], "rx"), plt.savefig(str(save_path.joinpath(f"{idx:05d}.tif"))), plt.close()
            try:
                pos_list_tmp = np.concatenate((pos_new, np.full((pos_new.shape[0], 1), idx)), axis=1)
            except:
                print(1)
            pos_list = np.append(pos_list, pos_list_tmp, axis=0)

        np.savetxt(str(save_path.parent.joinpath("plot.txt")), pos_list, fmt="%d")


