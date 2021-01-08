from pathlib import Path
import cv2
from skimage import measure
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import numpy as np
from utils import gaus_filter

if __name__ == "__main__":
    img_paths = sorted(Path("/home/kazuya/dataset/200814-200815-hMSCCells-phasehoechst and phase 24hours timelapse").glob("*/*_C0001.tif"))
    flu_paths = sorted(Path("/home/kazuya/dataset/200814-200815-hMSCCells-phasehoechst and phase 24hours timelapse").glob("*/*_C0002.tif"))

    save_path = Path("/home/kazuya/dataset/hMSC2")
    save_path.joinpath("ori").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("6").mkdir(parents=True, exist_ok=True)

    for i, (img_path, flu_path) in enumerate(zip(img_paths, flu_paths)):
        img = cv2.imread(str(img_path), 0)
        flu = cv2.imread(str(flu_path), 0)
        # plt.imshow(flu), plt.show()
        flu = cv2.GaussianBlur(flu, (7, 7), 1.5)
        flu = (flu - flu.min()) / (flu.max() - flu.min())
        flu[flu < 0.8] = 0
        flu[flu >= 0.8] = 255
        _, _, _, plot = cv2.connectedComponentsWithStats(flu.astype(np.uint8))

        # plot = peak_local_max(flu, min_distance=20, threshold_rel=0.8)

        result = np.zeros((img.shape[0], img.shape[1]))

        for y, x in plot:
            img_t = np.zeros((img.shape[0], img.shape[1]))  # likelihood map of one cell
            img_t[int(x)][int(y)] = 255  # plot a white dot
            img_t = gaus_filter(img_t, 101, 6)
            result = np.maximum(result, img_t)  # compare result with gaussian_img
        #  normalization
        result = 255 * result / result.max()
        result = result.astype("uint8")
        cv2.imwrite(str(save_path / Path("6/%05d.tif" % i)), result)
        cv2.imwrite(str(save_path / Path("ori/%05d.tif" % i)), img)
        print(i + 1)




