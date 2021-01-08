from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mask_RGB_paths = sorted(Path("/home/kazuya/dataset/review/riken/mask_color").glob("*.PNG"))
    save_path = Path("/home/kazuya/dataset/review/riken/mask")
    save_path.mkdir(parents=True, exist_ok=True)

    for mask_RGB_path in mask_RGB_paths:
        mask_RGB = np.array(Image.open(mask_RGB_path))
        mask_R = np.zeros((mask_RGB.shape[0], mask_RGB.shape[1]))
        mask_G = np.zeros((mask_RGB.shape[0], mask_RGB.shape[1]))
        mask_B = np.zeros((mask_RGB.shape[0], mask_RGB.shape[1]))
        mask_R[(mask_RGB[:, :, 0] == 255) & (mask_RGB[:, :, 1] != 255) & (mask_RGB[:, :, 2] != 255)] = 255
        mask_G[(mask_RGB[:, :, 0] != 255) & (mask_RGB[:, :, 1] == 255) & (mask_RGB[:, :, 2] != 255)] = 255
        mask_B[(mask_RGB[:, :, 0] != 255) & (mask_RGB[:, :, 1] != 255) & (mask_RGB[:, :, 2] == 255)] = 255

        labelsr, retr = cv2.connectedComponents(mask_R.astype(np.uint8))
        labelsg, retg = cv2.connectedComponents(mask_G.astype(np.uint8))
        retg[retg != 0] = retg[retg != 0] + (labelsr - 1)
        ret = retr + retg
        labels = labelsr + labelsg - 2
        labelsb, retb = cv2.connectedComponents(mask_B.astype(np.uint8))
        retb[retb != 0] = retb[retb != 0] + labels

        ret = ret + retb
        cv2.imwrite(str(save_path.joinpath(mask_RGB_path.stem + ".png")), ret)