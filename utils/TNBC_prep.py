#%%

from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt 

from matching import gaus_filter


img_dirs = sorted(Path('/home/kazuya/dataset/Cancer2/TNBC_NucleiSegmentation/TNBC_NucleiSegmentation').glob("Slide*/*png"))
gt_dirs = sorted(Path('/home/kazuya/dataset/Cancer2/TNBC_NucleiSegmentation/TNBC_NucleiSegmentation').glob("GT*/*png"))
save_dir = Path("/home/kazuya/dataset/Cancer2/TNBC_NucleiSegmentation")
save_dir.joinpath("cell_pos").mkdir(parents=True, exist_ok=True)



for idx, (img_path, gt_path) in enumerate(zip(img_dirs, gt_dirs)):
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(gt_path))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    nLabels, labelImages = cv2.connectedComponents(mask)
    plots = []
    for label in range(1, nLabels+1):
        x, y = np.where(labelImages == label)
        x_center = x.mean()
        y_center = y.mean()
        plots.append([idx,x_center,y_center])
    
    plots = np.array(plots)
    plots = plots[~np.isnan(plots).any(axis=1)]


    # plt.imshow(img), plt.plot(plots[:,2], plots[:,1], 'rx'), plt.savefig("test.png")

    save_dir.joinpath(f"{img_path.parent.stem}/cell_pos").mkdir(parents=True, exist_ok=True)
    save_dir.joinpath(f"{img_path.parent.stem}/ori").mkdir(parents=True, exist_ok=True)
    save_dir.joinpath(f"{img_path.parent.stem}/likeli").mkdir(parents=True, exist_ok=True)
    np.savetxt(str(save_dir.joinpath(f"{img_path.parent.stem}/cell_pos/{idx:04d}.txt")), plots, fmt="%d", delimiter=",")

    black = np.zeros((512, 512))
    result = black.copy()

    # 1013 - number of frame
    for _, x, y in plots:
        img_t = black.copy()  # likelihood map of one cell
        img_t[int(x)][int(y)] = 255  # plot a white dot
        img_t = gaus_filter(img_t, 301, 6)
        result = np.maximum(result, img_t)  # compare result with gaussian_img
    #  normalization
    result = 255 * result / result.max()
    result = result.astype("uint8")

    cv2.imwrite(str(save_dir.joinpath(f"{img_path.parent.stem}/ori/{idx:04d}.png")), img)
    cv2.imwrite(str(save_dir.joinpath(f"{img_path.parent.stem}/likeli/{idx:04d}.png")), result)
    print("finish")



