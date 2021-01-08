import cv2
from PIL import Image
from pathlib import Path
import numpy as np
from matching import gaus_filter
import os

if __name__ == '__main__':
    # img_dirs = sorted([p for p in Path("/home/kazuya/dataset/ushi_hai/additional").iterdir() if p.is_dir()])
    img_dirs = sorted([p for p in Path("/home/kazuya/dataset/ushi_hai/additional").glob("sample*") if p.is_dir()])
    for img_dir in img_dirs:
        # dir_name = img_dir
        # os.rename(img_dir, img_dir.parent.joinpath("img"))
        # dir_name.mkdir(parents=True, exist_ok=True)
        # os.rename(img_dir.parent.joinpath("img"), img_dir.joinpath("img"))
        gaus_var = 12
        save_path = Path(f"/home/kazuya/dataset/ushi_pre/additionnal/{img_dir.name}")
        save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
        save_path.joinpath(f"{gaus_var}").mkdir(parents=True, exist_ok=True)

        img_paths = sorted(img_dir.joinpath("img").glob("*.bmp"))

        for i, img_path in enumerate(img_paths):
            # if i > samp_max[sample_idx]:
            #     break
            img = Image.open(img_path)
            img = img.resize((256, 256))
            img.save(save_path.joinpath(f"img/{img_path.name}"))

            scale_v = 256 / Image.open(img_path).size[0]
            scale_h = 256 / Image.open(img_path).size[1]

        if img_dir.parent.joinpath(img_dir.name + ".txt").exists():
            cell_positions = np.loadtxt(str(img_dir.parent.joinpath(img_dir.name + ".txt")), skiprows=3)
            black = np.zeros((256, 256))

                # 1013 - number of frame
            for i in range(0, int(cell_positions[:, 0].max())):
                # likelihood map of one input
                result = black.copy()
                cells = cell_positions[cell_positions[:, 2] == i]
                for x, y, _, _ in cells:
                    x = int(x * scale_v)
                    y = int(y * scale_h)
                    img_t = black.copy()  # likelihood map of one cell
                    img_t[int(y)][int(x)] = 255  # plot a white dot
                    img_t = gaus_filter(img_t, 81, gaus_var)
                    result = np.maximum(result, img_t)  # compare result with gaussian_img
                #  normalization
                result = 255 * result / result.max()
                result = result.astype("uint8")
                cv2.imwrite(str(save_path.joinpath(f"{gaus_var}/{i:05d}.png")), result)
                print(i + 1)
        print("finish")
