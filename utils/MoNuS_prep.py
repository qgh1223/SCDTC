
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy as np
import cv2
from pathlib import Path

from matching import gaus_filter

def monusegmask():
    for mode in ["Training", "Test"]:
        img_paths = sorted(Path(f"/home/kazuya/dataset/Cancer/MoNuSeg{mode}Data/ori").glob("T*"))
        xml_paths = sorted(Path(f"/home/kazuya/dataset/Cancer/MoNuSeg{mode}Data/xml").glob("T*.xml"))
        save_path = Path(f"/home/kazuya/dataset//Cancer/MoNuSeg{mode}Data/mask")
        save_path.mkdir(parents=True, exist_ok=True)
        
        for img_path, xml_path in zip(img_paths, xml_paths):
            img = cv2.imread(str(img_path))
            # plt.imshow(img), plt.show()

            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            
            annotations = []
            regions = root.findall("./**Region")
            for region in regions:
                Vertices = region.findall("./Vertices/Vertex")
                plots = []
                for Vertex in Vertices:
                    x = int(float(Vertex.attrib['X']))
                    y = int(float(Vertex.attrib['Y']))
                    plots.append([x, y])
                annotations.append(np.array(plots))
            
            mask = np.zeros_like(img, np.uint16)
            for idx, annotation in enumerate(annotations):
                mask = cv2.drawContours(mask, [annotation], 0, (idx+1, idx+1, idx+1), thickness=cv2.FILLED, maxLevel=1000)

            cv2.imwrite(str(save_path.joinpath(img_path.stem+".png")), mask.astype(np.uint16))

def position():
    for mode in ["Training", "Test"]:
        mask_path = Path(f"/home/kazuya/dataset/dataset/Cancer/MoNuSeg{mode}Data/mask")
        mask_paths = sorted(mask_path.glob("T*.png"))
        img_paths = sorted(Path(f"/home/kazuya/dataset/dataset/Cancer/MoNuSeg{mode}Data/ori").glob("T*"))
        save_path = mask_path.parent.joinpath("cell_pos")
        save_path.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in zip(img_paths, mask_paths):
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), -1)

            plots = []
            for idx in range(1, mask.max()+1):
                x, y, c = np.where(mask == idx)
                x_center = x.mean()
                y_center = y.mean()
                plots.append([idx, x_center, y_center])
            plots = np.array(plots)
            plots = plots[~np.isnan(plots).any(axis=1)]
            # plt.imshow(img), plt.plot(plots[:,1], plots[:, 2], 'r1'), plt.show()
            # plt.imshow(mask), plt.show()
            np.savetxt(str(save_path.joinpath(f"{img_path.stem}.txt")), plots, fmt="%d", delimiter=",")
            # break


def likeli():
    for mode in ["Training", "Test"]:
        pos_path = Path(f"/home/kazuya/dataset/dataset/Cancer/MoNuSeg{mode}Data/cell_pos")
        pos_paths= sorted(pos_path.glob("*.txt"))

        output_path = Path(f"/home/kazuya/dataset/dataset/Cancer/MoNuSeg{mode}Data/likeli")
        output_path.mkdir(parents=True, exist_ok=True)

        black = np.zeros((1000, 1000))

        # load txt file
        for pos_path in pos_paths:
            cell_positions = np.loadtxt(pos_path, delimiter=",", skiprows=0)

            result = black.copy()

            # 1013 - number of frame
            for _, x, y in cell_positions:
                img_t = black.copy()  # likelihood map of one cell
                img_t[int(x)][int(y)] = 255  # plot a white dot
                img_t = gaus_filter(img_t, 301, 6)
                result = np.maximum(result, img_t)  # compare result with gaussian_img
            #  normalization
            result = 255 * result / result.max()
            result = result.astype("uint8")
            cv2.imwrite(str(output_path.joinpath(pos_path.stem + ".png")), result)
            print("finish")

if __name__ == "__main__":
    monusegmask()