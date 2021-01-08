import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path

def AppendTracklet(cell, track_let, track_part):
    id = int(cell.get('id'))
    if len(cell) == 1:
        track_info = track_part.copy()
        track_info[0, 1] = id
        for info in cell.find('.//ss'):
            track_info[0, 0] = int(info.get('i'))
            track_info[0, 2] = int(float(info.get('x')))
            track_info[0, 3] = int(float(info.get('y')))
            track_let = np.append(track_let, track_info, axis=0)
        return track_let

    if len(cell) >= 2:
        track_info = track_part.copy()
        track_info[0, 1] = id
        for info in cell.find('.//ss'):
            track_info[0, 0] = int(info.get('i'))
            track_info[0, 2] = int(float(info.get('x')))
            track_info[0, 3] = int(float(info.get('y')))
            track_let = np.append(track_let, track_info, axis=0)
        if len(cell.find('.//as')) >= 2:
            for chil_cell in cell.find('.//as'):
                track_info = track_part.copy()
                track_info[0, -1] = id
                track_let = np.append(track_let,
                                      AppendTracklet(chil_cell, np.empty((0, 5)).astype('int32'), track_info), axis=0)
            return track_let
        else:
            return track_let

    else:
        print(cell)
        return track_let

if __name__ == '__main__':
    # for seq in [2, 3, 4, 6, 7, 8, 9, 10]:
    for seq in [6, 7, 8, 9]:
        # load xml file
        xml_path = Path(f"/home/kazuya/hdd/dataset/ushi_hai/sample{seq}.xml")
        save_path = Path("/home/kazuya/hdd/dataset/ushi_hai").joinpath(xml_path.stem + ".txt")
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        # number of cell
        num_cell = len(root[0][0][0].findall("a"))

        black = np.zeros((1040, 1392))
        track_let = np.empty((0, 5)).astype('int32')
        track_part = np.zeros((1, 5)).astype('int32')
        track_part[0, -1] = -1
        for par_cell in root[0][0][0].findall("a"):
            id = int(par_cell.get('id'))
            print(id)
            track_let = AppendTracklet(par_cell, track_let, track_part.copy())

        np.savetxt(str(save_path), track_let.astype('int'))