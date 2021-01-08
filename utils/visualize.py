from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

x = Path("/home/kazuya/main/WSISPDR/outputs/guided/BF-C2DL-HSC/02/01763/each_peak").glob("*.mat")
gbs = []
for path in x:
    gb = loadmat(str(path))["gb"]
    xxx = gb[gb > (gb.max() * 0.01)].flatten()
    xxx.sort()
    norm_value = xxx[round(xxx.shape[0] * 0.99)]
    gb = gb / norm_value
    gbs.append(gb)
gbs_np = np.array(gbs)

gbs_0 = gbs_np.copy()
gbs_0[gbs_0 < (gbs_0.max() * 0.01)] = 0
gbs_0_arg = np.argmax(gbs_0, axis=0)
for idx in range(len(gbs)):
    plt.imshow(gbs_0_arg == idx), plt.show()
