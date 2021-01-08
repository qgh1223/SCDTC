from types import MethodType
import torch.nn as nn
from .guided_parts import guide_relu
from utils import gaus_filter, local_maxim
from scipy.io import savemat
import numpy as np
import cv2
from torch.nn.modules import Module
from collections import OrderedDict
from itertools import islice
import operator
import torch
import skimage
import matplotlib.pyplot as plt

SIG_LIST = {"BF-C2DL-HSC": 6,
            "BF-C2DL-MuSC": 9,
            "DIC-C2DH-HeLa": 9,
            "PhC-C2DH-U373": 9,
            "PhC-C2DL-PSC": 6,
            "Fluo-C2DL-MSC": 9,
            "Fluo-N2DH-GOWT1": 9,
            "Fluo-N2DH-SIM+": 9,
            "Fluo-N2DL-HeLa": 9,
            "C2C12": 9,
            "B23P17": 12,
            "Elmer": 3,
            "GBM": 9,
            "hMSC": 9,
            "riken": 9}


class Sequ(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequ, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequ, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input, input2):
        for module in self._modules.values():
            input = module(input, input2)
        return input


class GuidedModel(nn.Sequential):
    def __init__(self, net, args):
        super().__init__(net)
        self.inferencing = False
        self.shape = None
        self.dataset = args.dataset

        self.sigma = SIG_LIST[self.dataset]

        self.basis = np.zeros((self.sigma*6 + 1, self.sigma*6 + 1))
        self.basis[self.sigma*3, self.sigma*3] = 1
        self.basis = skimage.filters.gaussian(self.basis, sigma=self.sigma)
        self.basis = self.basis / self.basis.max()


    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module._original_forward = module.forward
                module.forward = MethodType(guide_relu, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.ReLU) and hasattr(module, "_original_forward"):
                module.forward = module._original_forward

    def forward(
            self,
            img,
            root_path,
            peak=None,
            class_threshold=0,
            peak_threshold=30,
            retrieval_cfg=None,
    ):
        assert img.dim() == 4, "PeakResponseMapping layer only supports batch mode."
        if self.inferencing:
            img.requires_grad_()

        # classification network forwarding
        class_response_maps = super().forward(img)
        # peak backpropagation
        # grad_output = mask
        ###
        # class_response_maps = class_response_maps["out"]
        # class_response_maps = torch.sigmoid(class_response_maps)
        # region = class_response_maps[0, 0].detach().cpu().numpy()
        #
        # cv2.imwrite(
        #         str(root_path.joinpath("detection.png")),
        #         (region * 255).astype(np.uint8),
        # )
        #
        # if img.grad is not None:
        #     img.grad.zero_()
        #
        # mask = np.zeros(region.shape, dtype=np.float32)
        # mask[region > 0.5] = 1
        # mask = mask.reshape([1, 1, region.shape[0], region.shape[1]])
        # mask = np.tile(mask, reps=(1, 3, 1, 1))
        # mask = torch.from_numpy(mask)
        # mask = mask.cuda()
        #
        # class_response_maps.backward(mask, retain_graph=True)
        # # class_response_maps["out"].backward(mask, retain_graph=True)
        # result = img.grad.detach().sum(1).clone().clamp(min=0).cpu().numpy()
        # savemat(str(root_path.joinpath("gb.mat")), {"gb": result[0], "mask": mask})
        # result = result[0] / result[0].max()
        # cv2.imwrite(str(root_path.joinpath("gb.png")), result)
        #
        # return result
        ###
        class_response_maps = torch.sigmoid(class_response_maps)
        pre_img = class_response_maps.detach().cpu().numpy()[0, 0]
        self.shape = pre_img.shape
        if peak is None:
            cv2.imwrite(
                str(root_path.joinpath("detection.png")),
                (pre_img * 255).astype(np.uint8),
            )

        # pre_img = cv2.imread("/home/kazuya/main/WSISPDR/detection.png", 0)
        # peak
        peaks = local_maxim((pre_img * 255).astype(np.uint8), 125, 2).astype(np.int)
        # peaks = local_maxim((pre_img).astype(np.uint8), 125, 2).astype(np.int)
        # if self.dataset not in ["MoNuSeg", "TNBC", "C2C12"]:
        #     if self.pad_kernel is not None:
        #         peaks = peaks[
        #             (peaks[:, 0] > self.pad_kernel[1][0]) & (peaks[:, 0] < pre_img.shape[0] - self.pad_kernel[1][0]) & (
        #                         peaks[:, 1] > self.pad_kernel[1][0]) & (
        #                         peaks[:, 1] < pre_img.shape[1] - self.pad_kernel[1][0])]

        if peaks.shape[0] > 0:
            gauses = []
            for label, peak in enumerate(peaks):
                tmp = np.zeros(self.shape)
                y = min(max(peak[0]-(self.sigma * 3 + 1), 0), self.shape[1] - (self.sigma * 6 + 1))
                x = min(max(peak[1]-(self.sigma * 3 + 1), 0), self.shape[0] - (self.sigma * 6 + 1))
                tmp[x:x+self.sigma * 6 + 1, y:y + self.sigma*6 + 1] = self.basis
                gauses.append(tmp)
            gauses = np.array(gauses)
            region = np.argmax(gauses, axis=0) + 1
            likely_map = np.max(gauses, axis=0)
            region[likely_map < 0.01] = 0
        else:
            region = np.zeros(self.shape, dtype=np.uint8)
            # region = cv2.circle(
            #     region, (int(peak[0]), int(peak[1])), self.sigma * 3, label + 1, thickness=-1
            # )

        '''
        gauses = []
        try:
            for peak in peaks:
                temp = np.zeros(self.shape)
                temp[peak[1], peak[0]] = 255
                gauses.append(gaus_filter(temp, 401, 12))
            region = np.argmax(gauses, axis=0) + 1
            likely_map = np.max(gauses, axis=0)
            region[likely_map < 0.01] = 0
            #
            # r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
        except ValueError:
            region = np.zeros(self.shape, dtype=np.uint8)
            likely_map = np.zeros(self.shape)
        '''

        gbs = []
        # each propagate
        peaks = np.insert(peaks, 0, [0, 0], axis=0)
        with open(root_path.joinpath("peaks.txt"), mode="w") as f:
            f.write("ID,x,y\n")
            for i in range(0, int(region.max()) + 1):
                if img.grad is not None:
                    img.grad.zero_()
                # f.write(f"{i},{peaks[i, 0]},{peaks[i ,1]}\n")
                f.write("{},{},{}\n".format(i, peaks[i, 0], peaks[i, 1]))
                mask = np.zeros(self.shape, dtype=np.float32)
                mask[region == i] = 1
                mask = mask.reshape([1, 1, self.shape[0], self.shape[1]])
                mask = torch.from_numpy(mask)
                mask = mask.cuda()

                class_response_maps.backward(mask, retain_graph=True)
                result = img.grad.detach().sum(1).clone().clamp(min=0).cpu().numpy()

                save_path = root_path.joinpath("each_peak")
                save_path.mkdir(parents=True, exist_ok=True)
                # np.savez_compressed(
                #     str(save_path.joinpath("{:04d}".format(i))),
                #     result[0], mask.detach().cpu().numpy(),
                # )
                savemat(
                    str(save_path.joinpath("{:04d}.mat".format(i))),
                    {"gb": result[0], "mask": mask},
                )
                gbs.append(result[0])
        return gbs

    def train(self, mode=True):
        super().train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super().train(False)
        self._patch()
        self.inferencing = True
        return self
