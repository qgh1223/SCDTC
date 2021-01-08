import torch.nn as nn
import torch
import torch.nn.functional as F


class BoundaryEnhancedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=1.3, alf=0.5, beta=0.8):
        super().__init__()
        self.weight = weight
        self.alf = alf
        self.beta = beta
        self.sigma = 0.001

    def forward(self, inputs, boundaries, targets):
        boundary_enhance = self.alf * torch.clamp(self.beta - boundaries, max=0)
        loss = (1 + boundary_enhance) * targets * torch.log(inputs + self.sigma) + self.weight * (
                1 - targets
        ) * torch.log(1 - inputs + self.sigma)
        return -torch.mean(loss)


class PixelwiseWBCELoss(nn.Module):
    def __init__(self):
        super(PixelwiseWBCELoss, self).__init__()

    def forward(self, input, target, wmp):
        return F.binary_cross_entropy(input, target, weight=wmp)


class MaskMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        pred = pred[mask == 1]
        target = target[mask == 1]
        return F.mse_loss(pred, target)


class focal_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 4
        self.alpha = 2

    def forward(self, input, target):
        loss_f = (1 - input[target == 1]) ** self.beta * torch.log(input[target == 1])
        loss_b = (1 - target[target != 1]) ** self.beta * (input[target != 1]) ** self.alpha * torch.log(
            1 - input[target != 1])
        return -(loss_b.sum() + loss_f.sum()) / (len(target[target == 1]) + 0.00001)


if __name__ == '__main__':
    import cv2
    import numpy as np

    img = cv2.imread("/home/kazuya/main/ushi_tracking/conf_eval.tif", 0)
    target = cv2.imread("/home/kazuya/main/ushi_tracking/conf_gt.tif", 0)
    img = (img / 255).astype(np.float32)
    target = (target / 255).astype(np.float32)
    import matplotlib.pyplot as plt

    plt.imshow(img), plt.show()
    plt.imshow(target), plt.show()
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    target = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)

    loss = focal_loss()
    loss(target, target)
    print(1)
