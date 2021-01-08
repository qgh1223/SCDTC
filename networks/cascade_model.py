from .network_parts import *
from .network_model import UNet


class CascadeUB(UNet):
    def __init__(self, n_channels, n_classes):
        super().__init__(2, 1)
        self.unet = UNet(n_channels, 1)
        self.up1_boundary = UpIncBoundary(512, 512, 256)
        self.up2_boundary = UpIncBoundary(512, 256, 128)
        self.up3_boundary = UpIncBoundary(256, 128, 64)
        self.up4_boundary = UpIncBoundary(128, 64, 64)
        self.outc_boundary = Outconv(64, n_classes)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = Outconv(128, n_classes)

    def forward(self, x):
        detection = self.unet(x)
        x = torch.cat((x, detection), dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_boundary = self.up1(x5, x4)
        x = self.up1_boundary(x5, x4, x_boundary)
        x_boundary = self.up2(x_boundary, x3)
        x = self.up2_boundary(x, x3, x_boundary)
        x_boundary = self.up3(x_boundary, x2)
        x = self.up3_boundary(x, x2, x_boundary)
        x_boundary = self.up4(x_boundary, x1)
        x = self.up4_boundary(x, x1, x_boundary)
        x_boundary = self.outc_boundary(x_boundary)
        x = self.outc(x)

        return detection, x, x_boundary

class CascadeUU(UNet):
    def __init__(self, n_channels, n_classes):
        super().__init__(2, 1)
        self.unetd = UNet(n_channels, 1)
        self.unets = UNet(2, n_classes)

    def forward(self, x):
        detection = self.unetd(x)
        x = torch.cat((x, detection), dim=1)
        segmentation = self.unets(x)
        return detection, segmentation

