"""Preserve LaneUNet-v1 parameter keys and operations for existing checkpoints."""

import torch
from torch import nn

CLASSES = ("background", "lane_left", "lane_right", "crosswalk", "speed_bump")
ARCHITECTURES = {4: "LaneUNet-v1", 5: "LaneUNet-v1-5class"}


class Block(nn.Module):
    def __init__(self, inside, outside):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(inside, outside, 3, padding=1, bias=False),
            nn.BatchNorm2d(outside), nn.ReLU(inplace=True),
            nn.Conv2d(outside, outside, 3, padding=1, bias=False),
            nn.BatchNorm2d(outside), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class LaneUNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        if num_classes not in ARCHITECTURES:
            raise ValueError("LaneUNet supports 4 or 5 classes")
        self.enc1 = Block(3, 16)
        self.enc2 = Block(16, 32)
        self.enc3 = Block(32, 64)
        self.enc4 = Block(64, 128)
        self.middle = Block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = Block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = Block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = Block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = Block(32, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        d = self.enc4(self.pool(c))
        x = self.middle(self.pool(d))
        x = self.dec4(torch.cat((self.up4(x), d), dim=1))
        x = self.dec3(torch.cat((self.up3(x), c), dim=1))
        x = self.dec2(torch.cat((self.up2(x), b), dim=1))
        x = self.dec1(torch.cat((self.up1(x), a), dim=1))
        return self.head(x)
