import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDispNet(nn.Module):
    def __init__(self):
        super(SimpleDispNet, self).__init__()
        # encoder: reduce H×W by 8×
        self.enc1 = nn.Conv2d(6,  32, kernel_size=7, stride=2, padding=3)  # → 32×H/2×W/2
        self.enc2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # → 64×H/4×W/4
        self.enc3 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1)  # →128×H/8×W/8

        # decoder: back up to original H×W
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # →64×H/4×W/4
        self.dec2 = nn.ConvTranspose2d(64,  32, kernel_size=4, stride=2, padding=1)  # →32×H/2×W/2
        self.dec1 = nn.ConvTranspose2d(32,  16, kernel_size=4, stride=2, padding=1)  # →16×H×W

        # final 1‑channel disparity
        self.out  = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # non‑linearities
        self.relu = nn.ReLU(inplace=True)

    def forward(self, imgL, imgR):
        # [B,3,H,W] each → concat → [B,6,H,W]
        x = torch.cat([imgL, imgR], dim=1)

        # encode
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))

        # decode
        x = self.relu(self.dec3(x))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec1(x))

        # predict disparity
        disp = self.out(x)    # [B,1,H,W]
        return disp