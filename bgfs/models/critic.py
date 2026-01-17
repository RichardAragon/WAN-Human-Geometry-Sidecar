from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p)
        self.gn = nn.GroupNorm(num_groups=min(32, cout), num_channels=cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gn(self.conv(x)))


class HeatmapCritic(nn.Module):
    """Lightweight CNN that predicts keypoint heatmaps from ROI crops.

    This is intentionally small: it’s a differentiable proxy for MediaPipe
    so we can backprop anatomy losses into diffusion training.
    """

    def __init__(self, num_keypoints: int, in_channels: int = 3):
        super().__init__()
        c = 64
        self.enc1 = nn.Sequential(ConvBlock(in_channels, c), ConvBlock(c, c))
        self.enc2 = nn.Sequential(ConvBlock(c, c * 2, s=2), ConvBlock(c * 2, c * 2))
        self.enc3 = nn.Sequential(ConvBlock(c * 2, c * 4, s=2), ConvBlock(c * 4, c * 4))
        self.dec2 = nn.Sequential(ConvBlock(c * 4 + c * 2, c * 2), ConvBlock(c * 2, c * 2))
        self.dec1 = nn.Sequential(ConvBlock(c * 2 + c, c), ConvBlock(c, c))
        self.head = nn.Conv2d(c, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        u2 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.head(d1)
