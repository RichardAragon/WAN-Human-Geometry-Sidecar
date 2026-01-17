from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def keypoints_to_heatmaps(
    keypoints_xy: np.ndarray,
    vis: np.ndarray,
    out_hw: Tuple[int, int],
    sigma: float = 2.0,
) -> np.ndarray:
    """Create 2D gaussian heatmaps.

    keypoints_xy: (K,2) in pixel coords (not normalized)
    vis: (K,) in [0,1]
    out_hw: (H,W)

    Returns: (K,H,W)
    """
    K = keypoints_xy.shape[0]
    H, W = out_hw
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    hms = np.zeros((K, H, W), dtype=np.float32)
    for k in range(K):
        if vis[k] <= 0.0:
            continue
        x, y = keypoints_xy[k]
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        hms[k] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma * sigma)) * float(vis[k])
    return hms


def softargmax2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """Convert heatmaps to coordinates with softargmax.

    heatmaps: (B,K,H,W)
    Returns: (B,K,2) in pixel coords (x,y)
    """
    B, K, H, W = heatmaps.shape
    hm = heatmaps.reshape(B, K, H * W)
    probs = torch.softmax(hm, dim=-1)
    idx = torch.arange(H * W, device=heatmaps.device, dtype=probs.dtype)
    exp = torch.sum(probs * idx[None, None, :], dim=-1)
    y = torch.floor(exp / W)
    x = exp - y * W
    return torch.stack([x, y], dim=-1)
