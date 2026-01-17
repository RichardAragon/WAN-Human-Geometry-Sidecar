from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzRegionDataset(Dataset):
    def __init__(self, manifest_json: Path, split: str = "train", val_split: float = 0.05, seed: int = 123):
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
        self.items: List[Dict] = manifest
        # deterministic shuffle
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.items)).tolist()
        self.items = [self.items[i] for i in idx]
        n_val = max(1, int(round(len(self.items) * val_split)))
        if split == "val":
            self.items = self.items[:n_val]
        else:
            self.items = self.items[n_val:]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.items[i]["npz"])
        z = np.load(path)
        roi = z["roi"]  # (T,H,W,3) uint8
        hms = z["heatmaps"]  # (T,K,H,W) float32

        # random frame sample for critic pretrain
        t = np.random.randint(0, roi.shape[0])
        x = roi[t].astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        y = hms[t].astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)
