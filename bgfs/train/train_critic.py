from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from bgfs.config import TrainConfig
from bgfs.data.npz_dataset import NpzRegionDataset
from bgfs.models.critic import HeatmapCritic


def train_critic(cfg: TrainConfig, manifest_json: Path, out_dir: Path, out_hw: int = 256, epochs: int = 5, lr: float = 3e-4) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_tr = NpzRegionDataset(manifest_json, split="train", val_split=cfg.data.val_split, seed=cfg.seed)
    ds_va = NpzRegionDataset(manifest_json, split="val", val_split=cfg.data.val_split, seed=cfg.seed)

    # infer K from one sample
    _, y0 = ds_tr[0]
    K = y0.shape[0]

    model = HeatmapCritic(num_keypoints=K)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    acc = Accelerator(mixed_precision=cfg.compute.mixed_precision)
    tr_loader = DataLoader(ds_tr, batch_size=32, shuffle=True, num_workers=cfg.compute.num_workers, pin_memory=True)
    va_loader = DataLoader(ds_va, batch_size=32, shuffle=False, num_workers=cfg.compute.num_workers, pin_memory=True)

    model, opt, tr_loader, va_loader = acc.prepare(model, opt, tr_loader, va_loader)

    best = 1e9
    for ep in range(epochs):
        model.train()
        pbar = tqdm(tr_loader, disable=not acc.is_local_main_process, desc=f"Critic train ep {ep+1}/{epochs}")
        for x, y in pbar:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            acc.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": float(loss.detach().cpu())})

        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for x, y in va_loader:
                pred = model(x)
                loss = F.mse_loss(pred, y)
                tot += float(loss.detach().cpu())
                n += 1
        val = tot / max(1, n)

        if acc.is_local_main_process:
            ckpt = out_dir / f"critic_ep{ep+1}.pt"
            acc.unwrap_model(model).cpu()
            torch.save({"state_dict": acc.unwrap_model(model).state_dict(), "K": K}, ckpt)
            acc.unwrap_model(model).to(acc.device)
            if val < best:
                best = val
                best_path = out_dir / "critic_best.pt"
                torch.save({"state_dict": acc.unwrap_model(model).state_dict(), "K": K}, best_path)

    return out_dir / "critic_best.pt"
