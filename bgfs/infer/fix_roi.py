from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

from diffusers import AutoencoderKLWan

from bgfs.geometry.mediapipe_extract import extract_landmarks_holistic
from bgfs.geometry.heatmaps import softargmax2d
from bgfs.geometry.kinematics import hand_length_consistency_loss, hand_joint_angle_smooth_loss
from bgfs.models.critic import HeatmapCritic


Region = Literal["hands", "face", "feet", "upper_body", "full_body"]


@dataclass
class FixerConfig:
    region: Region = "hands"
    # optimization
    steps: int = 25
    lr: float = 0.07
    strength: float = 1.0  # anatomy loss weight
    keep_weight: float = 0.05  # keep close to original
    temporal_weight: float = 0.10  # temporal smoothness in ROI
    roi_size: int = 256


def _bbox_from_hand_landmarks(lm: np.ndarray, vis: np.ndarray, H: int, W: int, margin: float = 0.25):
    if lm is None or vis is None:
        return None
    good = vis > 0.2
    if not np.any(good):
        return None
    xy = lm[good, :2]
    xs = xy[:, 0] * W
    ys = xy[:, 1] * H
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = (x1 - x0) * (1.0 + margin)
    bh = (y1 - y0) * (1.0 + margin)
    bw = max(bw, 16.0)
    bh = max(bh, 16.0)
    x0 = int(max(0, cx - bw / 2))
    x1 = int(min(W - 1, cx + bw / 2))
    y0 = int(max(0, cy - bh / 2))
    y1 = int(min(H - 1, cy + bh / 2))
    return x0, y0, x1, y1


def _get_hand_bboxes(frames_bgr: np.ndarray):
    """Return per-frame bbox = (x0,y0,x1,y1) for left and right hands."""
    T, H, W, _ = frames_bgr.shape
    bboxes = []
    for t in tqdm(range(T), desc="Detecting hands", leave=False):
        rgb = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2RGB)
        res = extract_landmarks_holistic(rgb)
        bb_l = _bbox_from_hand_landmarks(res.left_hand, res.left_hand_vis, H, W)
        bb_r = _bbox_from_hand_landmarks(res.right_hand, res.right_hand_vis, H, W)
        # fallback: if one is missing, reuse the other
        bboxes.append((bb_l, bb_r))
    return bboxes


def _crop_resize(frame: torch.Tensor, bbox: Tuple[int, int, int, int], size: int) -> torch.Tensor:
    # frame: (3,H,W)
    x0, y0, x1, y1 = bbox
    crop = frame[:, y0:y1, x0:x1]
    crop = crop.unsqueeze(0)
    crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
    return crop.squeeze(0)


def _paste_resize(dst: torch.Tensor, src_crop: torch.Tensor, bbox: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = bbox
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    src = src_crop.unsqueeze(0)
    src = F.interpolate(src, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    dst[:, y0:y1, x0:x1] = src


def fix_video_hands(
    in_mp4: Path,
    out_mp4: Path,
    model_id_for_vae: str,
    critic_ckpt: Path,
    cfg: FixerConfig = FixerConfig(),
    device: str = "cuda",
):
    """Post-hoc ROI latent optimization guided by a differentiable keypoint critic.

    This does NOT retrain Wan. It takes a generated video, finds hands, and optimizes VAE latents
    so decoded crops have more plausible hand geometry.
    """
    import imageio.v3 as iio

    frames = iio.imread(str(in_mp4), plugin="ffmpeg")  # (T,H,W,3) RGB
    if frames.ndim != 4:
        raise ValueError("Expected (T,H,W,3) frames")
    # work in BGR for OpenCV/mediapipe convenience
    frames_bgr = frames[..., ::-1].copy()
    T, H, W, _ = frames_bgr.shape

    bboxes = _get_hand_bboxes(frames_bgr)

    # load VAE + critic
    vae = AutoencoderKLWan.from_pretrained(model_id_for_vae, subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.eval()

    ck = torch.load(critic_ckpt, map_location="cpu")
    K = int(ck.get("K", 21))
    critic = HeatmapCritic(num_keypoints=K).to(device)
    critic.load_state_dict(ck["state_dict"], strict=True)
    critic.eval()

    # optimize per-frame hand crops (left and right separately) then paste back
    fixed = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)  # (T,3,H,W)
    orig = fixed.detach().clone()

    def anatomy_loss(crop: torch.Tensor) -> torch.Tensor:
        # crop: (3,S,S)
        hm = critic(crop.unsqueeze(0))  # (1,K,S,S)
        pts = softargmax2d(hm)  # (1,K,2)
        # normalize by S
        pts = pts / float(cfg.roi_size)
        pts = pts.squeeze(0)
        # two simple priors
        return hand_length_consistency_loss(pts) + 0.25 * hand_joint_angle_smooth_loss(pts)

    # iterate frames with temporal coupling by optimizing blocks
    for t in tqdm(range(T), desc="Fixing hands"):
        bb_l, bb_r = bboxes[t]
        for side, bb in [("L", bb_l), ("R", bb_r)]:
            if bb is None:
                continue
            crop = _crop_resize(fixed[t], bb, cfg.roi_size)
            crop0 = crop.detach().clone()

            # encode to latents
            with torch.no_grad():
                lat = vae.encode(crop.unsqueeze(0) * 2 - 1).latent_dist.sample()
                lat = lat * vae.config.scaling_factor

            lat = lat.detach().requires_grad_(True)
            opt = torch.optim.Adam([lat], lr=cfg.lr)

            for _ in range(cfg.steps):
                dec = vae.decode(lat / vae.config.scaling_factor).sample
                dec = (dec + 1) / 2
                dec = dec.clamp(0, 1).squeeze(0)

                loss_a = anatomy_loss(dec) * cfg.strength
                loss_keep = F.mse_loss(dec, crop0) * cfg.keep_weight

                # temporal smoothness: encourage similarity to previous fixed frame crop
                loss_temp = torch.tensor(0.0, device=device)
                if t > 0:
                    prev_bb_l, prev_bb_r = bboxes[t - 1]
                    prev_bb = prev_bb_l if side == "L" else prev_bb_r
                    if prev_bb is not None:
                        prev_crop = _crop_resize(fixed[t - 1].detach(), prev_bb, cfg.roi_size)
                        loss_temp = F.mse_loss(dec, prev_crop) * cfg.temporal_weight

                loss = loss_a + loss_keep + loss_temp
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            # write back decoded crop
            with torch.no_grad():
                dec = vae.decode(lat / vae.config.scaling_factor).sample
                dec = (dec + 1) / 2
                dec = dec.clamp(0, 1).squeeze(0)
                _paste_resize(fixed[t], dec, bb)

    out_frames = (fixed.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    # RGB
    iio.imwrite(str(out_mp4), out_frames, plugin="ffmpeg", fps=16)
    return out_mp4
