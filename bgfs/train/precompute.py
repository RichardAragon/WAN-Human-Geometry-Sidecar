from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

from bgfs.config import TrainConfig
from bgfs.data.video_io import list_videos, read_video_rgb, iter_clips
from bgfs.geometry.mediapipe_extract import extract_landmarks_holistic
from bgfs.geometry.heatmaps import keypoints_to_heatmaps


def _roi_from_landmarks(cfg: TrainConfig, lm, H: int, W: int) -> Tuple[int, int, int, int]:
    """Return (x0,y0,x1,y1) bbox in pixel coords."""
    # Default: hands = union of left+right hand landmarks if present, else use pose wrist indices.
    pts: List[Tuple[float, float]] = []

    def add(arr):
        if arr is None:
            return
        for x, y, _ in arr:
            pts.append((x * W, y * H))

    if cfg.region == "hands":
        add(lm.left_hand)
        add(lm.right_hand)
        if not pts and lm.pose is not None:
            # pose wrist indices: 15 left, 16 right
            for idx in (15, 16):
                x, y, _ = lm.pose[idx]
                pts.append((x * W, y * H))
    elif cfg.region == "face":
        add(lm.face)
        if not pts and lm.pose is not None:
            # approximate head from nose/eyes
            for idx in (0, 2, 5):
                x, y, _ = lm.pose[idx]
                pts.append((x * W, y * H))
    elif cfg.region in ("feet", "full_body", "upper_body"):
        add(lm.pose)

    if not pts:
        # fallback: center crop
        cx, cy = W // 2, H // 2
        s = min(H, W) // 2
        return max(cx - s // 2, 0), max(cy - s // 2, 0), min(cx + s // 2, W), min(cy + s // 2, H)

    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

    # margin
    bw, bh = x1 - x0, y1 - y0
    m = cfg.roi.margin
    x0 -= bw * m
    y0 -= bh * m
    x1 += bw * m
    y1 += bh * m

    x0 = int(np.clip(x0, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))
    x1 = int(np.clip(x1, x0 + 1, W))
    y1 = int(np.clip(y1, y0 + 1, H))

    # enforce min size
    min_s = cfg.roi.min_size
    bw, bh = x1 - x0, y1 - y0
    if bw < min_s or bh < min_s:
        s = max(min_s, bw, bh)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        x0 = int(np.clip(cx - s / 2, 0, W - 1))
        y0 = int(np.clip(cy - s / 2, 0, H - 1))
        x1 = int(np.clip(x0 + s, x0 + 1, W))
        y1 = int(np.clip(y0 + s, y0 + 1, H))

    return x0, y0, x1, y1


def _select_keypoints(cfg: TrainConfig, lm) -> Tuple[np.ndarray, np.ndarray]:
    """Return (K,2) in ROI pixel coords later, and visibility (K,)."""
    if cfg.region == "hands":
        # Prefer whichever hand is more visible; else concatenate both.
        lh = lm.left_hand
        rh = lm.right_hand
        lh_vis = lm.left_hand_vis
        rh_vis = lm.right_hand_vis

        if lh is None and rh is None:
            return np.zeros((21, 2), np.float32), np.zeros((21,), np.float32)

        if lh is not None and rh is None:
            return lh[:, :2].copy(), (lh_vis if lh_vis is not None else np.ones((21,), np.float32))
        if rh is not None and lh is None:
            return rh[:, :2].copy(), (rh_vis if rh_vis is not None else np.ones((21,), np.float32))

        # both present: choose one with higher mean vis (keeps K fixed)
        lv = float(np.mean(lh_vis)) if lh_vis is not None else 1.0
        rv = float(np.mean(rh_vis)) if rh_vis is not None else 1.0
        if rv >= lv:
            return rh[:, :2].copy(), (rh_vis if rh_vis is not None else np.ones((21,), np.float32))
        return lh[:, :2].copy(), (lh_vis if lh_vis is not None else np.ones((21,), np.float32))

    if cfg.region == "face":
        if lm.face is None:
            return np.zeros((478, 2), np.float32), np.zeros((478,), np.float32)
        vis = lm.face_vis if lm.face_vis is not None else np.ones((lm.face.shape[0],), np.float32)
        return lm.face[:, :2].copy(), vis

    # pose-based
    if lm.pose is None:
        return np.zeros((33, 2), np.float32), np.zeros((33,), np.float32)
    vis = lm.pose_vis if lm.pose_vis is not None else np.ones((33,), np.float32)
    return lm.pose[:, :2].copy(), vis


def precompute_cache(cfg: TrainConfig, out_hw: int = 256, stride: int = 12) -> Path:
    """Create ROI crops + keypoint heatmaps cache for critic training.

    Outputs:
      cache_dir/region/<video_stem>__s<start>.npz
      cache_dir/manifest.json
    """
    cache_root = cfg.data.cache_dir / "region" / cfg.region
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []

    videos = list_videos(cfg.data.data_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in {cfg.data.data_dir}")

    for vid in tqdm(videos, desc="Precompute cache"):
        frames, _fps = read_video_rgb(vid)
        # downsample spatially early for speed
        if cfg.data.resolution and max(frames.shape[1], frames.shape[2]) != cfg.data.resolution:
            scale = cfg.data.resolution / max(frames.shape[1], frames.shape[2])
            nh = int(round(frames.shape[1] * scale))
            nw = int(round(frames.shape[2] * scale))
            frames = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames], axis=0)

        # mediapipe expects RGB
        lms = extract_landmarks_holistic(frames)

        for start, clip in iter_clips(frames, clip_len=cfg.data.clip_len, stride=stride):
            # compute a single bbox for entire clip (more stable)
            H, W = clip.shape[1], clip.shape[2]
            bbs = []
            kps_list = []
            vis_list = []
            for t in range(cfg.data.clip_len):
                lm = lms[start + t]
                bb = _roi_from_landmarks(cfg, lm, H=H, W=W)
                bbs.append(bb)
                kps, vis = _select_keypoints(cfg, lm)
                kps_list.append(kps)
                vis_list.append(vis)

            # union bbox
            x0 = min(bb[0] for bb in bbs)
            y0 = min(bb[1] for bb in bbs)
            x1 = max(bb[2] for bb in bbs)
            y1 = max(bb[3] for bb in bbs)

            roi = clip[:, y0:y1, x0:x1, :]
            roi_resized = np.stack(
                [cv2.resize(r, (out_hw, out_hw), interpolation=cv2.INTER_AREA) for r in roi],
                axis=0,
            )

            # keypoints in ROI pixel coords
            hms = []
            for kps_norm, vis in zip(kps_list, vis_list):
                # norm -> pixel in original
                kps_px = np.stack([kps_norm[:, 0] * W, kps_norm[:, 1] * H], axis=-1)
                # shift into ROI
                kps_roi = kps_px - np.array([x0, y0], dtype=np.float32)[None, :]
                # scale into resized
                sx = out_hw / max(1.0, (x1 - x0))
                sy = out_hw / max(1.0, (y1 - y0))
                kps_rs = kps_roi * np.array([sx, sy], dtype=np.float32)[None, :]
                hms.append(keypoints_to_heatmaps(kps_rs, vis, (out_hw, out_hw), sigma=2.0))
            hms = np.stack(hms, axis=0)  # (T,K,H,W)

            # confidence filter: if avg visibility too low, drop
            avg_vis = float(np.mean([np.mean(v) for v in vis_list]))
            if avg_vis < cfg.temporal.track_confidence_threshold:
                continue

            out_path = cache_root / f"{vid.stem}__s{start:06d}.npz"
            np.savez_compressed(
                out_path,
                roi=roi_resized,
                heatmaps=hms,
                bbox=np.array([x0, y0, x1, y1], dtype=np.int32),
                src=str(vid),
                start=int(start),
            )

            manifest.append({"npz": str(out_path), "src": str(vid), "start": int(start)})

    man_path = cfg.data.cache_dir / "manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return man_path
