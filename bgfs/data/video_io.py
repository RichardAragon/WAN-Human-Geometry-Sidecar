from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def list_videos(data_dir: Path) -> List[Path]:
    vids: List[Path] = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    return sorted(vids)


@dataclass
class VideoClip:
    frames: np.ndarray  # (T,H,W,3) uint8
    fps: float
    path: Path
    start: int


def read_video_rgb(path: Path, max_frames: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Read full video into memory.

    Returns:
        frames: uint8 (T,H,W,3)
        fps: float (best effort; may be 0 if unknown)
    """
    meta = iio.immeta(str(path), plugin="ffmpeg")
    fps = float(meta.get("fps", 0.0) or 0.0)
    frames = iio.imread(str(path), plugin="ffmpeg")  # (T,H,W,3)
    if frames.ndim == 3:
        # sometimes comes as (T,H,W) grayscale
        frames = np.stack([frames] * 3, axis=-1)
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames.astype(np.uint8), fps


def iter_clips(
    frames: np.ndarray,
    clip_len: int,
    stride: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (start_idx, clip_frames)."""
    T = int(frames.shape[0])
    if T < clip_len:
        return
    for s in range(0, T - clip_len + 1, stride):
        yield s, frames[s : s + clip_len]
