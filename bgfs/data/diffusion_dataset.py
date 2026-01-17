from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from bgfs.data.video_io import list_videos, read_video_rgb, iter_clips


class OnTheFlyVideoDataset(Dataset):
    """Loads videos and samples short clips.

    Returns:
      cond: (3,H,W) float in [0,1] (first frame)
      clip: (T,3,H,W) float in [0,1]
      prompt: str
    """

    def __init__(
        self,
        data_dir: Path,
        resolution: int,
        clip_len: int,
        stride: int,
        prompts: List[str],
        seed: int = 123,
        max_videos: int | None = None,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.prompts = prompts
        vids = list_videos(data_dir)
        if max_videos is not None:
            vids = vids[:max_videos]
        if not vids:
            raise FileNotFoundError(f"No videos found in {data_dir}")
        self.videos = vids
        self.resolution = resolution
        self.clip_len = clip_len
        self.stride = stride

        # build an index of (video_idx, start)
        self.index: List[Tuple[int, int]] = []
        for vi, vp in enumerate(self.videos):
            frames, _ = read_video_rgb(vp, max_frames=4000)
            # use simple stride clips
            for start, _clip in iter_clips(frames, clip_len=clip_len, stride=stride):
                self.index.append((vi, int(start)))

        if not self.index:
            raise RuntimeError("No clips could be sampled (videos too short?)")
        self.rng.shuffle(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        vi, start = self.index[i]
        vp = self.videos[vi]
        frames, _ = read_video_rgb(vp)
        clip = frames[start : start + self.clip_len]

        # resize keeping aspect ratio by max side
        import cv2

        H, W = clip.shape[1], clip.shape[2]
        scale = self.resolution / max(H, W)
        nh = int(round(H * scale))
        nw = int(round(W * scale))
        clip = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in clip], axis=0)

        cond = clip[0]

        cond_t = torch.from_numpy(cond.astype(np.float32) / 255.0).permute(2, 0, 1)
        clip_t = torch.from_numpy(clip.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

        prompt = self.rng.choice(self.prompts) if self.prompts else "a person"
        return cond_t, clip_t, prompt
