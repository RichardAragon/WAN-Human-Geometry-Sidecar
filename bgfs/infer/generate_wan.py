from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.utils import export_to_video


def load_wan_i2v(model_id: str, dtype: torch.dtype = torch.bfloat16):
    """Load Wan 2.x I2V pipeline.

    Note: Wan VAE is recommended in float32 for quality.
    """
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    # upcast VAE
    pipe.vae.to(torch.float32)
    return pipe


def generate_video(
    model_id: str,
    prompt: str,
    negative_prompt: str,
    out_mp4: Path,
    image_path: Optional[Path] = None,
    num_frames: int = 81,
    guidance_scale: float = 5.0,
    fps: int = 16,
    seed: int = 0,
    device: str = "cuda",
):
    pipe = load_wan_i2v(model_id)
    pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    # image is optional for I2V; if omitted, pipeline behaves like T2V conditioned only on prompt (implementation dependent)
    image = None
    if image_path is not None:
        from diffusers.utils import load_image

        image = load_image(str(image_path))

    out = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    frames = out.frames[0]
    export_to_video(frames, str(out_mp4), fps=fps)
    return out_mp4
