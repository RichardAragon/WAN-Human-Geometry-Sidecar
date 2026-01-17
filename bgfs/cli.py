from __future__ import annotations

from pathlib import Path

import typer

from bgfs.config import TrainConfig
from bgfs.utils.io import load_model_config
from bgfs.train.precompute import precompute_roi_cache
from bgfs.train.train_critic import train_critic
from bgfs.infer.generate_wan import generate_video
from bgfs.infer.fix_roi import fix_video_roi

app = typer.Typer(add_completion=False, help="BGFS - Body-part Geometry Fixer Sidecar for Wan2.2")


@app.command()
def precompute(
    config: Path = typer.Option(..., "--config", help="YAML config for dataset/caching"),
):
    """Extract ROI crops + pseudo-label heatmaps (MediaPipe) into a cache."""
    cfg = load_model_config(config, TrainConfig)
    man = precompute_roi_cache(cfg)
    typer.echo(f"Wrote manifest: {man}")


@app.command()
def train_critic_cmd(
    config: Path = typer.Option(..., "--config"),
    out_dir: Path = typer.Option(Path("outputs/critic"), "--out-dir"),
    epochs: int = typer.Option(5, "--epochs"),
):
    """Train a differentiable heatmap critic to replace MediaPipe at train/inference time."""
    cfg = load_model_config(config, TrainConfig)
    manifest = cfg.data.cache_dir / "manifest.json"
    ckpt = train_critic(cfg, manifest, out_dir, epochs=epochs)
    typer.echo(f"Saved critic: {ckpt}")


@app.command()
def generate(
    model_id: str = typer.Option("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "--model-id"),
    prompt: str = typer.Option(..., "--prompt"),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
    out: Path = typer.Option(Path("out.mp4"), "--out"),
    image: Path = typer.Option(None, "--image"),
    num_frames: int = typer.Option(81, "--num-frames"),
    guidance_scale: float = typer.Option(5.0, "--guidance-scale"),
    seed: int = typer.Option(0, "--seed"),
    fps: int = typer.Option(16, "--fps"),
):
    """Generate a video with Wan."""
    generate_video(
        model_id=model_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        out_mp4=out,
        image_path=image,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        seed=seed,
        fps=fps,
    )
    typer.echo(f"Wrote: {out}")


@app.command()
def fix(
    in_mp4: Path = typer.Option(..., "--in"),
    out_mp4: Path = typer.Option(Path("fixed.mp4"), "--out"),
    critic_ckpt: Path = typer.Option(..., "--critic"),
    model_id: str = typer.Option("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "--model-id"),
    region: str = typer.Option("hands", "--region"),
    steps: int = typer.Option(25, "--steps"),
    lr: float = typer.Option(0.07, "--lr"),
    strength: float = typer.Option(1.0, "--strength"),
):
    """Post-hoc latent optimization to improve ROI geometry using the critic and simple kinematic losses."""
    fix_video_roi(
        in_mp4=in_mp4,
        out_mp4=out_mp4,
        critic_ckpt=critic_ckpt,
        model_id=model_id,
        region=region,
        steps=steps,
        lr=lr,
        strength=strength,
    )
    typer.echo(f"Wrote: {out_mp4}")


if __name__ == "__main__":
    app()
