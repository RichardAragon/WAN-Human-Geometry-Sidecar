# bgfs/cli.py
from __future__ import annotations

from pathlib import Path

import typer

from bgfs.config import TrainConfig
from bgfs.utils.io import load_model_config
from bgfs.compat import import_module, resolve_callable, call_with_fallback, CompatError

app = typer.Typer(add_completion=False, help="BGFS - Body-part Geometry Fixer Sidecar for Wan2.2")


def _load_cfg(config: Path) -> TrainConfig:
    return load_model_config(config, TrainConfig)


@app.command("precompute")
def precompute(
    config: Path = typer.Option(..., "--config", help="YAML config for dataset/caching"),
):
    """Extract ROI crops + pseudo-label heatmaps (MediaPipe) into a cache."""
    cfg = _load_cfg(config)

    mod = import_module("bgfs.train.precompute")
    fn = resolve_callable(
        mod,
        candidates=[
            "precompute_roi_cache",
            "precompute_cache",
            "precompute_roi",
            "build_roi_cache",
            "precompute",
            "main",
        ],
    )

    man = call_with_fallback(fn, cfg)
    typer.echo(f"Wrote manifest: {man}")


@app.command("train-critic")
def train_critic_cmd(
    config: Path = typer.Option(..., "--config"),
    out_dir: Path = typer.Option(Path("outputs/critic"), "--out-dir"),
    epochs: int = typer.Option(5, "--epochs"),
):
    """Train a differentiable heatmap critic to replace MediaPipe at train/inference time."""
    cfg = _load_cfg(config)
    manifest = cfg.data.cache_dir / "manifest.json"

    mod = import_module("bgfs.train.train_critic")
    fn = resolve_callable(
        mod,
        candidates=[
            "train_critic",
            "train",
            "fit",
            "main",
        ],
    )

    ckpt = call_with_fallback(fn, cfg, manifest, out_dir, epochs=epochs)
    typer.echo(f"Saved critic: {ckpt}")


@app.command("generate")
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
    mod = import_module("bgfs.infer.generate_wan")
    fn = resolve_callable(
        mod,
        candidates=[
            "generate_video",
            "generate",
            "run",
            "main",
        ],
    )

    # Prefer kwargs so we survive signature changes.
    call_with_fallback(
        fn,
        model_id=model_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        out_mp4=out,
        out=out,  # alternate name
        image_path=image,
        image=image,  # alternate name
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        seed=seed,
        fps=fps,
    )
    typer.echo(f"Wrote: {out}")


@app.command("fix")
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
    mod = import_module("bgfs.infer.fix_roi")
    fn = resolve_callable(
        mod,
        candidates=[
            "fix_video_roi",
            "fix_roi",
            "fix",
            "run",
            "main",
        ],
    )

    try:
        call_with_fallback(
            fn,
            in_mp4=in_mp4,
            out_mp4=out_mp4,
            critic_ckpt=critic_ckpt,
            critic=critic_ckpt,  # alternate name
            model_id=model_id,
            region=region,
            steps=steps,
            lr=lr,
            strength=strength,
        )
    except CompatError as e:
        raise typer.BadParameter(str(e))

    typer.echo(f"Wrote: {out_mp4}")

