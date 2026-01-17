from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RegionName = Literal[
    "hands",
    "face",
    "feet",
    "upper_body",
    "full_body",
    "custom_mask",
]


class ROIConfig(BaseModel):
    """How we locate/crop the target region across frames."""

    strategy: Literal["pose_track", "detector_track", "user_mask"] = "pose_track"
    margin: float = Field(0.25, description="Extra margin around ROI as fraction of bbox size")
    min_size: int = Field(192, description="Minimum crop size (square) before resize")


class ConstraintConfig(BaseModel):
    constraint_strength: float = 1.0
    joint_limit_mode: Literal["soft", "hard"] = "soft"
    topology_guard: bool = True


class TemporalConfig(BaseModel):
    temporal_smoothness: float = 1.0
    track_confidence_threshold: float = 0.6
    occlusion_handling: Literal["freeze", "predict_through", "ignore"] = "freeze"


class TeacherGuidanceConfig(BaseModel):
    enabled: bool = True
    guidance_steps_frac: float = 0.6  # last 60% steps
    guidance_lr: float = 0.08
    guidance_schedule: Literal["late_heavy", "flat"] = "late_heavy"


class AdapterConfig(BaseModel):
    adapter_type: Literal["lora", "ia3", "small_head"] = "lora"
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: Literal["attn", "attn+ffn"] = "attn"
    dropout: float = 0.0


class DataConfig(BaseModel):
    data_dir: Path
    cache_dir: Path = Path("./cache")
    resolution: int = 480
    clip_len: int = 49
    fps: int = 24
    val_split: float = 0.05


class ComputeConfig(BaseModel):
    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16"
    gradient_checkpointing: bool = True
    microbatch: int = 1
    grad_accum: int = 4
    num_workers: int = 4


class TrainConfig(BaseModel):
    project: str = "bgfs"
    seed: int = 123
    model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    region: RegionName = "hands"

    roi: ROIConfig = ROIConfig()
    constraints: ConstraintConfig = ConstraintConfig()
    temporal: TemporalConfig = TemporalConfig()
    teacher: TeacherGuidanceConfig = TeacherGuidanceConfig()
    adapter: AdapterConfig = AdapterConfig()
    data: DataConfig
    compute: ComputeConfig = ComputeConfig()

    # Stage B
    lr: float = 1e-4
    train_steps: int = 2000
    save_every: int = 250

    # Optional prompt conditioning for teacher samples
    prompts: List[str] = Field(default_factory=lambda: [
        "a person gesturing with their hands",
        "a person waving hello",
        "a person typing on a keyboard",
        "a person holding an object",
    ])

    extra: Dict[str, Any] = Field(default_factory=dict)
