# bgfs/utils/io.py
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_config(path: Path, cls: Type[T]) -> T:
    return cls.model_validate(load_yaml(path))


def dump_config(cfg: BaseModel) -> Dict[str, Any]:
    return cfg.model_dump(mode="python")


def load_model_config(path: Path, cls: Type[T]) -> T:
    """
    Backwards-compat wrapper used by CLI.
    Historically the CLI imported `load_model_config`.
    """
    return load_config(path, cls)
