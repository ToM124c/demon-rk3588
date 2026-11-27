"""环境变量与路径解析：所有模式共用的“地基”。"""

from __future__ import annotations

import os
from pathlib import Path


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = PROJECT_ROOT / "model"


def model_path(env_name: str, filename: str, fallback: str) -> Path:
    """模型路径解析：环境变量 > 仓库内 model/ > 外部默认路径。"""
    explicit = os.environ.get(env_name)
    if explicit:
        return Path(explicit)
    local = LOCAL_MODEL_DIR / filename
    if local.is_file():
        return local
    return Path(fallback)


def resolve_rknn_path(model_path: Path) -> Path:
    """允许传入目录：取目录下第一个 .rknn。"""
    path = Path(model_path)
    if path.is_file():
        return path
    matches = sorted(path.rglob("*.rknn"))
    if not matches:
        raise FileNotFoundError(f"No .rknn file found under: {path}")
    return matches[0]
