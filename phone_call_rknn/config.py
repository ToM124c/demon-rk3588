"""全局配置：模型、阈值、运行参数，全部可用环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env(name: str, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
POSE_MODEL = Path(_env("POSE_MODEL", "/mnt/jl/yolov26-rk3588/model/yolo26n-pose-rk3588.rknn"))
DETECT_MODEL = Path(_env("DETECT_MODEL", "/mnt/jl/yolov26-rk3588/model/0704-PHS.rknn"))

# --------------------------------------------------------------------------
# 输入输出
# --------------------------------------------------------------------------
DEFAULT_SOURCE = _env("DEFAULT_SOURCE", "/mnt/jl/yolov26-rk3588/video/VID_20260603_102426_720p_h264.mp4")
DEFAULT_OUTPUT_DIR = Path(_env("DEFAULT_OUTPUT_DIR", "/mnt/jl/yolov26-rk3588/runs/0709/HPS-VID_102426-1"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

# --------------------------------------------------------------------------
# 姿态关键点
# --------------------------------------------------------------------------
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10

DEFAULT_PERSON_SCALE = 1000.0
MIN_PERSON_SCALE = 30.0

DETECT_NAMES = {0: "hand", 1: "phone", 2: "seatbelt"}

# --------------------------------------------------------------------------
# 判定阈值
# --------------------------------------------------------------------------
PHONE_HAND_PIXEL_THRES = _env_float("PHONE_HAND_PIXEL_THRES", 150.0)
PHONE_HAND_IOU_THRES = _env_float("PHONE_HAND_IOU_THRES", 0.0)
WRIST_PHONE_PIXEL_THRES = _env_float("WRIST_PHONE_PIXEL_THRES", 150.0)

MODEL_INPUT_SIZE = _env_int("MODEL_IMGSZ", 640)


def resolve_rknn_path(model_path) -> Path:
    path = Path(model_path)
    if path.is_file():
        return path
    matches = sorted(path.rglob("*.rknn"))
    if not matches:
        raise FileNotFoundError(f"No .rknn file found under: {path}")
    return matches[0]


# --------------------------------------------------------------------------
# 引擎配置
# --------------------------------------------------------------------------
@dataclass
class EngineConfig:
    """加速推理链路参数；对应 CLI/env 见 app/cli.py。"""

    pose_model: Path = POSE_MODEL
    detect_model: Path = DETECT_MODEL
    imgsz: int = MODEL_INPUT_SIZE

    det_conf: float = _env_float("DET_CONF", 0.35)
    person_conf: float = _env_float("PERSON_CONF", 0.25)
    kpt_conf: float = _env_float("KPT_CONF", 0.30)
    iou: float = _env_float("NMS_IOU", 0.45)
    phone_hand_pixel_thres: float = PHONE_HAND_PIXEL_THRES
    phone_hand_iou_thres: float = PHONE_HAND_IOU_THRES
    wrist_phone_pixel_thres: float = WRIST_PHONE_PIXEL_THRES

    # 推理后端
    detect_cores: str = _env("DETECT_CORES", "0,1")
    pose_cores: str = _env("POSE_CORES", "2")
    zero_copy: bool = _env_bool("ZERO_COPY", True)
    npu_warmup: int = _env_int("NPU_WARMUP", 5)

    # 解码 / 编码
    decode: str = _env("DECODE", "vpu")            # vpu（GStreamer 硬解）| cpu（OpenCV）
    codec: str = _env("CODEC", "h264")
    encoder: str = _env("ENCODER", "vpu")          # vpu | mp4v
    bitrate: Optional[int] = _env_int("BITRATE", 0) or None
    decode_queue: int = _env_int("DECODE_QUEUE", 4)
    encode_queue: int = _env_int("ENCODE_QUEUE", 2)
    frame_budget: int = _env_int("FRAME_BUDGET", 6)

    # 输出
    frame_stride: int = _env_int("FRAME_STRIDE", 1)
    no_save_vis: bool = True
    display: bool = False
    quiet: bool = True


def default_engine_config(**overrides) -> EngineConfig:
    cfg = EngineConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"unknown EngineConfig field: {key}")
        setattr(cfg, key, value)
    return cfg
