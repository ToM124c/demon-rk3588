"""护送 / 交警手势合规检测：配置与常量。

模型来自 /mnt/jl/yolov26-rk3588/model/：
- 检测：0710-escort-hand.rknn（单类 hand，输入 INT8，3 个 stride 的 4+1 输出）
- 姿态：0710-escort-pose-rk3588.rknn（输入 FP16，输出 (56, 8400) FP16）
两者都可用 ``ESCORT_*`` 环境变量覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import model_path


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
# 模型与输出
# --------------------------------------------------------------------------
ESCORT_POSE_MODEL = model_path("ESCORT_POSE_MODEL", "0710-escort-pose-rk3588.rknn",
                               "/mnt/jl/yolov26-rk3588/model/0710-escort-pose-rk3588.rknn")
ESCORT_HAND_MODEL = model_path("ESCORT_DETECT_MODEL", "0710-escort-hand.rknn",
                               "/mnt/jl/yolov26-rk3588/model/0710-escort-hand.rknn")

DEFAULT_SOURCE = _env("ESCORT_SOURCE", "/mnt/jl/yolov26-rk3588/video/test-2/test-escort-3.mp4")
DEFAULT_OUTPUT_DIR = Path(_env("ESCORT_OUTPUT_DIR", "/mnt/jl/yolov26-rk3588/runs/escort"))

LABEL_COMPLIANT = "compliant"
LABEL_NON_COMPLIANT = "non_compliant"

# --------------------------------------------------------------------------
# 关键点 / 角色 / 颜色
# --------------------------------------------------------------------------
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KEYPOINT_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

ROLE_ORDER = ("right_police", "suspect", "left_police")
ROLE_COLORS = {
    "suspect": (0, 0, 255),
    "right_police": (255, 0, 0),
    "left_police": (255, 120, 0),
}
SIDE_COLORS = {
    "left": (0, 255, 0),
    "right": (255, 0, 255),
}
HAND_SIDE_LABELS = {"left": "左手", "right": "右手"}
TARGET_LABELS = {
    "left_elbow": "左肘",
    "right_elbow": "右肘",
}


# --------------------------------------------------------------------------
# 运行配置
# --------------------------------------------------------------------------
@dataclass
class EscortConfig:
    pose_model: Path = ESCORT_POSE_MODEL
    hand_model: Path = ESCORT_HAND_MODEL
    imgsz: int = 640

    det_conf: float = _env_float("ESCORT_DET_CONF", 0.30)
    person_conf: float = _env_float("ESCORT_PERSON_CONF", 0.25)
    kpt_conf: float = _env_float("ESCORT_KPT_CONF", 0.30)
    iou: float = _env_float("ESCORT_IOU", 0.45)
    hand_max_pixel_dist: float = _env_float("ESCORT_HAND_MAX_PIXEL_DIST", 180.0)
    contact_pixel_thres: float = _env_float("ESCORT_CONTACT_PIXEL_THRES", 95.0)

    # 一个 worker 同时跑 hand + pose，绑在同一个 NPU 核上；多个核用逗号分隔
    cores: str = _env("ESCORT_CORES", "2")
    zero_copy: bool = _env_bool("ZERO_COPY", True)
    decode: str = _env("ESCORT_DECODE", "vpu")
    codec: str = _env("ESCORT_CODEC", "h264")
    frame_stride: int = _env_int("ESCORT_FRAME_STRIDE", 1)
    decode_queue: int = _env_int("ESCORT_DECODE_QUEUE", 4)
    frame_budget: int = _env_int("ESCORT_FRAME_BUDGET", 6)
    encode_queue: int = _env_int("ESCORT_ENCODE_QUEUE", 2)
    npu_warmup: int = _env_int("ESCORT_NPU_WARMUP", 3)

    no_save_vis: bool = True
    quiet: bool = True


def default_escort_config(**overrides) -> EscortConfig:
    cfg = EscortConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"unknown EscortConfig field: {key}")
        setattr(cfg, key, value)
    return cfg
