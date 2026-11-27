"""司机行为模式（driver）的配置：模型、阈值、关键点、引擎参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..common.config import env_bool, env_float, env_int, env_str, model_path

# --------------------------------------------------------------------------
# 模型（环境变量 > 仓库 model/ > 外部默认路径）
# --------------------------------------------------------------------------
POSE_MODEL = model_path("POSE_MODEL", "yolo26n-pose-rk3588.rknn",
                        "/mnt/jl/yolov26-rk3588/model/yolo26n-pose-rk3588.rknn")
DETECT_MODEL = model_path("DETECT_MODEL", "0704-PHS.rknn",
                          "/mnt/jl/yolov26-rk3588/model/0704-PHS.rknn")

# --------------------------------------------------------------------------
# 输入输出
# --------------------------------------------------------------------------
DEFAULT_SOURCE = env_str("DEFAULT_SOURCE", "/mnt/jl/yolov26-rk3588/video/VID_20260603_102426_720p_h264.mp4")
DEFAULT_OUTPUT_DIR = Path(env_str("DEFAULT_OUTPUT_DIR", "/mnt/jl/yolov26-rk3588/runs/0709/HPS-VID_102426-1"))

# --------------------------------------------------------------------------
# 姿态关键点（17 点模型的下标）
# --------------------------------------------------------------------------
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10

DEFAULT_PERSON_SCALE = 1000.0
MIN_PERSON_SCALE = 30.0

# --------------------------------------------------------------------------
# 检测类别与判定阈值
# --------------------------------------------------------------------------
DETECT_NAMES = {0: "hand", 1: "phone", 2: "seatbelt"}

PHONE_HAND_PIXEL_THRES = env_float("PHONE_HAND_PIXEL_THRES", 150.0)
PHONE_HAND_IOU_THRES = env_float("PHONE_HAND_IOU_THRES", 0.0)
WRIST_PHONE_PIXEL_THRES = env_float("WRIST_PHONE_PIXEL_THRES", 150.0)

MODEL_INPUT_SIZE = env_int("MODEL_IMGSZ", 640)


@dataclass
class EngineConfig:
    """加速推理链路参数；对应 CLI/env 见 driver/cli.py。"""

    pose_model: Path = POSE_MODEL
    detect_model: Path = DETECT_MODEL
    imgsz: int = MODEL_INPUT_SIZE

    det_conf: float = env_float("DET_CONF", 0.35)
    person_conf: float = env_float("PERSON_CONF", 0.25)
    kpt_conf: float = env_float("KPT_CONF", 0.30)
    iou: float = env_float("NMS_IOU", 0.45)
    phone_hand_pixel_thres: float = PHONE_HAND_PIXEL_THRES
    phone_hand_iou_thres: float = PHONE_HAND_IOU_THRES
    wrist_phone_pixel_thres: float = WRIST_PHONE_PIXEL_THRES

    # 推理后端
    detect_cores: str = env_str("DETECT_CORES", "0,1")
    pose_cores: str = env_str("POSE_CORES", "2")
    npu_warmup: int = env_int("NPU_WARMUP", 5)

    # 解码 / 编码
    decode: str = env_str("DECODE", "vpu")            # vpu（硬解+dma-buf）| cpu（纯软解）
    decode_fallback: bool = env_bool("DECODE_FALLBACK", True)
    codec: str = env_str("CODEC", "h264")
    encoder: str = env_str("ENCODER", "vpu")          # vpu | mp4v
    bitrate: Optional[int] = env_int("BITRATE", 0) or None
    decode_queue: int = env_int("DECODE_QUEUE", 4)
    encode_queue: int = env_int("ENCODE_QUEUE", 2)
    frame_budget: int = env_int("FRAME_BUDGET", 6)

    # 输出
    frame_stride: int = env_int("FRAME_STRIDE", 1)
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
