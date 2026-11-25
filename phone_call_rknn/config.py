from pathlib import Path


POSE_MODEL = "/mnt/jl/yolov26-rk3588/model/yolo26n-pose-rk3588.rknn"
# DETECT_MODEL = "/mnt/jl/yolov26-rk3588/model-pose-detect/0610-hand-phone.rknn"
DETECT_MODEL = "/mnt/jl/yolov26-rk3588/model/0704-PHS.rknn"
DEFAULT_SOURCE = "/mnt/jl/yolov26-rk3588/video/VID_20260603_102426_720p_h264.mp4"
DEFAULT_OUTPUT_DIR = "/mnt/jl/yolov26-rk3588/runs/0709/HPS-VID_102426-1"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10

DEFAULT_PERSON_SCALE = 1000.0
MIN_PERSON_SCALE = 30.0
DETECT_NAMES = {0: "hand", 1: "phone", 2: "seatbelt"}

PHONE_HAND_PIXEL_THRES = 150.0
PHONE_HAND_IOU_THRES = 0.0
WRIST_PHONE_PIXEL_THRES = 150.0


def resolve_rknn_path(model_path):
    path = Path(model_path)
    if path.is_file():
        return path
    matches = sorted(path.rglob("*.rknn"))
    if not matches:
        raise FileNotFoundError(f"No .rknn file found under: {path}")
    return matches[0]

