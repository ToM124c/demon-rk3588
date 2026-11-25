import argparse

from .config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    DETECT_MODEL,
    PHONE_HAND_IOU_THRES,
    PHONE_HAND_PIXEL_THRES,
    POSE_MODEL,
    WRIST_PHONE_PIXEL_THRES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RKNNLite driver phone-call detection with pose and hand-phone models."
    )
    parser.add_argument("--pose-model", default=POSE_MODEL, help="关键点检测 RKNN 模型路径或目录")
    parser.add_argument("--detect-model", default=DETECT_MODEL, help="hand/phone/seatbelt RKNN 模型路径或目录")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="输入图片、图片目录或视频路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--imgsz", type=int, default=640, help="模型输入尺寸")
    parser.add_argument("--det-conf", type=float, default=0.35, help="hand/phone/seatbelt 检测置信度阈值")
    parser.add_argument("--person-conf", type=float, default=0.25, help="人体框置信度阈值")
    parser.add_argument("--kpt-conf", type=float, default=0.30, help="手腕关键点置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--phone-hand-pixel-thres", type=float, default=PHONE_HAND_PIXEL_THRES, help="phone 和 hand 中心点像素距离阈值")
    parser.add_argument("--phone-hand-iou-thres", type=float, default=PHONE_HAND_IOU_THRES, help="phone 和 hand 检测框交并比阈值")
    parser.add_argument("--wrist-phone-pixel-thres", type=float, default=WRIST_PHONE_PIXEL_THRES, help="phone 和 wrist 关键点像素距离阈值")
    parser.add_argument("--pose-input-layout", choices=("nhwc", "nchw"), default="nhwc", help="pose 输入排布")
    parser.add_argument("--detect-input-layout", choices=("nhwc", "nchw"), default="nhwc", help="detect 输入排布")
    parser.add_argument("--no-save-vis", action="store_true", help="不保存可视化图片/视频")
    parser.add_argument("--quiet", action="store_true", help="不在标准输出逐帧打印 JSON")
    return parser.parse_args()

