"""模型输出解码（与业务无关）：几何工具、检测/姿态后处理、CPU 预处理。

这里只处理 numpy 数据，不 import hw/（硬件细节在 `hw/`，编排在 `pipeline/`）。
"""

from .geometry import (
    box_area,
    box_center,
    box_iou,
    box_pair_iou,
    intersection_area,
    is_finite_array,
    nms_class_aware,
    point_dist,
    point_in_box,
    scale_boxes_to_original,
    scale_keypoints_to_original,
    sigmoid,
)
from .postprocess import decode_detect_outputs, decode_detect_quantized, decode_pose_half, decode_pose_outputs

__all__ = [
    "box_area",
    "box_center",
    "box_iou",
    "box_pair_iou",
    "intersection_area",
    "is_finite_array",
    "nms_class_aware",
    "point_dist",
    "point_in_box",
    "scale_boxes_to_original",
    "scale_keypoints_to_original",
    "sigmoid",
    "decode_detect_outputs",
    "decode_detect_quantized",
    "decode_pose_half",
    "decode_pose_outputs",
]
