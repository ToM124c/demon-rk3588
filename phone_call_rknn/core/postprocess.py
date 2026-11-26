"""模型输出解码：浮点（RKNNLite 兜底）与量化（want_float=0）两条路径。

量化路径的原则：**先在 int8/fp16 域里用算子最少的比较做筛选，只对候选行反量化**，
避免把整个输出张量（pose 是 470k 元素）先转成 float32。
"""

from __future__ import annotations

import numpy as np

from ..config import DETECT_NAMES
from .geometry import (
    box_center,
    maybe_sigmoid,
    nms_class_aware,
    scale_boxes_to_original,
    scale_keypoints_to_original,
    sigmoid,
    xywh_to_xyxy,
)


# --------------------------------------------------------------------------
# 通用小工具
# --------------------------------------------------------------------------
def as_prediction_matrix(outputs):
    if outputs is None or len(outputs) == 0:
        return np.empty((0, 0), dtype=np.float32)
    arr = np.asarray(outputs[0], dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)
    if arr.shape[0] <= arr.shape[1]:
        arr = arr.T
    return arr


def make_detection_dicts(boxes, scores, classes, keep):
    dets = []
    for idx in keep:
        box = boxes[idx].astype(np.float32)
        cls_id = int(classes[idx])
        dets.append(
            {
                "box": box,
                "center": box_center(box),
                "conf": float(scores[idx]),
                "class": DETECT_NAMES.get(cls_id, str(cls_id)),
            }
        )
    return dets


def _filter_boxes(boxes, scores, classes, iou_thres):
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    keep = nms_class_aware(boxes, scores, classes, iou_thres)
    return make_detection_dicts(boxes, scores, classes, keep)


# --------------------------------------------------------------------------
# 浮点路径（RKNNLite / 兜底）
# --------------------------------------------------------------------------
def decode_detect_decoded(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres):
    pred = as_prediction_matrix(outputs)
    if pred.size == 0 or pred.shape[1] < 6:
        return None

    boxes_raw = pred[:, :4]
    class_scores = maybe_sigmoid(pred[:, 4:])
    classes = np.argmax(class_scores, axis=1).astype(np.int32)
    scores = class_scores[np.arange(len(pred)), classes]
    mask = np.all(np.isfinite(boxes_raw), axis=1) & np.isfinite(scores) & (scores >= conf_thres)
    if not np.any(mask):
        return []

    boxes_raw = boxes_raw[mask]
    classes = classes[mask]
    scores = scores[mask]
    x1, y1, x2, y2 = boxes_raw.T
    xyxy_ratio = np.mean((x2 > x1) & (y2 > y1))
    boxes = boxes_raw.copy() if xyxy_ratio > 0.85 else xywh_to_xyxy(boxes_raw)
    boxes = scale_boxes_to_original(boxes, scale, pad_x, pad_y, image_shape)
    valid = (classes >= 0) & (classes < len(DETECT_NAMES))
    boxes, classes, scores = boxes[valid], classes[valid], scores[valid]
    return _filter_boxes(boxes, scores, classes, iou_thres)


def decode_detect_split_outputs(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres):
    if outputs is None or len(outputs) < 6:
        return None

    strides = [8, 16, 32]
    boxes_all, scores_all, classes_all = [], [], []

    for i, stride in enumerate(strides):
        reg = np.asarray(outputs[i * 2], dtype=np.float32)
        cls = np.asarray(outputs[i * 2 + 1], dtype=np.float32)
        reg = np.squeeze(reg)
        cls = np.squeeze(cls)
        if reg.ndim != 3 or cls.ndim != 3 or reg.shape[0] != 4 or cls.shape[0] < len(DETECT_NAMES):
            return None

        _, height, width = reg.shape
        probs = maybe_sigmoid(cls[: len(DETECT_NAMES)].reshape(len(DETECT_NAMES), -1).T)
        classes = np.argmax(probs, axis=1).astype(np.int32)
        scores = probs[np.arange(len(probs)), classes]
        boxes = _decode_reg(reg, height, width, stride)
        mask = np.all(np.isfinite(boxes), axis=1) & (scores >= conf_thres)
        if not np.any(mask):
            continue
        boxes_all.append(boxes[mask])
        scores_all.append(scores[mask])
        classes_all.append(classes[mask])

    if not boxes_all:
        return []
    boxes = scale_boxes_to_original(np.concatenate(boxes_all), scale, pad_x, pad_y, image_shape)
    scores = np.concatenate(scores_all)
    classes = np.concatenate(classes_all)
    return _filter_boxes(boxes, scores, classes, iou_thres)


def _decode_reg(reg, height, width, stride):
    """(4,H,W) 的 ltrb 距离 -> (H*W,4) 像素框。"""
    reg_flat = reg.reshape(4, -1)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    x1 = (-reg_flat[0] + grid_x + 0.5) * stride
    y1 = (-reg_flat[1] + grid_y + 0.5) * stride
    x2 = (reg_flat[2] + grid_x + 0.5) * stride
    y2 = (reg_flat[3] + grid_y + 0.5) * stride
    return np.stack([x1, y1, x2, y2], axis=1)


def decode_detect_outputs(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres):
    dets = decode_detect_decoded(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres)
    if dets is not None:
        return dets
    dets = decode_detect_split_outputs(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres)
    if dets is not None:
        return dets
    shapes = [tuple(np.asarray(out).shape) for out in outputs]
    raise RuntimeError(f"Unsupported detect RKNN output shapes: {shapes}")


# --------------------------------------------------------------------------
# 量化路径（want_float=0）
# --------------------------------------------------------------------------
def decode_detect_quantized(outputs, infos, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres):
    """6 输出（4 回归 + 3 分类/stride）的 int8 解码，只在候选行上反量化。"""
    n = len(outputs)
    if n < 6 or n % 2 != 0:
        raise RuntimeError(f"quantized detect expects 4+3 split outputs, got {n}")

    strides = [8, 16, 32]
    boxes_all, scores_all, classes_all = [], [], []

    for i, stride in enumerate(strides):
        reg_q = np.asarray(outputs[i * 2])
        cls_q = np.asarray(outputs[i * 2 + 1])
        reg_i, cls_i = infos[i * 2], infos[i * 2 + 1]
        if reg_q.ndim != 3 or cls_q.ndim != 3:
            raise RuntimeError(f"unexpected quantized output shape {reg_q.shape}/{cls_q.shape}")

        n_cls = min(int(cls_q.shape[0]), len(DETECT_NAMES))
        cls_slice = cls_q[:n_cls].astype(np.int16)
        cls_idx = np.argmax(cls_slice, axis=0)                      # int 域 argmax == float 域 argmax
        q_max = np.take_along_axis(cls_slice, cls_idx[None], axis=0)[0]
        scores = (q_max.astype(np.float32) - cls_i.zp) * cls_i.scale
        scores = sigmoid(scores) if (scores.size and (scores.min() < 0.0 or scores.max() > 1.0)) else scores
        mask_flat = scores.reshape(-1) >= conf_thres
        if not mask_flat.any():
            continue

        classes = cls_idx.reshape(-1)
        height, width = reg_q.shape[1], reg_q.shape[2]
        boxes = _decode_reg_quantized(reg_q, reg_i, height, width, stride)
        boxes_all.append(boxes[mask_flat])
        scores_all.append(scores.reshape(-1)[mask_flat])
        classes_all.append(classes[mask_flat])

    if not boxes_all:
        return []
    boxes = scale_boxes_to_original(np.concatenate(boxes_all), scale, pad_x, pad_y, image_shape)
    scores = np.concatenate(scores_all)
    classes = np.concatenate(classes_all).astype(np.int32)
    return _filter_boxes(boxes, scores, classes, iou_thres)


def _decode_reg_quantized(reg_q, info, height, width, stride):
    """int8 回归头 -> 像素框；只对全场做一次 int16 提升，避免 float 全量转换。"""
    reg = reg_q.astype(np.int16)
    reg = (reg - int(info.zp)).astype(np.float32) * np.float32(info.scale)
    grid = np.arange(width, dtype=np.float32)
    grid_y = np.arange(height, dtype=np.float32)
    x1 = (-reg[0] + grid[None, :] + 0.5) * stride
    y1 = (-reg[1] + grid_y[:, None] + 0.5) * stride
    x2 = (reg[2] + grid[None, :] + 0.5) * stride
    y2 = (reg[3] + grid_y[:, None] + 0.5) * stride
    return np.stack(
        [x1.reshape(-1), y1.reshape(-1), x2.reshape(-1), y2.reshape(-1)], axis=1
    )


def _pose_channels(channels: int):
    if (channels - 5) % 3 == 0:
        return 4, 5, False   # (obj_col, kpt_start, needs_obj_cls) 单类别置信度
    if (channels - 6) % 3 == 0:
        return 4, 6, True    # obj * cls
    if (channels - 4) % 3 == 0:
        return None, 4, False  # 没有置信度列
    raise RuntimeError(f"Unsupported pose prediction channel count: {channels}")


def decode_pose_half(output, scale, pad_x, pad_y, image_shape, person_conf_thres, iou_thres):
    """pose 输出（原生 float16, (C,N) 或 (1,C,N)）：先按置信度筛选再转 float32。"""
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        raise RuntimeError(f"Unsupported pose output shape: {pred.shape}")
    if pred.shape[0] > pred.shape[1]:
        pred = pred.T
    channels = int(pred.shape[0])
    obj_col, kpt_start, needs_obj_cls = _pose_channels(channels)
    if (channels - kpt_start) % 3 != 0 or channels - kpt_start < 51:
        raise RuntimeError(f"Unsupported pose keypoint dims: {channels - kpt_start}")

    if obj_col is None:
        person_scores = np.ones((pred.shape[1],), dtype=np.float32)
    else:
        person_scores = pred[obj_col].astype(np.float32)
        person_scores = maybe_sigmoid(person_scores)
        if needs_obj_cls:
            cls_scores = maybe_sigmoid(pred[obj_col + 1].astype(np.float32))
            person_scores = person_scores * cls_scores

    idx = np.flatnonzero(person_scores >= person_conf_thres)
    if idx.size == 0:
        return []

    rows = pred[:, idx].astype(np.float32).T          # (n, C)，只转候选
    boxes_raw = rows[:, :4]
    kpts = rows[:, kpt_start:]
    boxes = xywh_to_xyxy(boxes_raw)
    boxes = scale_boxes_to_original(boxes, scale, pad_x, pad_y, image_shape)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, kpts, scores = boxes[valid], kpts[valid], person_scores[idx][valid]
    keep = nms_class_aware(boxes, scores, np.zeros((len(boxes),), dtype=np.int32), iou_thres)

    persons = []
    for i in keep:
        row = kpts[i].reshape(-1, 3)
        xy = scale_keypoints_to_original(row[:, :2], scale, pad_x, pad_y, image_shape)
        conf = maybe_sigmoid(row[:, 2])
        persons.append({"box": boxes[i].astype(np.float32), "score": float(scores[i]), "xy": xy, "conf": conf})
    return persons


def decode_pose_outputs(outputs, scale, pad_x, pad_y, image_shape, person_conf_thres, iou_thres):
    pred = as_prediction_matrix(outputs)
    if pred.size == 0 or pred.shape[1] < 5:
        shapes = [tuple(np.asarray(out).shape) for out in outputs]
        raise RuntimeError(f"Unsupported pose RKNN output shapes: {shapes}")

    channels = pred.shape[1]
    if (channels - 5) % 3 == 0:
        person_scores = maybe_sigmoid(pred[:, 4])
        kpt_start = 5
    elif (channels - 6) % 3 == 0:
        obj = maybe_sigmoid(pred[:, 4])
        cls = maybe_sigmoid(pred[:, 5])
        person_scores = obj * cls
        kpt_start = 6
    elif (channels - 4) % 3 == 0:
        person_scores = np.ones((len(pred),), dtype=np.float32)
        kpt_start = 4
    else:
        raise RuntimeError(f"Unsupported pose prediction channel count: {channels}")

    kpt_dims = channels - kpt_start
    if kpt_dims < 51 or kpt_dims % 3 != 0:
        raise RuntimeError(f"Unsupported pose keypoint dims: {kpt_dims}")

    boxes_raw = pred[:, :4]
    mask = np.all(np.isfinite(boxes_raw), axis=1) & np.isfinite(person_scores) & (person_scores >= person_conf_thres)
    if not np.any(mask):
        return []

    boxes = xywh_to_xyxy(boxes_raw[mask])
    scores = person_scores[mask]
    kpts = pred[mask, kpt_start:]
    boxes = scale_boxes_to_original(boxes, scale, pad_x, pad_y, image_shape)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, kpts = boxes[valid], scores[valid], kpts[valid]
    keep = nms_class_aware(boxes, scores, np.zeros((len(boxes),), dtype=np.int32), iou_thres)

    persons = []
    for idx in keep:
        row = kpts[idx].reshape(-1, 3)
        xy = scale_keypoints_to_original(row[:, :2], scale, pad_x, pad_y, image_shape)
        conf = maybe_sigmoid(row[:, 2])
        persons.append({"box": boxes[idx].astype(np.float32), "score": float(scores[idx]), "xy": xy, "conf": conf})
    return persons
