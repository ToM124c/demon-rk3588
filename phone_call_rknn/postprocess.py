import numpy as np

from .config import DETECT_NAMES
from .geometry import (
    box_center,
    maybe_sigmoid,
    nms_class_aware,
    scale_boxes_to_original,
    scale_keypoints_to_original,
    xywh_to_xyxy,
)


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
    valid = (classes >= 0) & (classes < len(DETECT_NAMES)) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    classes = classes[valid]
    scores = scores[valid]
    keep = nms_class_aware(boxes, scores, classes, iou_thres)
    return make_detection_dicts(boxes, scores, classes, keep)


def decode_detect_split_outputs(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres):
    if outputs is None or len(outputs) < 6:
        return None

    strides = [8, 16, 32]
    boxes_all = []
    scores_all = []
    classes_all = []

    for i, stride in enumerate(strides):
        reg = np.asarray(outputs[i * 2], dtype=np.float32)
        cls = np.asarray(outputs[i * 2 + 1], dtype=np.float32)
        reg = np.squeeze(reg)
        cls = np.squeeze(cls)
        if reg.ndim != 3 or cls.ndim != 3 or reg.shape[0] != 4 or cls.shape[0] < len(DETECT_NAMES):
            return None

        _, height, width = reg.shape
        reg_flat = reg.reshape(4, -1)
        cls_flat = cls[: len(DETECT_NAMES)].reshape(len(DETECT_NAMES), -1).T
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        x1 = (-reg_flat[0] + grid_x + 0.5) * stride
        y1 = (-reg_flat[1] + grid_y + 0.5) * stride
        x2 = (reg_flat[2] + grid_x + 0.5) * stride
        y2 = (reg_flat[3] + grid_y + 0.5) * stride
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        probs = maybe_sigmoid(cls_flat)
        classes = np.argmax(probs, axis=1).astype(np.int32)
        scores = probs[np.arange(len(probs)), classes]
        mask = np.all(np.isfinite(boxes), axis=1) & np.isfinite(scores) & (scores >= conf_thres)
        if not np.any(mask):
            continue
        boxes_all.append(boxes[mask])
        scores_all.append(scores[mask])
        classes_all.append(classes[mask])

    if not boxes_all:
        return []

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    classes = np.concatenate(classes_all, axis=0)
    boxes = scale_boxes_to_original(boxes, scale, pad_x, pad_y, image_shape)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    scores = scores[valid]
    classes = classes[valid]
    keep = nms_class_aware(boxes, scores, classes, iou_thres)
    return make_detection_dicts(boxes, scores, classes, keep)


def decode_detect_outputs(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres):
    dets = decode_detect_decoded(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres)
    if dets is not None:
        return dets
    dets = decode_detect_split_outputs(outputs, scale, pad_x, pad_y, image_shape, conf_thres, iou_thres)
    if dets is not None:
        return dets
    shapes = [tuple(np.asarray(out).shape) for out in outputs]
    raise RuntimeError(f"Unsupported detect RKNN output shapes: {shapes}")


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
    boxes = boxes[valid]
    scores = scores[valid]
    kpts = kpts[valid]
    keep = nms_class_aware(boxes, scores, np.zeros((len(boxes),), dtype=np.int32), iou_thres)

    persons = []
    for idx in keep:
        row = kpts[idx].reshape(-1, 3)
        xy = scale_keypoints_to_original(row[:, :2], scale, pad_x, pad_y, image_shape)
        conf = maybe_sigmoid(row[:, 2])
        persons.append({"box": boxes[idx].astype(np.float32), "score": float(scores[idx]), "xy": xy, "conf": conf})
    return persons

