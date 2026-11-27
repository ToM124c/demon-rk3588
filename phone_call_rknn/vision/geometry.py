import numpy as np


def sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def maybe_sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        return sigmoid(values)
    return values


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def scale_boxes_to_original(boxes, scale, pad_x, pad_y, image_shape):
    image_h, image_w = image_shape[:2]
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_h - 1)
    return boxes


def scale_keypoints_to_original(keypoints, scale, pad_x, pad_y, image_shape):
    image_h, image_w = image_shape[:2]
    points = np.asarray(keypoints, dtype=np.float32).copy()
    points[..., 0] = (points[..., 0] - pad_x) / scale
    points[..., 1] = (points[..., 1] - pad_y) / scale
    points[..., 0] = np.clip(points[..., 0], 0, image_w - 1)
    points[..., 1] = np.clip(points[..., 1], 0, image_h - 1)
    return points


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


def nms_class_aware(boxes, scores, classes, iou_thres):
    if len(boxes) == 0:
        return []
    keep = []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int32)
    for cls_id in np.unique(classes):
        idxs = np.where(classes == cls_id)[0]
        order = idxs[np.argsort(scores[idxs])[::-1]]
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            ious = box_iou(boxes[current], boxes[order[1:]])
            order = order[1:][ious <= iou_thres]
    keep.sort(key=lambda idx: float(scores[idx]), reverse=True)
    return keep


def point_dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def box_pair_iou(box_a, box_b):
    return float(box_iou(np.asarray(box_a, dtype=np.float32), np.asarray([box_b], dtype=np.float32))[0])


def is_finite_array(values):
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float32))))



def box_area(box):
    """xyxy 框面积。"""
    box = np.asarray(box, dtype=np.float32)
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def intersection_area(box_a, box_b):
    """两个 xyxy 框的交集面积。"""
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def point_in_box(point, box, margin=0.0):
    """点是否落在 xyxy 框内（可外扩 margin 像素）。"""
    if point is None:
        return False
    x, y = float(point[0]), float(point[1])
    box = np.asarray(box, dtype=np.float32)
    return (
        float(box[0]) - margin <= x <= float(box[2]) + margin
        and float(box[1]) - margin <= y <= float(box[3]) + margin
    )
