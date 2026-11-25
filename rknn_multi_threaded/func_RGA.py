import cv2
import numpy as np
import time

from rknn_multi_threaded.rga_helper import get_rga_letterbox
from rknn_multi_threaded.vpu_capture import NV12Frame

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.35, 0.45, 640

CLASSES = ("hand", "phone")


def letterbox_cpu(image, size=IMG_SIZE, color=(114, 114, 114)):
    src_h, src_w = image.shape[:2]
    scale = min(size / src_w, size / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top: top + new_h, left: left + new_w] = resized
    return canvas, scale, float(left), float(top)


def letterbox(image, size=IMG_SIZE, color=(114, 114, 114)):
    bgr_canvas, scale, pad_x, pad_y = letterbox_cpu(image, size=size, color=color)
    return bgr_canvas, scale, pad_x, pad_y


def nv12_to_bgr_cpu(nv12_data, width, height):
    return cv2.cvtColor(nv12_data, cv2.COLOR_YUV2BGR_NV12)


def preprocess(image):
    rga = get_rga_letterbox(size=IMG_SIZE)
    if isinstance(image, NV12Frame):
        if rga is not None:
            try:
                rgb, scale, pad_x, pad_y, bgr = rga.letterbox_from_nv12(
                    image.data, image.width, image.height, image.pixel_format
                )
                tensor = np.ascontiguousarray(rgb[None], dtype=np.uint8)
                return tensor, scale, pad_x, pad_y, bgr
            except RuntimeError:
                pass
        if image.pixel_format == "nv12":
            bgr = nv12_to_bgr_cpu(image.data, image.width, image.height)
        else:
            raise RuntimeError("NV12_10LE40 preprocess requires RGA")
        bgr_canvas, scale, pad_x, pad_y = letterbox_cpu(bgr)
        rgb = cv2.cvtColor(bgr_canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb[None], dtype=np.uint8)
        return tensor, scale, pad_x, pad_y, bgr

    if rga is not None:
        try:
            rgb, scale, pad_x, pad_y = rga.letterbox(image)
            tensor = np.ascontiguousarray(rgb[None], dtype=np.uint8)
            return tensor, scale, pad_x, pad_y, image
        except RuntimeError:
            pass

    bgr_canvas, scale, pad_x, pad_y = letterbox_cpu(image)
    rgb = cv2.cvtColor(bgr_canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb[None], dtype=np.uint8)
    return tensor, scale, pad_x, pad_y, image


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


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


def nms_class_aware(boxes, scores, classes, iou_thres=NMS_THRESH):
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
        cls_id = int(classes[idx])
        if cls_id < 0 or cls_id >= len(CLASSES):
            continue
        box = boxes[idx].astype(np.float32)
        dets.append(
            {
                "box": box,
                "conf": float(scores[idx]),
                "class_id": cls_id,
                "class": CLASSES[cls_id],
            }
        )
    return dets


def decode_detect_decoded(outputs, scale, pad_x, pad_y, image_shape):
    pred = as_prediction_matrix(outputs)
    if pred.size == 0 or pred.shape[1] < 6:
        return None

    boxes_raw = pred[:, :4]
    class_scores = maybe_sigmoid(pred[:, 4: 4 + len(CLASSES)])
    classes = np.argmax(class_scores, axis=1).astype(np.int32)
    scores = class_scores[np.arange(len(pred)), classes]
    mask = np.all(np.isfinite(boxes_raw), axis=1) & np.isfinite(scores) & (scores >= OBJ_THRESH)
    if not np.any(mask):
        return []

    boxes_raw = boxes_raw[mask]
    classes = classes[mask]
    scores = scores[mask]
    x1, y1, x2, y2 = boxes_raw.T
    xyxy_ratio = np.mean((x2 > x1) & (y2 > y1))
    boxes = boxes_raw.copy() if xyxy_ratio > 0.85 else xywh_to_xyxy(boxes_raw)
    boxes = scale_boxes_to_original(boxes, scale, pad_x, pad_y, image_shape)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    scores = scores[valid]
    classes = classes[valid]
    keep = nms_class_aware(boxes, scores, classes)
    return make_detection_dicts(boxes, scores, classes, keep)


def decode_detect_split_outputs(outputs, scale, pad_x, pad_y, image_shape):
    if outputs is None or len(outputs) < 6:
        return None

    strides = [8, 16, 32]
    boxes_all = []
    scores_all = []
    classes_all = []

    for i, stride in enumerate(strides):
        reg = np.squeeze(np.asarray(outputs[i * 2], dtype=np.float32))
        cls = np.squeeze(np.asarray(outputs[i * 2 + 1], dtype=np.float32))
        if reg.ndim != 3 or cls.ndim != 3 or reg.shape[0] != 4 or cls.shape[0] < len(CLASSES):
            return None

        _, height, width = reg.shape
        reg_flat = reg.reshape(4, -1)
        cls_flat = cls[: len(CLASSES)].reshape(len(CLASSES), -1).T
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
        mask = np.all(np.isfinite(boxes), axis=1) & np.isfinite(scores) & (scores >= OBJ_THRESH)
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
    keep = nms_class_aware(boxes, scores, classes)
    return make_detection_dicts(boxes, scores, classes, keep)


def decode_detect_outputs(outputs, scale, pad_x, pad_y, image_shape):
    dets = decode_detect_decoded(outputs, scale, pad_x, pad_y, image_shape)
    if dets is not None:
        return dets

    dets = decode_detect_split_outputs(outputs, scale, pad_x, pad_y, image_shape)
    if dets is not None:
        return dets

    shapes = [tuple(np.asarray(out).shape) for out in outputs]
    raise RuntimeError("Unsupported detect RKNN output shapes: {}".format(shapes))


def draw(image, dets):
    colors = {
        "hand": (0, 180, 0),
        "phone": (0, 255, 255),
    }
    for det in dets:
        x1, y1, x2, y2 = map(int, det["box"])
        color = colors.get(det["class"], (255, 0, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            "{} {:.2f}".format(det["class"], det["conf"]),
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def myFunc(rknn_lite, IMG):
    preprocess_start = time.perf_counter()
    tensor, scale, pad_x, pad_y, bgr = preprocess(IMG)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    infer_start = time.perf_counter()
    outputs = rknn_lite.inference(inputs=[tensor])
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    postprocess_start = time.perf_counter()
    dets = decode_detect_outputs(outputs, scale, pad_x, pad_y, bgr.shape)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

    draw_start = time.perf_counter()
    result = bgr.copy()
    draw(result, dets)
    draw_ms = (time.perf_counter() - draw_start) * 1000.0
    del bgr

    timing = {
        "pre_ms": preprocess_ms,
        "infer_ms": infer_ms,
        "post_ms": postprocess_ms,
        "draw_ms": draw_ms,
    }
    return result, timing
