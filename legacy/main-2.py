import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
"""
先跑detect模型，当检测到两个hand和phone时候，计算hand和phone检测框中心点距离，
当出现min(distance(hand1,phone),distance(hand2,phone))距离小于150像素且检测框交并比大于0.1时候，则认为危险，否则判定为正常。
如果没检测到phone，则认为正常。
当出现一个hand和一个phone时候，先计算hand和phone的像素距离，如果小于150像素且检测框交并比大于0.1判定为危险。
如果超过150像素，则再将这一帧传给pose模型，计算关键点左右手腕和phone之间的距离，
如果min(distance(left_wrist,phone),distance(right_wrist,phone))距离小于150像素，则认为危险，否则判定为正常。
"""
POSE_MODEL = "/mnt/jl/yolov26-rk3588/model/yolo26n-pose_rknn_model/yolo26n-pose-rk3588.rknn"
DETECT_MODEL = "/mnt/jl/yolov26-rk3588/model-pose-detect/0701-hand-phone-seatbelt.rknn"
DEFAULT_SOURCE = "/mnt/jl/yolov26-rk3588/test/VID_20260603_102116.mp4"
DEFAULT_OUTPUT_DIR = "/mnt/jl/yolov26-rk3588/runs/0610-2"

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
PHONE_HAND_IOU_THRES = 0
WRIST_PHONE_PIXEL_THRES = 150.0
CHINESE_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
_FONT_CACHE = {}


def get_chinese_font(size):
    if ImageFont is None:
        return None
    if size not in _FONT_CACHE:
        font = None
        for path in CHINESE_FONT_PATHS:
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]


def draw_chinese_lines(canvas, lines, origin=(40, 40), font_size=32):
    font = get_chinese_font(font_size)
    if Image is None or ImageDraw is None or font is None:
        y = origin[1] + font_size
        for text, color in lines:
            fallback = text.encode("ascii", errors="ignore").decode("ascii") or "status"
            cv2.putText(canvas, fallback, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y += font_size + 8
        return canvas

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    x, y = origin
    line_gap = 8
    for text, color in lines:
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((x, y), text, font=font, fill=rgb, stroke_width=2, stroke_fill=(0, 0, 0))
        bbox = draw.textbbox((x, y), text, font=font, stroke_width=2)
        y = bbox[3] + line_gap
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


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


def resolve_rknn_path(model_path):
    path = Path(model_path)
    if path.is_file():
        return path
    matches = sorted(path.rglob("*.rknn"))
    if not matches:
        raise FileNotFoundError(f"No .rknn file found under: {path}")
    return matches[0]


def letterbox(image, size=640, color=(114, 114, 114)):
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
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, float(left), float(top)


def preprocess(image, imgsz, layout):
    resized, scale, pad_x, pad_y = letterbox(image, imgsz)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if layout == "nchw":
        rgb = rgb.transpose(2, 0, 1)
    return np.ascontiguousarray(rgb[None], dtype=np.uint8), scale, pad_x, pad_y


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


class RKNNRunner:
    def __init__(self, model_path, imgsz, input_layout):
        self.model_path = resolve_rknn_path(model_path)
        self.imgsz = imgsz
        self.input_layout = input_layout
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {self.model_path}, ret={ret}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {self.model_path}, ret={ret}")

    def infer(self, frame):
        runner_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        tensor, scale, pad_x, pad_y = preprocess(frame, self.imgsz, self.input_layout)
        preprocess_sec = time.perf_counter() - preprocess_start

        inference_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[tensor])
        rknn_inference_sec = time.perf_counter() - inference_start
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.model_path}")
        total_runner_sec = time.perf_counter() - runner_start
        timing = {
            "preprocess_ms": preprocess_sec * 1000.0,
            "postprocess_ms": 0.0,
            "rknn_inference_ms": rknn_inference_sec * 1000.0,
            "total_runner_ms": total_runner_sec * 1000.0,
        }
        return outputs, scale, pad_x, pad_y, timing

    def infer_tensor(self, tensor):
        inference_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[tensor])
        rknn_inference_sec = time.perf_counter() - inference_start
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.model_path}")
        timing = {
            "preprocess_ms": 0.0,
            "postprocess_ms": 0.0,
            "rknn_inference_ms": rknn_inference_sec * 1000.0,
            "total_runner_ms": rknn_inference_sec * 1000.0,
        }
        return outputs, timing

    def release(self):
        self.rknn.release()


def point_dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def box_pair_iou(box_a, box_b):
    return float(box_iou(np.asarray(box_a, dtype=np.float32), np.asarray([box_b], dtype=np.float32))[0])


def is_finite_array(values):
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float32))))


def person_scale_from_shoulders(xy, confs, kpt_conf):
    if LEFT_SHOULDER_IDX >= len(xy) or RIGHT_SHOULDER_IDX >= len(xy):
        return DEFAULT_PERSON_SCALE
    if confs[LEFT_SHOULDER_IDX] < kpt_conf or confs[RIGHT_SHOULDER_IDX] < kpt_conf:
        return DEFAULT_PERSON_SCALE
    left_shoulder = xy[LEFT_SHOULDER_IDX]
    right_shoulder = xy[RIGHT_SHOULDER_IDX]
    if not is_finite_array(left_shoulder) or not is_finite_array(right_shoulder):
        return DEFAULT_PERSON_SCALE
    shoulder_width = point_dist(left_shoulder, right_shoulder)
    if not np.isfinite(shoulder_width) or shoulder_width < MIN_PERSON_SCALE:
        return DEFAULT_PERSON_SCALE
    return shoulder_width


def get_wrist_points(persons, kpt_conf):
    wrists = []
    for person_idx, person in enumerate(persons):
        xy = np.asarray(person["xy"], dtype=np.float32)
        confs = np.asarray(person["conf"], dtype=np.float32)
        left = None
        right = None
        if LEFT_WRIST_IDX < len(xy) and confs[LEFT_WRIST_IDX] >= kpt_conf and is_finite_array(xy[LEFT_WRIST_IDX]):
            left = xy[LEFT_WRIST_IDX]
        if RIGHT_WRIST_IDX < len(xy) and confs[RIGHT_WRIST_IDX] >= kpt_conf and is_finite_array(xy[RIGHT_WRIST_IDX]):
            right = xy[RIGHT_WRIST_IDX]
        if left is not None or right is not None:
            wrists.append(
                {
                    "person_idx": person_idx,
                    "left": left,
                    "right": right,
                    "scale": person_scale_from_shoulders(xy, confs, kpt_conf),
                }
            )
    return wrists


def assign_hands_to_wrists(wrists, hand_dets, thres):
    assignments = {i: {"left": None, "right": None} for i in range(len(wrists))}
    candidates = []
    for wrist_idx, wrist in enumerate(wrists):
        scale = wrist.get("scale", DEFAULT_PERSON_SCALE)
        for side in ("left", "right"):
            wrist_point = wrist[side]
            if wrist_point is None:
                continue
            for hand_idx, hand in enumerate(hand_dets):
                dist_norm = point_dist(hand["center"], wrist_point) / scale
                if dist_norm <= thres:
                    candidates.append((dist_norm, wrist_idx, side, hand_idx))

    used_hands = set()
    for dist_norm, wrist_idx, side, hand_idx in sorted(candidates):
        if hand_idx in used_hands or assignments[wrist_idx][side] is not None:
            continue
        assignments[wrist_idx][side] = hand_dets[hand_idx]
        assignments[wrist_idx][side]["wrist_dist_norm"] = float(dist_norm)
        used_hands.add(hand_idx)
    return assignments


def check_phone_on_hands(assignments, phone_dets, wrists, thres):
    matches = []
    for wrist_idx, wrist in enumerate(wrists):
        scale = wrist.get("scale", DEFAULT_PERSON_SCALE)
        for side in ("left", "right"):
            hand = assignments[wrist_idx].get(side)
            if hand is None:
                continue
            for phone in phone_dets:
                dist_norm = point_dist(hand["center"], phone["center"]) / scale
                if dist_norm <= thres:
                    matches.append(
                        {
                            "person_idx": int(wrist["person_idx"]),
                            "side": side,
                            "distance_norm": float(dist_norm),
                            "hand_conf": float(hand["conf"]),
                            "phone_conf": float(phone["conf"]),
                        }
                    )
    return bool(matches), matches


def make_hand_phone_match(mode, hand_idx, phone_idx, hand, phone):
    distance_px = point_dist(hand["center"], phone["center"])
    return {
        "mode": mode,
        "hand_idx": int(hand_idx),
        "phone_idx": int(phone_idx),
        "distance_px": round(float(distance_px), 3),
        "iou": round(box_pair_iou(hand["box"], phone["box"]), 4),
        "hand_conf": round(float(hand["conf"]), 4),
        "phone_conf": round(float(phone["conf"]), 4),
    }


def hand_phone_matches(hand_dets, phone_dets, mode):
    matches = []
    for hand_idx, hand in enumerate(hand_dets):
        for phone_idx, phone in enumerate(phone_dets):
            matches.append(make_hand_phone_match(mode, hand_idx, phone_idx, hand, phone))
    return sorted(matches, key=lambda match: match["distance_px"])


def is_hand_phone_hit(match, args):
    return (
        match is not None
        and match["distance_px"] < args.phone_hand_pixel_thres
        and match["iou"] > args.phone_hand_iou_thres
    )


def evaluate_detect_rule(hand_dets, phone_dets, args):
    if not phone_dets:
        return False, [], "no_phone", False
    if not hand_dets:
        return False, [], "no_hand", False

    if len(hand_dets) >= 2:
        matches = hand_phone_matches(hand_dets, phone_dets, "detect_hand_phone")
        match = next((item for item in matches if is_hand_phone_hit(item, args)), matches[0] if matches else None)
        danger = is_hand_phone_hit(match, args)
        reason = "detect_2plus_hands_phone_hit" if danger else "detect_2plus_hands_phone_miss"
        return danger, [match] if match else [], reason, False

    matches = hand_phone_matches(hand_dets, phone_dets, "detect_single_hand_phone")
    match = next((item for item in matches if is_hand_phone_hit(item, args)), matches[0] if matches else None)
    if is_hand_phone_hit(match, args):
        return True, [match], "detect_single_hand_phone_hit", False
    if match is not None and match["distance_px"] <= args.phone_hand_pixel_thres:
        return False, [match], "detect_single_hand_phone_iou_miss", False

    return False, [match] if match else [], "need_pose_wrist_phone_check", True


def closest_wrist_phone_match(wrists, phone_dets):
    best_match = None
    for wrist_idx, wrist in enumerate(wrists):
        for side in ("left", "right"):
            wrist_point = wrist[side]
            if wrist_point is None:
                continue
            for phone_idx, phone in enumerate(phone_dets):
                distance_px = point_dist(wrist_point, phone["center"])
                match = {
                    "mode": "pose_wrist_phone",
                    "person_idx": int(wrist["person_idx"]),
                    "wrist_idx": int(wrist_idx),
                    "side": side,
                    "phone_idx": int(phone_idx),
                    "distance_px": round(float(distance_px), 3),
                    "phone_conf": round(float(phone["conf"]), 4),
                }
                if best_match is None or match["distance_px"] < best_match["distance_px"]:
                    best_match = match
    return best_match


def to_json_box(box):
    return [round(float(v), 2) for v in np.asarray(box, dtype=np.float32).reshape(-1)]


def to_json_point(point):
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def make_frame_record(
    frame_idx,
    source_name,
    persons,
    wrists,
    hand_dets,
    phone_dets,
    seatbelt_dets,
    danger,
    matches,
    decision,
):
    driver_seatbelt_on = bool(seatbelt_dets)
    return {
        "frame_index": int(frame_idx),
        "source": source_name,
        "danger": bool(danger),
        "driver_seatbelt_on": driver_seatbelt_on,
        "decision": decision,
        "person_count": len(persons),
        "wrist_count": len(wrists),
        "hand_count": len(hand_dets),
        "phone_count": len(phone_dets),
        "seatbelt_count": len(seatbelt_dets),
        "wrists": [
            {
                "person_idx": int(w["person_idx"]),
                "left": to_json_point(w["left"]),
                "right": to_json_point(w["right"]),
                "scale": round(float(w["scale"]), 3),
            }
            for w in wrists
        ],
        "hands": [{"box": to_json_box(det["box"]), "conf": round(float(det["conf"]), 4)} for det in hand_dets],
        "phones": [{"box": to_json_box(det["box"]), "conf": round(float(det["conf"]), 4)} for det in phone_dets],
        "seatbelts": [{"box": to_json_box(det["box"]), "conf": round(float(det["conf"]), 4)} for det in seatbelt_dets],
        "matches": matches,
    }


def draw_annotations(frame, wrists, assignments, hand_dets, phone_dets, seatbelt_dets, danger):
    canvas = frame.copy()
    for wrist_idx, wrist in enumerate(wrists):
        for side in ("left", "right"):
            point = wrist[side]
            if point is None:
                continue
            color = (0, 255, 0) if side == "left" else (255, 0, 0)
            cv2.circle(canvas, (int(point[0]), int(point[1])), 5, color, -1)
            hand = assignments[wrist_idx].get(side)
            if hand is not None:
                x1, y1, x2, y2 = map(int, hand["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{side}_hand {hand['conf']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    assigned_ids = {id(assignments[i][side]) for i in assignments for side in ("left", "right") if assignments[i][side] is not None}
    for hand in hand_dets:
        if id(hand) in assigned_ids:
            continue
        x1, y1, x2, y2 = map(int, hand["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 1)
        cv2.putText(canvas, f"hand {hand['conf']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)

    for phone in phone_dets:
        x1, y1, x2, y2 = map(int, phone["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(canvas, f"phone {phone['conf']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    for seatbelt in seatbelt_dets:
        x1, y1, x2, y2 = map(int, seatbelt["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            canvas,
            f"seatbelt {seatbelt['conf']:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )

    call_color = (0, 0, 255) if danger else (0, 255, 0)
    seatbelt_on = bool(seatbelt_dets)
    seatbelt_color = (0, 255, 0) if seatbelt_on else (0, 0, 255)
    canvas = draw_chinese_lines(
        canvas,
        [
            (f"打电话：{'是' if danger else '否'}", call_color),
            (f"驾驶员安全带：{'已系' if seatbelt_on else '未系'}", seatbelt_color),
        ],
    )
    return canvas


def run_one_frame(frame, frame_idx, source_name, pose_runner, detect_runner, args):
    shared_preprocess_ms = 0.0
    pose_timing = zero_runner_timing()
    persons = []
    wrists = []
    assignments = {}

    share_preprocess = (
        pose_runner.imgsz == detect_runner.imgsz
        and pose_runner.input_layout == detect_runner.input_layout
    )
    if share_preprocess:
        preprocess_start = time.perf_counter()
        tensor, det_scale, det_pad_x, det_pad_y = preprocess(frame, detect_runner.imgsz, detect_runner.input_layout)
        shared_preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        det_outputs, detect_timing = detect_runner.infer_tensor(tensor)
    else:
        tensor = None
        det_outputs, det_scale, det_pad_x, det_pad_y, detect_timing = detect_runner.infer(frame)

    detect_postprocess_start = time.perf_counter()
    dets = decode_detect_outputs(det_outputs, det_scale, det_pad_x, det_pad_y, frame.shape, args.det_conf, args.iou)
    hand_dets = [det for det in dets if det["class"] == "hand"]
    phone_dets = [det for det in dets if det["class"] == "phone"]
    seatbelt_dets = [det for det in dets if det["class"] == "seatbelt"]
    danger, matches, decision, need_pose = evaluate_detect_rule(hand_dets, phone_dets, args)
    detect_timing["postprocess_ms"] = (time.perf_counter() - detect_postprocess_start) * 1000.0
    detect_timing["total_runner_ms"] += detect_timing["postprocess_ms"]

    if need_pose:
        if share_preprocess:
            pose_outputs, pose_timing = pose_runner.infer_tensor(tensor)
            pose_scale = det_scale
            pose_pad_x = det_pad_x
            pose_pad_y = det_pad_y
        else:
            pose_outputs, pose_scale, pose_pad_x, pose_pad_y, pose_timing = pose_runner.infer(frame)
        pose_postprocess_start = time.perf_counter()
        persons = decode_pose_outputs(
            pose_outputs, pose_scale, pose_pad_x, pose_pad_y, frame.shape, args.person_conf, args.iou
        )
        wrists = get_wrist_points(persons, args.kpt_conf)
        assignments = {i: {"left": None, "right": None} for i in range(len(wrists))}
        pose_match = closest_wrist_phone_match(wrists, phone_dets)
        if pose_match is not None:
            danger = pose_match["distance_px"] < args.wrist_phone_pixel_thres
            matches.append(pose_match)
            decision = "pose_wrist_phone_hit" if danger else "pose_wrist_phone_miss"
        else:
            decision = "pose_no_valid_wrist"
        pose_timing["postprocess_ms"] = (time.perf_counter() - pose_postprocess_start) * 1000.0
        pose_timing["total_runner_ms"] += pose_timing["postprocess_ms"]

    record = make_frame_record(
        frame_idx, source_name, persons, wrists, hand_dets, phone_dets, seatbelt_dets, danger, matches, decision
    )
    total_preprocess_ms = shared_preprocess_ms + pose_timing["preprocess_ms"] + detect_timing["preprocess_ms"]
    total_postprocess_ms = pose_timing["postprocess_ms"] + detect_timing["postprocess_ms"]
    total_rknn_inference_ms = pose_timing["rknn_inference_ms"] + detect_timing["rknn_inference_ms"]
    total_runner_ms = total_preprocess_ms + total_postprocess_ms + total_rknn_inference_ms
    timing = {
        "pose": pose_timing,
        "detect": detect_timing,
        "total": {
            "shared_preprocess_ms": shared_preprocess_ms,
            "preprocess_ms": total_preprocess_ms,
            "postprocess_ms": total_postprocess_ms,
            "rknn_inference_ms": total_rknn_inference_ms,
            "total_runner_ms": total_runner_ms,
            "runner_fps": 1000.0 / total_runner_ms if total_runner_ms > 0 else 0.0,
        },
    }
    record["timing"] = round_timing(timing)
    vis = draw_annotations(frame, wrists, assignments, hand_dets, phone_dets, seatbelt_dets, danger)
    return record, vis, timing


def zero_runner_timing():
    return {
        "preprocess_ms": 0.0,
        "postprocess_ms": 0.0,
        "rknn_inference_ms": 0.0,
        "total_runner_ms": 0.0,
    }


def round_timing(timing):
    return {
        name: {key: round(float(value), 3) for key, value in values.items()}
        for name, values in timing.items()
    }


def print_frame_speed(frame_idx, source_name, timing):
    pose = timing["pose"]
    detect = timing["detect"]
    total = timing["total"]
    print(
        f"Frame speed: source={source_name}, frame={frame_idx}, "
        f"shared_pre={total['shared_preprocess_ms']:.2f}ms, "
        f"pose(pre={pose['preprocess_ms']:.2f}ms,post={pose['postprocess_ms']:.2f}ms,"
        f"rknn={pose['rknn_inference_ms']:.2f}ms,total={pose['total_runner_ms']:.2f}ms), "
        f"detect(pre={detect['preprocess_ms']:.2f}ms,post={detect['postprocess_ms']:.2f}ms,"
        f"rknn={detect['rknn_inference_ms']:.2f}ms,total={detect['total_runner_ms']:.2f}ms), "
        f"total(pre={total['preprocess_ms']:.2f}ms,post={total['postprocess_ms']:.2f}ms,rknn={total['rknn_inference_ms']:.2f}ms,"
        f"runner={total['total_runner_ms']:.2f}ms,fps={total['runner_fps']:.2f})"
    )


def new_timing_sums():
    return {
        "shared_preprocess_ms": 0.0,
        "pose_preprocess_ms": 0.0,
        "pose_postprocess_ms": 0.0,
        "pose_rknn_inference_ms": 0.0,
        "pose_frame_count": 0,
        "detect_preprocess_ms": 0.0,
        "detect_postprocess_ms": 0.0,
        "detect_rknn_inference_ms": 0.0,
        "total_preprocess_ms": 0.0,
        "total_postprocess_ms": 0.0,
        "total_rknn_inference_ms": 0.0,
        "total_runner_ms": 0.0,
    }


def add_timing_sums(timing_sums, timing):
    timing_sums["shared_preprocess_ms"] += timing["total"]["shared_preprocess_ms"]
    timing_sums["pose_preprocess_ms"] += timing["pose"]["preprocess_ms"]
    timing_sums["pose_postprocess_ms"] += timing["pose"]["postprocess_ms"]
    timing_sums["pose_rknn_inference_ms"] += timing["pose"]["rknn_inference_ms"]
    if timing["pose"]["total_runner_ms"] > 0:
        timing_sums["pose_frame_count"] += 1
    timing_sums["detect_preprocess_ms"] += timing["detect"]["preprocess_ms"]
    timing_sums["detect_postprocess_ms"] += timing["detect"]["postprocess_ms"]
    timing_sums["detect_rknn_inference_ms"] += timing["detect"]["rknn_inference_ms"]
    timing_sums["total_preprocess_ms"] += timing["total"]["preprocess_ms"]
    timing_sums["total_postprocess_ms"] += timing["total"]["postprocess_ms"]
    timing_sums["total_rknn_inference_ms"] += timing["total"]["rknn_inference_ms"]
    timing_sums["total_runner_ms"] += timing["total"]["total_runner_ms"]


def print_overall_speed(label, frame_count, timing_sums):
    runner_ms_total = timing_sums["total_runner_ms"]
    if frame_count <= 0 or runner_ms_total <= 0:
        print(f"{label} speed summary: no frames")
        return
    avg_ms = runner_ms_total / frame_count
    fps = 1000.0 * frame_count / runner_ms_total
    pose_frame_count = timing_sums["pose_frame_count"]
    pose_divisor = pose_frame_count if pose_frame_count > 0 else 1
    pose_avg_ms = timing_sums["pose_rknn_inference_ms"] / pose_divisor if pose_frame_count > 0 else 0.0
    print(f"{label} speed summary: frames={frame_count}, avg_total={avg_ms:.2f}ms/frame, fps={fps:.2f}")
    print(f"前处理时间: total={timing_sums['total_preprocess_ms']:.2f}ms, avg={timing_sums['total_preprocess_ms'] / frame_count:.2f}ms/frame")
    print(f"后处理时间: total={timing_sums['total_postprocess_ms']:.2f}ms, avg={timing_sums['total_postprocess_ms'] / frame_count:.2f}ms/frame")
    print(
        f"模型推理时间: total={timing_sums['total_rknn_inference_ms']:.2f}ms, "
        f"avg={timing_sums['total_rknn_inference_ms'] / frame_count:.2f}ms/frame"
    )
    print(
        f"detect模型推理时间: total={timing_sums['detect_rknn_inference_ms']:.2f}ms, "
        f"avg={timing_sums['detect_rknn_inference_ms'] / frame_count:.2f}ms/frame"
    )
    print(
        f"pose模型推理时间: frames={pose_frame_count}, total={timing_sums['pose_rknn_inference_ms']:.2f}ms, "
        f"avg={pose_avg_ms:.2f}ms/pose-frame"
    )


def write_record(jsonl_file, record, quiet):
    line = json.dumps(record, ensure_ascii=False)
    jsonl_file.write(line + "\n")
    jsonl_file.flush()
    if not quiet:
        print(line)


def is_video(path):
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def image_files(path):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return []


def process_images(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file):
    paths = image_files(source_path)
    if not paths:
        raise FileNotFoundError(f"No supported image files found: {source_path}")
    vis_dir = output_dir / "vis_images"
    if not args.no_save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    danger_count = 0
    timing_sums = new_timing_sums()
    for idx, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        record, vis, timing = run_one_frame(frame, idx, path.name, pose_runner, detect_runner, args)
        add_timing_sums(timing_sums, timing)
        danger_count += int(record["danger"])
        write_record(jsonl_file, record, args.quiet)
        if not args.no_save_vis:
            cv2.imwrite(str(vis_dir / f"{path.stem}_phone_call.jpg"), vis)
    print(f"Images done: total={len(paths)}, danger={danger_count}")
    print_overall_speed("Images", len(paths), timing_sums)


def process_video(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    out_video = output_dir / f"{source_path.stem}_phone_call.mp4"
    if not args.no_save_vis:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create video writer: {out_video}")

    frame_idx = 0
    danger_count = 0
    timing_sums = new_timing_sums()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        record, vis, timing = run_one_frame(frame, frame_idx, source_path.name, pose_runner, detect_runner, args)
        add_timing_sums(timing_sums, timing)
        # print_frame_speed(frame_idx, source_path.name, timing)
        danger_count += int(record["danger"])
        write_record(jsonl_file, record, args.quiet)
        if writer is not None:
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Visualization video saved: {out_video}")
    print(f"Video done: total_frames={frame_idx}, danger_frames={danger_count}")
    print_overall_speed("Video", frame_idx, timing_sums)


def main():
    args = parse_args()
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "frame_results.jsonl"

    pose_runner = RKNNRunner(args.pose_model, args.imgsz, args.pose_input_layout)
    detect_runner = RKNNRunner(args.detect_model, args.imgsz, args.detect_input_layout)
    try:
        with result_path.open("w", encoding="utf-8") as jsonl_file:
            if is_video(source_path):
                process_video(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file)
            else:
                process_images(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file)
        print(f"Frame results saved: {result_path}")
    finally:
        pose_runner.release()
        detect_runner.release()


if __name__ == "__main__":
    main()
