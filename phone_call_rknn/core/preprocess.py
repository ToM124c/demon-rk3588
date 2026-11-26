import cv2
import numpy as np


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

