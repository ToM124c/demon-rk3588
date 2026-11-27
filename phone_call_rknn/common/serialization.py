"""坐标序列化：JSON 里的框/点统一保留两位小数。"""

from __future__ import annotations

import numpy as np


def to_json_box(box) -> list:
    return [round(float(v), 2) for v in np.asarray(box, dtype=np.float32).reshape(-1)]


def to_json_point(point) -> list | None:
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]
