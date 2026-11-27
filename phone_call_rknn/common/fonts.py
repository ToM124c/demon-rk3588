"""中文字体加载（可视化状态面板用），带进程内缓存。"""

from __future__ import annotations

from typing import Any

try:  # Pillow 是可选依赖（没有就退回 cv2.putText 英文占位）
    from PIL import ImageFont
except ImportError:  # pragma: no cover
    ImageFont = None  # type: ignore[assignment]

CHINESE_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
_FONT_CACHE: dict[int, Any] = {}


def get_chinese_font(size: int):
    """返回 PIL ImageFont（找不到字体或没装 Pillow 时返回 None）。"""
    if ImageFont is None:
        return None
    if size not in _FONT_CACHE:
        font = None
        for path in CHINESE_FONT_PATHS:
            try:
                font = ImageFont.truetype(path, size=size)
                break
            except OSError:
                continue
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]
