"""把一帧写进模型输入张量（dma-buf 优先，RGA 单次 CSC）。

从 `driver/engine.py` 抽出来的公共环节：司机模式和护送模式都要做同一件事，
区别只是“写进哪块张量 / 要不要写两次”。
"""

from __future__ import annotations

from ..hw.frame import Frame, frame_to_bgr
from ..hw.rga import RK_FORMAT_BGR_888


def preprocess_frame(letterboxer, frame: Frame, im2d=None):
    """把 `frame` 预处理好写进 `letterboxer` 绑定的目标张量。

    返回 `(dst_array_or_none, scale, pad_x, pad_y)`；目标是 dma-buf 时第一项为 None。
    """
    if im2d is None:
        im2d = letterboxer.im2d

    if frame.fd is not None:
        try:
            return letterboxer.from_fd(frame.fd, frame.width, frame.height,
                                       frame.pixel_format, frame.wstride, frame.hstride)
        except RuntimeError:
            # 10bit 等 RGA 不支持的格式：退回 CPU 转 BGR 再走 RGA
            bgr = frame_to_bgr(frame, im2d=im2d)
            src = im2d.wrap_va(bgr, frame.width, frame.height, frame.width, frame.height, RK_FORMAT_BGR_888)
            try:
                return letterboxer.from_buffer(src, frame.width, frame.height)
            finally:
                im2d.release(src)

    if frame.va is not None:
        if frame.va.ndim == 3:
            src = im2d.wrap_va(frame.va, frame.width, frame.height, frame.width, frame.height, RK_FORMAT_BGR_888)
            try:
                return letterboxer.from_buffer(src, frame.width, frame.height)
            finally:
                im2d.release(src)
        return letterboxer.from_nv12(frame.va, frame.width, frame.height, frame.pixel_format)

    raise RuntimeError("frame has neither dma-buf fd nor CPU buffer")
