"""帧对象：把硬解输出的 dma-buf fd / 内存布局与生命周期收拢到一处。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class UnsupportedSample(RuntimeError):
    """该 sample 无法以零拷贝方式表达（多平面/非 dma-buf/stride 不明）。"""


@dataclass
class Frame:
    """一帧解码结果。fd 优先（零拷贝），va 为可选的 CPU 视图。"""

    width: int
    height: int
    pixel_format: str = "nv12"
    fd: Optional[int] = None
    va: Optional[np.ndarray] = None
    wstride: Optional[int] = None
    hstride: Optional[int] = None
    pts: Optional[int] = None
    _keepalive: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.wstride is None:
            self.wstride = self.width
        if self.hstride is None:
            self.hstride = self.height

    @property
    def is_dmabuf(self) -> bool:
        return self.fd is not None

    @property
    def nbytes(self) -> int:
        return int(self.wstride) * int(self.hstride) * 3 // 2

    def release(self) -> None:
        """释放底层引用（GStreamer buffer / numpy 数组）。"""
        self._keepalive = None
        self.fd = None

    def __enter__(self) -> "Frame":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def frame_from_sample(sample, width: int, height: int, pixel_format: str = "nv12") -> Frame:
    """从 GStreamer appsink 的 sample 里取出 dma-buf fd 与内存布局。

    只在 `n_memory == 1` 且能推断出 stride 时返回 fd 帧；否则抛
    `UnsupportedSample`，调用方回退到 VA 路径。
    """
    from .gst import GstAllocators, GstVideo  # 延迟导入，避免无 gi 环境报错

    buf = sample.get_buffer()
    if buf is None:
        raise UnsupportedSample("empty buffer")
    if buf.n_memory() != 1:
        raise UnsupportedSample(f"expect 1 memory, got {buf.n_memory()}")
    mem = buf.get_memory(0)
    if not GstAllocators.is_dmabuf_memory(mem):
        raise UnsupportedSample("memory is not dma-buf")

    size = mem.get_sizes()[0]
    wstride, hstride = width, height
    if size != width * height * 3 // 2:
        meta = buf.get_meta(GstVideo.VideoMeta.get_info())
        if meta is None:
            raise UnsupportedSample(f"padded dma-buf {size}B without GstVideoMeta")
        wstride = int(meta.stride[0])
        hstride = hstride if wstride == width else (size * 2 // 3) // wstride

    fd = GstAllocators.dmabuf_memory_get_fd(mem)
    if fd < 0:
        raise UnsupportedSample("cannot get dmabuf fd")
    return Frame(
        width=width,
        height=height,
        pixel_format=pixel_format,
        fd=fd,
        wstride=wstride,
        hstride=hstride,
        pts=buf.pts,
        _keepalive=sample,
    )


def frame_to_bgr(frame: Frame, im2d=None) -> np.ndarray:
    """按需把一帧转成 BGR（可视化 / 违规取证），正常推理链路不会调用。"""
    from . import rga

    if frame.va is not None:
        if frame.va.ndim == 3 and frame.va.shape[2] == 3:
            return frame.va
        return rga.nv12_to_bgr(frame.va, frame.width, frame.height, frame.pixel_format, im2d=im2d)
    if frame.fd is not None:
        try:
            return rga.fd_to_bgr(
                frame.fd, frame.width, frame.height, frame.wstride, frame.hstride,
                frame.pixel_format, im2d=im2d,
            )
        except RuntimeError:
            from .gst import copy_frame_to_nv12

            nv12 = copy_frame_to_nv12(frame)
            return rga.nv12_to_bgr(nv12, frame.width, frame.height, frame.pixel_format, im2d=im2d)
    raise RuntimeError("frame has neither fd nor va")
