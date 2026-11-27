"""GStreamer 层：硬解出 dma-buf、硬编 H.264。

- `DmaBufCapture`：`mppvideodec` 输出 `video/x-raw(memory:DMABuf)`，
  通过 appsink 拿到 **dma-buf fd**，全程不进 CPU 内存（零拷贝起点）。
- `OpenCVNv12Capture`：没装 PyGObject 时的兜底（appsink 取 NV12 到 numpy）。
- `VpuWriter`：`appsrc -> videoconvert -> mpph264enc -> mp4mux` 硬编。

依赖：系统 `python3-gi`（Debian/Ubuntu 自带）。conda 环境里通过把
`/usr/lib/python3/dist-packages` 加入 sys.path 复用系统 gi。
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .frame import Frame, UnsupportedSample, frame_from_sample

_SYSTEM_GI_PATH = "/usr/lib/python3/dist-packages"

Gst: Any = None
GstApp: Any = None
GstVideo: Any = None
GstAllocators: Any = None
GST_IMPORT_ERROR: Optional[Exception] = None


def _import_gst():
    """导入 gi/GStreamer；失败时记录原因（供上层回退）。"""
    global Gst, GstApp, GstVideo, GstAllocators, GST_IMPORT_ERROR
    if Gst is not None:
        return True
    if os.environ.get("PHONE_CALL_DISABLE_GST") == "1":
        GST_IMPORT_ERROR = RuntimeError("disabled by PHONE_CALL_DISABLE_GST")
        return False
    added = False
    try:
        import gi  # noqa: F401
    except ModuleNotFoundError:
        if os.path.isdir(_SYSTEM_GI_PATH) and _SYSTEM_GI_PATH not in sys.path:
            sys.path.append(_SYSTEM_GI_PATH)
            added = True
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        gi.require_version("GstVideo", "1.0")
        gi.require_version("GstAllocators", "1.0")
        from gi.repository import Gst as _Gst, GstApp as _GstApp
        from gi.repository import GstVideo as _GstVideo, GstAllocators as _GstAllocators

        _Gst.init(None)
        Gst, GstApp, GstVideo, GstAllocators = _Gst, _GstApp, _GstVideo, _GstAllocators
        return True
    except Exception as exc:  # pragma: no cover - 依赖运行环境
        GST_IMPORT_ERROR = exc
        if added:
            sys.path.remove(_SYSTEM_GI_PATH)
        return False


def gst_available() -> bool:
    return _import_gst()


_SINK_PROPS = "appsink drop=false sync=false max-buffers=4"


class DmaBufCapture:
    """硬解 + dma-buf appsink，`read()` 返回 `Frame(fd=...)`。"""

    def __init__(self, source, codec: str = "h264", width: Optional[int] = None,
                 height: Optional[int] = None, fps: Optional[float] = None,
                 rtsp_latency: int = 100):
        if not _import_gst():
            raise RuntimeError(f"PyGObject/GStreamer 不可用: {GST_IMPORT_ERROR}")
        self.source = str(source)
        self.codec = codec.lower()
        self.rtsp_latency = rtsp_latency
        self.pipeline_desc = self._build(width, height, fps)
        self.pipeline = Gst.parse_launch(self.pipeline_desc)
        self.appsink = self.pipeline.get_by_name("sink")
        self._lock = threading.Lock()
        self._eos = False
        self._pending = None
        self.pipeline.set_state(Gst.State.PLAYING)
        self.width, self.height = 0, 0
        self._probe_caps()

    # ---- 构图 -----------------------------------------------------
    def _decoder(self) -> str:
        if self.codec not in {"h264", "h265", "hevc"}:
            raise ValueError("codec must be one of: h264, h265, hevc")
        parser = "h265parse" if self.codec in {"h265", "hevc"} else "h264parse"
        return f"{parser} ! mppvideodec"

    def _build(self, width, height, fps) -> str:
        mem_caps = "video/x-raw(memory:DMABuf),format=NV12"
        sink = f"{mem_caps} ! {_SINK_PROPS} name=sink"
        src = self.source
        if src.isdigit():
            caps = ""
            parts = []
            if width:
                parts.append(f"width={int(width)}")
            if height:
                parts.append(f"height={int(height)}")
            if fps:
                parts.append(f"framerate={int(fps)}/1")
            if parts:
                caps = " ! video/x-raw," + ",".join(parts)
            # 摄像头一般拿不到 dma-buf，交给 OpenCV 兜底；这里仅尝试
            return f"v4l2src device=/dev/video{src}{caps} ! videoconvert ! {sink}"
        if src.startswith(("rtsp://", "rtmp://")):
            depay = "rtph265depay" if self.codec in {"h265", "hevc"} else "rtph264depay"
            return (
                f'rtspsrc location="{src}" latency={self.rtsp_latency} ! '
                f"{depay} ! {self._decoder()} ! {sink}"
            )
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {src}")
        demux = self._demux_for_suffix(path.suffix.lower())
        location = str(path).replace('"', '\\"')
        return f'filesrc location="{location}" ! {demux} ! {self._decoder()} ! {sink}'

    @staticmethod
    def _demux_for_suffix(suffix: str) -> str:
        if suffix in {".mp4", ".mov", ".m4v"}:
            return "qtdemux"
        if suffix in {".mkv", ".webm"}:
            return "matroskademux"
        if suffix in {".ts", ".mts", ".m2ts"}:
            return "tsdemux"
        if suffix in {".h264", ".264", ".h265", ".265", ".hevc"}:
            return "identity"
        raise ValueError(f"unsupported container: {suffix}")

    def _probe_caps(self) -> None:
        sample = self.appsink.try_pull_sample(int(2e9))
        if sample is None:
            raise RuntimeError(f"cannot pull first sample from pipeline:\n{self.pipeline_desc}")
        caps = sample.get_caps().get_structure(0)
        self.width = int(caps.get_value("width"))
        self.height = int(caps.get_value("height"))
        self._pending = sample  # 第一帧留给第一次 read()

    # ---- 读取 -----------------------------------------------------
    def is_eos(self) -> bool:
        return self._eos

    def read(self):
        with self._lock:
            sample = self._pending
            self._pending = None
            if sample is None:
                sample = self.appsink.try_pull_sample(int(1e9))
            if sample is None:
                self._eos = True
                return None
            try:
                return frame_from_sample(sample, self.width, self.height, self._pixel_format())
            except UnsupportedSample:
                return None

    def drop_one(self) -> bool:
        """丢弃一帧：不取 fd、不建 Frame（抽帧时用，最省 CPU）。"""
        with self._lock:
            if self._pending is not None:
                self._pending = None
                return True
            return self.appsink.try_pull_sample(int(1e9)) is not None

    def _pixel_format(self) -> str:
        return "nv12_10le40" if self.codec in {"h265", "hevc"} else "nv12"

    def release(self) -> None:
        pipeline = self.pipeline
        try:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
        finally:
            self.pipeline = None


class OpenCVNv12Capture:
    """兜底：OpenCV + GStreamer appsink，NV12 进 numpy（有一次拷贝）。"""

    def __init__(self, source, codec: str = "h264", fallback: bool = True):
        self.source = str(source)
        self.codec = codec.lower()
        self.pipeline = self._build()
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER) if self.pipeline else None
        if (self.cap is None or not self.cap.isOpened()) and fallback:
            self.cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
            self.pipeline = None
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"cannot open capture: {self.source}\n{self.pipeline}")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if h > w and h > w * 0.75:      # OpenCV NV12 appsink 常报打包高度
            h = h * 2 // 3
        self.width, self.height = w, h

    def _build(self):
        if self.source.isdigit():
            return None
        sink = "video/x-raw,format=NV12 ! appsink drop=false sync=false max-buffers=4"
        if self.source.startswith(("rtsp://", "rtmp://")):
            depay = "rtph265depay" if self.codec in {"h265", "hevc"} else "rtph264depay"
            return f'rtspsrc location="{self.source}" latency=100 ! {depay} ! {self._decoder()} ! {sink}'
        path = Path(self.source)
        return f'filesrc location="{path}" ! {DmaBufCapture._demux_for_suffix(path.suffix.lower())} ! {self._decoder()} ! {sink}'

    def _decoder(self):
        parser = "h265parse" if self.codec in {"h265", "hevc"} else "h264parse"
        return f"{parser} ! mppvideodec"

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        if frame.ndim == 2:
            return Frame(width=self.width, height=self.height, pixel_format="nv12", va=frame, _keepalive=frame)
        return Frame(width=self.width, height=self.height, pixel_format="bgr", va=frame, _keepalive=frame)

    def drop_one(self) -> bool:
        return bool(self.cap is not None and self.cap.grab())

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class CvCapture:
    """纯 CPU 软解（对照/兜底）：OpenCV 直接读，返回 BGR 帧。"""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open capture: {source}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.pipeline_desc = None

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return Frame(width=self.width, height=self.height, pixel_format="bgr", va=frame, _keepalive=frame)

    def drop_one(self) -> bool:
        return bool(self.cap is not None and self.cap.grab())

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_capture(source, codec: str = "h264", zero_copy: bool = True, fallback: bool = True,
                 cpu: bool = False):
    """按优先级选择采集后端：CPU 软解 / dma-buf 零拷贝 / NV12 兜底。"""
    if cpu:
        return CvCapture(source)
    if zero_copy and _import_gst():
        try:
            return DmaBufCapture(source, codec=codec)
        except Exception as exc:  # pragma: no cover - 板端差异
            if not fallback:
                raise RuntimeError(f"dma-buf capture failed: {exc}") from exc
    return OpenCVNv12Capture(source, codec=codec, fallback=fallback)


class VpuWriter:
    """动图/视频输出：`appsrc -> videoconvert -> mpph264enc -> mp4mux`。"""

    def __init__(self, output_path, width: int, height: int, fps: float,
                 bitrate: Optional[int] = None, fallback: bool = False):
        self.output_path = Path(output_path)
        if self.output_path.suffix.lower() != ".mp4":
            self.output_path = self.output_path.with_suffix(".mp4")
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.bitrate = bitrate
        self.pipeline = self._build()
        self.writer = cv2.VideoWriter(self.pipeline, cv2.CAP_GSTREAMER, 0, self.fps, (self.width, self.height))
        if not self.writer.isOpened() and fallback:
            self.pipeline = None
            self.writer = cv2.VideoWriter(str(self.output_path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps,
                                          (self.width, self.height))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {self.output_path}")

    def _build(self):
        location = str(self.output_path).replace('"', '\\"')
        enc = f"mpph264enc bps={int(self.bitrate)}" if self.bitrate else "mpph264enc"
        return (
            "appsrc is-live=false block=true max-bytes=0 do-timestamp=true ! "
            "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
            "video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=NV12 ! "
            f"{enc} ! h264parse ! "
            "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
            "mp4mux fragment-duration=1000000000 fragment-mode=first-moov-then-finalise ! "
            f'filesink location="{location}"'
        )

    def write(self, frame):
        return self.writer.write(frame)

    def release(self):
        self.writer.release()


def open_writer(output_path, width, height, fps, encoder: str = "vpu", bitrate=None, fallback: bool = False):
    if encoder in {"vpu", "mpp"}:
        try:
            return VpuWriter(output_path, width, height, fps, bitrate=bitrate, fallback=fallback)
        except Exception:
            if not fallback:
                raise
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open mp4v writer: {output_path}")
    return writer


def copy_frame_to_nv12(frame) -> np.ndarray:
    """把 dma-buf 帧映射拷贝成 numpy NV12（仅用于 RGA 无法直接读 fd 的兜底路径）。"""
    if frame.va is not None:
        return frame.va
    sample = frame._keepalive
    if sample is None or not _import_gst():
        raise RuntimeError("cannot copy frame: no sample / PyGObject unavailable")
    buf = sample.get_buffer()
    ok, info = buf.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("gst_buffer_map failed")
    try:
        nbytes = frame.width * frame.height * 3 // 2
        data = np.frombuffer(info.data, dtype=np.uint8, count=nbytes)
        return data.reshape(frame.height * 3 // 2, frame.width).copy()
    finally:
        buf.unmap(info)
