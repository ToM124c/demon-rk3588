import argparse
import os
from pathlib import Path

import cv2


DEFAULT_APPSINK = "appsink drop=false sync=false max-buffers=4"


class NV12Frame:
    """Decoded NV12/NV12_10LE40 frame from VPU appsink (no CPU videoconvert)."""

    __slots__ = ("data", "width", "height", "pixel_format")

    def __init__(self, data, width, height, pixel_format="nv12"):
        self.data = data
        self.width = int(width)
        self.height = int(height)
        self.pixel_format = pixel_format

    @property
    def shape(self):
        return (self.height, self.width)


def normalize_video_dims(width, height):
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return width, height
    # OpenCV NV12 appsink often reports packed height (e.g. 1620 for 1080p).
    if height > width and height > width * 0.75:
        height = height * 2 // 3
    return width, height


class VPUVideoCapture:
    """RK3588 VPU/MPP hard-decoding wrapper with a cv2.VideoCapture-like API.

    File/RTSP pipelines output NV12 (or NV12_10LE40 for HEVC) via appsink.
    Color conversion is deferred to RGA in the inference preprocess path.
    """

    def __init__(self, source, codec="h264", width=None, height=None, fps=None, fallback=False):
        self.source = str(source)
        self.codec = codec.lower()
        self.width = width
        self.height = height
        self.fps = fps
        self.fallback = fallback
        self.output_format = "bgr" if self.source.isdigit() else self._default_output_format()
        self.pipeline = self._build_pipeline()
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened() and fallback:
            self.pipeline = None
            self.output_format = "bgr"
            self.cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)

        raw_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap.isOpened() else 0
        raw_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap.isOpened() else 0
        self.frame_width, self.frame_height = normalize_video_dims(raw_w, raw_h)

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return ret, frame
        if self.output_format == "nv12" and frame is not None and frame.ndim == 2:
            return ret, NV12Frame(frame, self.frame_width, self.frame_height, self._pixel_format())
        return ret, frame

    def release(self):
        self.cap.release()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH and self.frame_width > 0:
            return float(self.frame_width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT and self.frame_height > 0:
            return float(self.frame_height)
        return self.cap.get(prop_id)

    def getBackendName(self):
        try:
            return self.cap.getBackendName()
        except cv2.error:
            return "unknown"

    def _default_output_format(self):
        # OpenCV appsink cannot reliably negotiate NV12_10LE40 from HEVC yet.
        if self.codec in {"h265", "hevc"}:
            return "bgr"
        return "nv12"

    def _pixel_format(self):
        if self.codec in {"h265", "hevc"}:
            return "nv12_10le40"
        return "nv12"

    def _build_pipeline(self):
        if self.source.isdigit():
            return self._camera_pipeline(int(self.source))
        if self.source.startswith(("rtsp://", "rtmp://")):
            return self._rtsp_pipeline(self.source)
        return self._file_pipeline(Path(self.source))

    def _decoder(self):
        if self.codec not in {"h264", "h265", "hevc"}:
            raise ValueError("codec must be one of: h264, h265, hevc")
        parser = "h265parse" if self.codec in {"h265", "hevc"} else "h264parse"
        return f"{parser} ! mppvideodec"

    def _nv12_sink(self):
        caps = "video/x-raw,format=NV12"
        return f"{caps} ! {DEFAULT_APPSINK}"

    def _bgr_sink(self):
        caps = "video/x-raw,format=BGR"
        return f"videoconvert ! {caps} ! {DEFAULT_APPSINK}"

    def _decode_sink(self):
        if self.output_format == "nv12":
            return self._nv12_sink()
        return self._bgr_sink()

    def _file_pipeline(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        demux = self._demux_for_suffix(path.suffix.lower())
        location = str(path).replace('"', '\\"')
        return f'filesrc location="{location}" ! {demux} ! {self._decoder()} ! {self._decode_sink()}'

    def _rtsp_pipeline(self, url):
        if self.codec in {"h265", "hevc"}:
            depay = "rtph265depay"
        else:
            depay = "rtph264depay"
        return f'rtspsrc location="{url}" latency=100 ! {depay} ! {self._decoder()} ! {self._decode_sink()}'

    def _camera_pipeline(self, index):
        size_caps = []
        if self.width:
            size_caps.append(f"width={int(self.width)}")
        if self.height:
            size_caps.append(f"height={int(self.height)}")
        if self.fps:
            size_caps.append(f"framerate={int(self.fps)}/1")

        caps = ""
        if size_caps:
            caps = " ! video/x-raw," + ",".join(size_caps)
        return f"v4l2src device=/dev/video{index}{caps} ! {self._bgr_sink()}"

    @staticmethod
    def _demux_for_suffix(suffix):
        if suffix in {".mp4", ".mov", ".m4v"}:
            return "qtdemux"
        if suffix in {".mkv", ".webm"}:
            return "matroskademux"
        if suffix in {".ts", ".mts", ".m2ts"}:
            return "tsdemux"
        if suffix in {".h264", ".264", ".h265", ".265", ".hevc"}:
            return "identity"
        raise ValueError(f"Unsupported video container: {suffix}")


def open_vpu_capture(source, codec="h264", width=None, height=None, fps=None, fallback=False):
    """Create a VPU hard-decoding capture object."""
    cap = VPUVideoCapture(source, codec=codec, width=width, height=height, fps=fps, fallback=fallback)
    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open VPU capture. Check that OpenCV has GStreamer support "
            "and Rockchip mppvideodec is installed.\nPipeline: {}".format(cap.pipeline)
        )
    return cap


def main():
    parser = argparse.ArgumentParser(description="Test RK3588 VPU/MPP hard-decoding capture.")
    parser.add_argument("source", help="Video file, RTSP URL, or camera index")
    parser.add_argument("--codec", choices=("h264", "h265", "hevc"), default="h264", help="Encoded video codec")
    parser.add_argument("--fallback", action="store_true", help="Fallback to normal cv2.VideoCapture if VPU open fails")
    args = parser.parse_args()

    cap = open_vpu_capture(args.source, codec=args.codec, fallback=args.fallback)
    print("Backend:", cap.getBackendName())
    if cap.pipeline:
        print("Pipeline:", cap.pipeline)

    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if isinstance(frame, NV12Frame):
            preview = cv2.cvtColor(frame.data, cv2.COLOR_YUV2BGR_NV12)
        else:
            preview = frame
        cv2.imshow("vpu_capture", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Decoded frames:", frames)


if __name__ == "__main__":
    os.environ.setdefault("GST_DEBUG", "1")
    main()
