import cv2
from pathlib import Path


class VPUVideoWriter:
    """RK3588 VPU/MPP hardware H.264 encoder with a cv2.VideoWriter-like API.

    Uses OpenCV GStreamer backend and Rockchip mpph264enc.
    Input frames are BGR uint8 images, same as cv2.VideoWriter.
    """

    def __init__(self, output_path, width, height, fps, bitrate=None, fallback=False):
        self.output_path = Path(output_path)
        if self.output_path.suffix.lower() != ".mp4":
            self.output_path = self.output_path.with_suffix(".mp4")

        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.bitrate = bitrate
        self.fallback = fallback
        self.pipeline = self._build_pipeline()
        self.writer = cv2.VideoWriter(
            self.pipeline,
            cv2.CAP_GSTREAMER,
            0,
            self.fps,
            (self.width, self.height),
        )

        if not self.writer.isOpened() and fallback:
            self.pipeline = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height),
            )

    def isOpened(self):
        return self.writer.isOpened()

    def write(self, frame):
        return self.writer.write(frame)

    def release(self):
        self.writer.release()

    def _build_pipeline(self):
        location = str(self.output_path).replace('"', '\\"')
        enc = "mpph264enc"
        if self.bitrate:
            enc = "mpph264enc bps={}".format(int(self.bitrate))
        return (
            "appsrc is-live=false block=true max-bytes=0 do-timestamp=true ! "
            "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
            "video/x-raw,format=BGR ! videoconvert ! "
            "video/x-raw,format=NV12 ! {enc} ! h264parse ! "
            "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
            "mp4mux fragment-duration=1000000000 fragment-mode=first-moov-then-finalise ! "
            'filesink location="{location}"'
        ).format(enc=enc, location=location)


MPPVideoWriter = VPUVideoWriter


class MP4VVideoWriter:
    """Software mp4v encoder with the same API as VPUVideoWriter."""

    def __init__(self, output_path, width, height, fps):
        self.output_path = Path(output_path)
        self.pipeline = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            float(fps),
            (int(width), int(height)),
        )

    def isOpened(self):
        return self.writer.isOpened()

    def write(self, frame):
        return self.writer.write(frame)

    def release(self):
        self.writer.release()


def open_vpu_writer(output_path, width, height, fps, bitrate=None, fallback=False):
    """Create a VPU hardware H.264 video writer."""
    writer = VPUVideoWriter(
        output_path,
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        fallback=fallback,
    )
    if not writer.isOpened():
        raise RuntimeError(
            "Failed to open VPU video writer. Check OpenCV GStreamer support "
            "and Rockchip mpph264enc plugin.\nPipeline: {}".format(writer.pipeline)
        )
    return writer


def open_video_writer(output_path, width, height, fps, encoder="vpu", bitrate=None, fallback=False):
    encoder = encoder.lower()
    if encoder in {"vpu", "mpp"}:
        return open_vpu_writer(
            output_path,
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            fallback=fallback,
        )

    if encoder == "mp4v":
        writer = MP4VVideoWriter(output_path, width, height, fps)
        if not writer.isOpened():
            raise RuntimeError("Failed to open mp4v video writer: {}".format(writer.output_path))
        return writer

    raise ValueError("encoder must be one of: vpu, mpp, mp4v")
