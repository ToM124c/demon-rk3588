"""解码后端只允许两种模式：纯硬件（mppvideodec + dma-buf）/ 纯软件（OpenCV）。

用法：python tests/test_capture_backends.py
"""

from __future__ import annotations

from _util import check, hr, test_video

from phone_call_rknn.driver.config import DETECT_MODEL
from phone_call_rknn.hw.gst import CvCapture, DmaBufCapture, open_capture
from phone_call_rknn.hw.rga import Letterboxer
from phone_call_rknn.hw.rknn import CORE_2, TENSOR_NHWC, TENSOR_UINT8, RknnContext
from phone_call_rknn.pipeline.frames import preprocess_frame


def main() -> int:
    video = str(test_video())
    ok = True

    hr("硬件模式（DECODE=vpu）")
    hw = open_capture(video, codec="h264", cpu=False)
    ok &= check(isinstance(hw, DmaBufCapture), f"返回 DmaBufCapture（实际 {type(hw).__name__}）")
    frame = hw.read()
    assert frame is not None, "硬件模式读不到帧"
    ok &= check(frame.fd is not None, f"帧带 dma-buf fd（fd={frame.fd}）")
    ok &= check(frame.pixel_format == "nv12", f"像素格式来自协商 caps（{frame.pixel_format}）")
    ok &= check(hw.drop_one() is True, "drop_one 可用")

    hr("软件模式（DECODE=cpu）")
    sw = open_capture(video, codec="h264", cpu=True)
    ok &= check(isinstance(sw, CvCapture), f"返回 CvCapture（实际 {type(sw).__name__}）")
    sw_frame = sw.read()
    assert sw_frame is not None, "软件模式读不到帧"
    ok &= check(sw_frame.fd is None, "软件模式帧没有 fd")
    ok &= check(sw_frame.pixel_format == "bgr", f"软件模式给 BGR（{sw_frame.pixel_format}）")

    hr("两种模式都能进 RGA 预处理（同一块 RKNN 输入张量）")
    ctx = RknnContext(DETECT_MODEL, core_mask=CORE_2, name="t")
    try:
        tensor = ctx.bind_owned_input(pass_through=False, dtype=TENSOR_UINT8, fmt=TENSOR_NHWC)
        lb = Letterboxer(640)
        lb.bind_fd(tensor.fd)
        for label, fr in (("fd 帧", frame), ("BGR 帧", sw_frame)):
            _rgb, scale, pad_x, pad_y = preprocess_frame(lb, fr, None)
            ok &= check(0 < scale <= 1 and pad_x >= 0 and pad_y >= 0,
                        f"{label} 预处理成功（scale={scale:.2f}, pad=({pad_x:.0f},{pad_y:.0f})）")
    finally:
        frame.release()
        sw_frame.release()
        hw.release()
        sw.release()
        ctx.release()

    hr("硬件失败时的回退策略")
    fb = open_capture(video, codec="h265", fallback=True)   # h264 流用 h265parse 必然失败
    ok &= check(isinstance(fb, CvCapture), "fallback=True -> 回退软件解码")
    fb.release()
    try:
        open_capture(video, codec="h265", fallback=False)
        ok &= check(False, "fallback=False 应当抛错")
    except RuntimeError as exc:
        ok &= check("hardware decode failed" in str(exc), f"fallback=False -> 抛错（{exc}）")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
