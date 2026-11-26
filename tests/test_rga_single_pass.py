"""验证 RGA 单次 CSC：一步 letterbox vs 旧的三步实现（cvtcolor + resize + cvtcolor）。

用法：python tests/test_rga_single_pass.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from _util import check, hr, test_video

from phone_call_rknn.hw.gst import DmaBufCapture
from phone_call_rknn.hw.rga import Im2d, Letterboxer


def legacy_letterbox(nv12_bytes, width, height, im2d):
    """旧实现：NV12->BGR -> resize -> BGR->RGB（两次全画幅色彩转换）。"""
    from legacy.rknn_multi_threaded.rga_helper import get_rga_letterbox

    rgb, scale, pad_x, pad_y, _ = get_rga_letterbox(640).letterbox_from_nv12(nv12_bytes, width, height, "nv12")
    return rgb


def main() -> int:
    video = test_video()
    cap = DmaBufCapture(str(video))
    hr(f"RGA 单次 CSC（{cap.width}x{cap.height} -> 640x640 RGB）")

    im2d = Im2d()
    lb = Letterboxer(640, im2d=im2d)
    lb.owned_array()                      # CPU 张量作为对比目标（同一 RGA 通道）

    frames = []
    for _ in range(5):
        frame = cap.read()
        if frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print("FAIL 无法解码测试视频")
        return 1

    from gi.repository import Gst

    diffs, new_ms, old_ms = [], [], []
    for frame in frames:
        ok, info = frame._keepalive.get_buffer().map(Gst.MapFlags.READ)
        nv12 = np.frombuffer(info.data, np.uint8)[: frame.width * frame.height * 3 // 2].reshape(
            frame.height * 3 // 2, frame.width).copy()
        frame._keepalive.get_buffer().unmap(info)

        t0 = time.perf_counter()
        rgb, scale, pad_x, pad_y = lb.from_fd(frame.fd, frame.width, frame.height, frame.pixel_format,
                                              frame.wstride, frame.hstride)
        new_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        ref = legacy_letterbox(nv12, frame.width, frame.height, im2d)
        old_ms.append((time.perf_counter() - t0) * 1000)

        diffs.append(float(np.abs(rgb.astype(np.int16) - ref.astype(np.int16)).mean()))
        frame.release()

    mean_diff = statistics.mean(diffs)
    print(f"new: {statistics.mean(new_ms):.2f} ms/frame | old: {statistics.mean(old_ms):.2f} ms/frame")
    print(f"mean abs diff vs old path: {mean_diff:.3f}（0.2 级别来自缩放插值差异）")
    ok = check(mean_diff < 1.0, f"单次 CSC 结果与旧实现一致（diff={mean_diff:.3f}）")
    ok &= check(statistics.mean(new_ms) < statistics.mean(old_ms),
                f"单次 CSC 更快（{statistics.mean(new_ms):.2f}ms < {statistics.mean(old_ms):.2f}ms）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
