"""验证 want_float=0：ctypes int8/fp16 输出 + 自反量化 == RKNNLite 浮点结果。

用法：python tests/test_rknn_quantized.py
"""

from __future__ import annotations

import ctypes

import numpy as np

from _util import check, hr, test_video

from phone_call_rknn.driver.config import DETECT_MODEL, POSE_MODEL
from phone_call_rknn.hw.gst import DmaBufCapture
from phone_call_rknn.hw.rga import Letterboxer
from phone_call_rknn.hw.rknn import CORE_0_1_2, TENSOR_NHWC, TENSOR_UINT8, RknnContext, dequantize


def compare(name, model_path, tensor):
    from rknnlite.api import RKNNLite

    lite = RKNNLite(verbose=False)
    assert lite.load_rknn(str(model_path)) == 0
    assert lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
    ref = lite.inference(inputs=[tensor])
    assert ref is not None, "RKNNLite inference returned None"
    lite.release()

    ctx = RknnContext(model_path, core_mask=CORE_0_1_2, name=name)
    buf = ctx.bind_owned_input(pass_through=False, dtype=TENSOR_UINT8, fmt=TENSOR_NHWC)
    raw = np.ctypeslib.as_array((ctypes.c_uint8 * buf.size).from_address(buf.mem.virt_addr))
    raw[:] = tensor.reshape(-1)
    outs = ctx.run()

    worst = 0.0
    for i, out in enumerate(outs):
        info = ctx.outputs[i]
        cand = dequantize(out, info.zp, info.scale) if out.dtype.kind in "iu" else out.astype(np.float32)
        ref_flat = np.asarray(ref[i]).reshape(-1).astype(np.float32)
        cand = cand.reshape(-1)
        n = min(cand.size, ref_flat.size)
        worst = max(worst, float(np.abs(cand[:n] - ref_flat[:n]).max()))
        print(f"  {name} out[{i}]: dtype={out.dtype} scale={info.scale:.6g} zp={info.zp} max_diff={np.abs(cand[:n]-ref_flat[:n]).max():.6g}")
    ctx.release()
    return worst


def main() -> int:
    video = test_video()
    cap = DmaBufCapture(str(video))
    frame = cap.read()
    cap.release()
    assert frame is not None and frame.fd is not None
    lb = Letterboxer(640)
    rgb, scale, pad_x, pad_y = lb.from_fd(frame.fd, frame.width, frame.height, frame.pixel_format,
                                          frame.wstride, frame.hstride)
    frame.release()
    assert rgb is not None
    tensor = np.ascontiguousarray(rgb[None], dtype=np.uint8)

    hr("detect：int8 输出 + 自反量化")
    worst_detect = compare("detect", DETECT_MODEL, tensor)
    hr("pose：fp16 原生输出（want_float=0 直接拿 fp16）")
    worst_pose = compare("pose", POSE_MODEL, tensor)

    ok = check(worst_detect < 1e-4, f"detect 与 RKNNLite 浮点结果一致（max={worst_detect:g}）")
    ok &= check(worst_pose < 1e-2, f"pose 与 RKNNLite 浮点结果一致（max={worst_pose:g}）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
