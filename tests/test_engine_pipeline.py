"""端到端验证：零拷贝引擎能跑通、结果结构正确、吞吐达标。

用法：python tests/test_engine_pipeline.py [--frames 40] [--stride 15]
"""

from __future__ import annotations

import argparse
import statistics
import time

from _util import check, hr, test_video

from phone_call_rknn.driver.config import EngineConfig
from phone_call_rknn.driver.engine import AccelEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--min-fps", type=float, default=8.0)
    args = parser.parse_args()

    video = test_video()
    cfg = EngineConfig(no_save_vis=True, quiet=True)
    rows = []
    t0 = time.time()
    engine = AccelEngine(cfg).start()
    try:
        for result in engine.iter_video(video, source_name="test", frame_stride=args.stride):
            record = result.record
            assert record["frame_index"] == result.frame_idx
            assert "decision" in record and "danger" in record
            result.bgr()          # 触发一次按需 BGR（取证路径）
            rows.append(result.timing)
            if len(rows) >= args.frames:
                break
    finally:
        engine.release()
    wall = time.time() - t0

    hr("零拷贝引擎")
    fps = len(rows) / wall
    d = [r["detect"] for r in rows]
    print(f"frames={len(rows)} wall={wall:.2f}s -> {fps:.1f} FPS（stride={args.stride}）")
    print(f"preprocess={statistics.mean(x['preprocess_ms'] for x in d):.2f}ms "
          f"infer={statistics.mean(x['rknn_inference_ms'] for x in d):.2f}ms "
          f"postprocess={statistics.mean(x['postprocess_ms'] for x in d):.2f}ms "
          f"total={statistics.mean(x['total_runner_ms'] for x in d):.2f}ms")

    ok = check(len(rows) >= min(args.frames, 10), f"产出 {len(rows)} 帧结果")
    ok &= check(fps >= args.min_fps, f"吞吐 {fps:.1f} FPS >= {args.min_fps} FPS")
    ok &= check(statistics.mean(x["preprocess_ms"] for x in d) < 6.0,
                "RGA 单次预处理 < 6ms/帧")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
