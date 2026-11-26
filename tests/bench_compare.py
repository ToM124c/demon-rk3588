"""重构前后吞吐/耗时对比：新引擎（零拷贝+int8+RGA 单次）vs 旧管线。

用法：python tests/bench_compare.py [--frames 40] [--stride 15]
"""

from __future__ import annotations

import argparse
import statistics
import time

from _util import hr, test_video

from phone_call_rknn.config import EngineConfig
from phone_call_rknn.pipeline.engine import AccelEngine


def run_new(video, frames, stride):
    cfg = EngineConfig(no_save_vis=True, quiet=True)
    rows = []
    engine = AccelEngine(cfg).start()
    t0 = time.time()
    try:
        for result in engine.iter_video(video, source_name="bench", frame_stride=stride):
            rows.append(result.timing["detect"])
            if len(rows) >= frames:
                break
    finally:
        engine.release()
    return rows, time.time() - t0


def run_old(video, frames, stride):
    from legacy.engine_accel_old import AccelConfig, create_thread_pools, iter_accel_video_results

    cfg = AccelConfig(no_save_vis=True, quiet=True)
    detect_pool, pose_pool = create_thread_pools(cfg)
    rows = []
    t0 = time.time()
    try:
        for result in iter_accel_video_results(video, cfg, detect_pool, pose_pool,
                                               source_name="bench", writer=None, frame_stride=stride):
            rows.append(result["timing"]["detect"])
            if len(rows) >= frames:
                break
    finally:
        detect_pool.release()
        pose_pool.release()
    return rows, time.time() - t0


def summarize(label, rows, wall):
    print(f"{label:6s} {len(rows):3d} 帧  {wall:6.2f}s  {len(rows)/wall:6.1f} FPS  "
          f"pre={statistics.mean(r['preprocess_ms'] for r in rows):6.2f}ms  "
          f"infer={statistics.mean(r['rknn_inference_ms'] for r in rows):6.2f}ms  "
          f"post={statistics.mean(r['postprocess_ms'] for r in rows):6.2f}ms  "
          f"total={statistics.mean(r['total_runner_ms'] for r in rows):6.2f}ms")
    return len(rows) / wall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=15)
    args = parser.parse_args()
    video = test_video()

    hr(f"旧管线 vs 新引擎（{args.frames} 帧，stride={args.stride}）")
    old_rows, old_wall = run_old(video, args.frames, args.stride)
    new_rows, new_wall = run_new(video, args.frames, args.stride)
    old_fps = summarize("旧", old_rows, old_wall)
    new_fps = summarize("新", new_rows, new_wall)
    print(f"\n吞吐提升: {new_fps/old_fps:.2f}x | 单帧耗时下降: "
          f"{(1 - statistics.mean(r['total_runner_ms'] for r in new_rows) / statistics.mean(r['total_runner_ms'] for r in old_rows))*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
