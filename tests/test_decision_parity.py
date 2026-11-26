"""A/B 一致性：新引擎与旧管线的逐帧判定是否一致（重构不能改变结论）。

用法：python tests/test_decision_parity.py [--frames 40]
"""

from __future__ import annotations

import argparse

from _util import check, hr, test_video

from phone_call_rknn.config import EngineConfig
from phone_call_rknn.pipeline.engine import AccelEngine


def run_new(video, frames, stride):
    cfg = EngineConfig(no_save_vis=True, quiet=True)
    engine = AccelEngine(cfg).start()
    rows = {}
    try:
        for result in engine.iter_video(video, source_name="parity", frame_stride=stride):
            rows[result.frame_idx] = (result.decision, bool(result.danger))
            if len(rows) >= frames:
                break
    finally:
        engine.release()
    return rows


def run_old(video, frames, stride):
    from legacy.engine_accel_old import AccelConfig, create_thread_pools, iter_accel_video_results

    cfg = AccelConfig(no_save_vis=True, quiet=True)
    detect_pool, pose_pool = create_thread_pools(cfg)
    rows = {}
    try:
        for result in iter_accel_video_results(video, cfg, detect_pool, pose_pool,
                                               source_name="parity", writer=None, frame_stride=stride):
            record = result["record"]
            rows[result["frame_idx"]] = (record["decision"], bool(record["danger"]))
            if len(rows) >= frames:
                break
    finally:
        detect_pool.release()
        pose_pool.release()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--min-ratio", type=float, default=0.9)
    args = parser.parse_args()

    video = test_video()
    new_rows = run_new(video, args.frames, args.stride)
    old_rows = run_old(video, args.frames, args.stride)

    hr("逐帧判定一致性")
    common = sorted(set(new_rows) & set(old_rows))
    same = sum(1 for idx in common if new_rows[idx] == old_rows[idx])
    ratio = same / len(common) if common else 0.0
    print(f"共同帧 {len(common)}，完全一致 {same}（{ratio*100:.1f}%）")
    for idx in common:
        if new_rows[idx] != old_rows[idx]:
            print(f"  frame {idx}: new={new_rows[idx]} old={old_rows[idx]}")

    ok = check(len(common) >= min(args.frames, 10), f"共同帧数 {len(common)}")
    ok &= check(ratio >= args.min_ratio, f"判定一致率 {ratio*100:.1f}% >= {args.min_ratio*100:.0f}%")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
