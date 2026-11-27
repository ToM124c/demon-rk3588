"""护送模式端到端：零拷贝管线跑通、记录结构正确、可选与旧脚本逐帧比对。

    python tests/test_escort_pipeline.py                 # 只跑新管线
    python tests/test_escort_pipeline.py --parity        # 额外调用外部旧脚本做 A/B
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _util import check, escort_video, hr

from phone_call_rknn.escort.config import EscortConfig, LABEL_COMPLIANT, LABEL_NON_COMPLIANT
from phone_call_rknn.escort.engine import EscortPipeline

LEGACY_SCRIPT = os.environ.get(
    "ESCORT_LEGACY_SCRIPT", "/mnt/jl/yolov26-rk3588/inference-police-escort-rknn-hand-assign.py"
)


def run_new(video, frames, stride):
    cfg = EscortConfig(no_save_vis=True, quiet=True)
    rows = []
    pipeline = EscortPipeline(cfg).start()
    t0 = time.time()
    try:
        for result in pipeline.iter_video(video, source_name="test", frame_stride=stride):
            assert result.record["frame_index"] == result.frame_idx
            assert result.label in (LABEL_COMPLIANT, LABEL_NON_COMPLIANT)
            vis = result.visualize()          # 触发一次绘制（也是违规取证路径）
            assert vis is not None and vis.shape[2] == 3
            rows.append(result)
            if len(rows) >= frames:
                break
    finally:
        pipeline.release()
    return rows, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--parity", action="store_true", help="与外部旧脚本逐帧比对")
    args = parser.parse_args()

    video = escort_video()
    rows, wall = run_new(video, args.frames, args.stride)

    hr("护送零拷贝管线")
    detect_infer = statistics.mean(r.record["timing"]["detect"]["rknn_inference_ms"] for r in rows)
    pose_infer = statistics.mean(r.record["timing"]["pose"]["rknn_inference_ms"] for r in rows)
    pre = statistics.mean(r.record["timing"]["total"]["shared_preprocess_ms"] for r in rows)
    total = statistics.mean(r.record["timing"]["total"]["total_runner_ms"] for r in rows)
    print(f"frames={len(rows)} wall={wall:.2f}s | shared_pre={pre:.2f}ms "
          f"hand_infer={detect_infer:.2f}ms pose_infer={pose_infer:.2f}ms total={total:.1f}ms")
    labels = {}
    for r in rows:
        labels[r.label] = labels.get(r.label, 0) + 1
    print("labels:", labels)

    ok = check(len(rows) >= min(args.frames, 3), f"产出 {len(rows)} 帧结果")
    ok &= check(pre < 6.0, f"RGA 单次预处理 {pre:.2f}ms < 6ms")
    ok &= check(total < 400.0, f"单帧 total {total:.1f}ms < 400ms")
    ok &= check(all("roles" in r.record and "hands" in r.record for r in rows), "记录含 roles/hands")

    if args.parity:
        legacy = Path(LEGACY_SCRIPT)
        if not legacy.is_file():
            print(f"[SKIP] 旧脚本不存在: {legacy}")
            return 0 if ok else 1
        with tempfile.TemporaryDirectory(prefix="escort_parity_") as tmp:
            out = Path(tmp) / "old"
            out.mkdir()
            cmd = [sys.executable, str(legacy), "--source", str(video), "--output-dir", str(out),
                   "--no-save-vis", "--quiet"]
            subprocess.run(cmd, cwd=str(legacy.parent), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            old = {}
            for line in (out / "frame_results.jsonl").open():
                r = json.loads(line)
                old[r["frame_index"]] = r
            same = sum(1 for r in rows if r.frame_idx in old
                       and r.label == old[r.frame_idx]["label"]
                       and r.record["reason"] == old[r.frame_idx]["reason"])
            common = sum(1 for r in rows if r.frame_idx in old)
            ratio = same / common if common else 0.0
            hr("与旧脚本逐帧比对")
            print(f"common={common} identical={same} ({ratio*100:.1f}%)")
            ok &= check(ratio >= 0.9, f"判定一致率 {ratio*100:.1f}% >= 90%")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
