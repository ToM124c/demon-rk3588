"""参考链路命令行入口（RKNNLite 单线程 + numpy，用于对照/小图集）。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phone_call_rknn.app.cli import parse_args
from phone_call_rknn.core.media import is_video
from phone_call_rknn.pipeline.lite_runner import RKNNRunner
from phone_call_rknn.pipeline.simple import process_images, process_video


def main() -> int:
    args = parse_args()
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "frame_results.jsonl"

    pose_runner = RKNNRunner(args.pose_model, args.imgsz, args.pose_input_layout)
    detect_runner = RKNNRunner(args.detect_model, args.imgsz, args.detect_input_layout)
    try:
        with result_path.open("w", encoding="utf-8") as jsonl_file:
            if is_video(source_path):
                process_video(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file)
            else:
                process_images(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file)
        print(f"Frame results saved: {result_path}")
    finally:
        pose_runner.release()
        detect_runner.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
