"""加速链路命令行入口。

    python -m phone_call_rknn.main_accel --source video.mp4 --save-vis
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # python main_accel.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from phone_call_rknn.app.cli import parse_engine_args
from phone_call_rknn.core.records import write_record
from phone_call_rknn.core.timing import add_timing_sums, new_timing_sums, print_overall_speed
from phone_call_rknn.hw.gst import open_writer
from phone_call_rknn.pipeline.engine import AccelEngine


def probe_source(source, codec: str):
    """取分辨率/帧率（用 OpenCV 探测，不参与推理）。"""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        return 0, 0, 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return width, height, fps


def main(argv=None) -> int:
    args, cfg = parse_engine_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "frame_results.jsonl"
    source_name = Path(str(args.source)).name if not str(args.source).isdigit() else f"camera{args.source}"

    width, height, fps = probe_source(args.source, cfg.codec)
    writer = None
    if not cfg.no_save_vis and width > 0 and height > 0:
        writer = open_writer(output_dir / f"{Path(str(args.source)).stem}_accel.mp4", width, height, fps,
                             encoder=cfg.encoder, bitrate=cfg.bitrate, fallback=True)

    timing_sums = new_timing_sums()
    danger_count = processed = 0
    start = time.time()
    engine = AccelEngine(cfg).start()
    try:
        with result_path.open("w", encoding="utf-8") as jsonl_file:
            for result in engine.iter_video(args.source, source_name=source_name, writer=writer,
                                            frame_stride=cfg.frame_stride):
                write_record(jsonl_file, result.record, cfg.quiet)
                add_timing_sums(timing_sums, result.timing)
                danger_count += int(result.danger)
                processed += 1
                if cfg.display and result.vis is not None:
                    cv2.imshow("driver-phone-call-accel", result.vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        engine.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"Frame results saved: {result_path}")
    print(f"Video done: total_frames={processed}, danger_frames={danger_count}")
    if processed and elapsed > 0:
        print(f"Wall speed: {processed / elapsed:.2f} FPS")
    print_overall_speed("Video", processed, timing_sums)
    if writer is not None:
        print(f"Visualization video saved: {writer.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
