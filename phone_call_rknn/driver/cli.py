"""司机模式命令行入口。

    python -m phone_call_rknn.driver                     # 加速链路
    python -m phone_call_rknn.driver --save-vis          # 加速链路 + 可视化视频
    python -m phone_call_rknn.driver --simple            # 参考链路（RKNNLite 单线程）
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from ..common.media import is_video, probe_video_size
from ..common.timing import add_timing_sums, new_timing_sums, print_overall_speed
from ..hw.gst import open_writer
from .config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    DETECT_MODEL,
    PHONE_HAND_IOU_THRES,
    PHONE_HAND_PIXEL_THRES,
    POSE_MODEL,
    WRIST_PHONE_PIXEL_THRES,
    EngineConfig,
)
from .engine import AccelEngine
from .records import write_record

# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------
def parse_engine_args(argv=None):
    """加速链路参数：返回 (args, EngineConfig)。"""
    base = EngineConfig()
    p = argparse.ArgumentParser(
        prog="driver",
        description="RK3588 司机行为检测：VPU 硬解(dma-buf) + RGA 单次 CSC + RKNN 零拷贝/int8",
    )
    p.add_argument("--source", default=DEFAULT_SOURCE, help="视频文件、RTSP URL 或摄像头编号")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    p.add_argument("--pose-model", default=str(POSE_MODEL), help="pose RKNN 模型路径或目录")
    p.add_argument("--detect-model", default=str(DETECT_MODEL), help="hand/phone/seatbelt RKNN 模型路径或目录")
    p.add_argument("--imgsz", type=int, default=base.imgsz, help="模型输入尺寸")
    p.add_argument("--det-conf", type=float, default=base.det_conf)
    p.add_argument("--person-conf", type=float, default=base.person_conf)
    p.add_argument("--kpt-conf", type=float, default=base.kpt_conf)
    p.add_argument("--iou", type=float, default=base.iou)
    p.add_argument("--phone-hand-pixel-thres", type=float, default=PHONE_HAND_PIXEL_THRES)
    p.add_argument("--phone-hand-iou-thres", type=float, default=PHONE_HAND_IOU_THRES)
    p.add_argument("--wrist-phone-pixel-thres", type=float, default=WRIST_PHONE_PIXEL_THRES)
    p.add_argument("--detect-cores", default=base.detect_cores, help="detect 使用的 NPU core，如 0,1")
    p.add_argument("--pose-cores", default=base.pose_cores, help="pose 使用的 NPU core，如 2")
    p.add_argument("--decode", choices=("vpu", "cpu"), default=base.decode,
                   help="vpu 纯硬件解码(dma-buf) / cpu 纯软件解码(OpenCV)")
    p.add_argument("--strict-decode", action="store_true", help="硬件解码失败时不回退软解，直接报错")
    p.add_argument("--codec", choices=("h264", "h265", "hevc"), default=base.codec)
    p.add_argument("--encoder", choices=("vpu", "mp4v"), default=base.encoder, help="输出编码")
    p.add_argument("--bitrate", type=int, default=None, help="硬编码码率 bps")
    p.add_argument("--decode-queue", type=int, default=base.decode_queue)
    p.add_argument("--encode-queue", type=int, default=base.encode_queue)
    p.add_argument("--frame-budget", type=int, default=base.frame_budget, help="整条流水线同时持有的帧数")
    p.add_argument("--frame-stride", type=int, default=base.frame_stride, help="每 N 帧推理一帧")
    p.add_argument("--save-vis", action="store_true", help="保存可视化视频（默认不保存）")
    p.add_argument("--display", action="store_true", help="实时显示（需要 GUI）")
    p.add_argument("--quiet", action="store_true", help="不逐帧打印 JSON")
    args = p.parse_args(argv)

    cfg = replace(
        base,
        pose_model=args.pose_model,
        detect_model=args.detect_model,
        imgsz=args.imgsz,
        det_conf=args.det_conf,
        person_conf=args.person_conf,
        kpt_conf=args.kpt_conf,
        iou=args.iou,
        phone_hand_pixel_thres=args.phone_hand_pixel_thres,
        phone_hand_iou_thres=args.phone_hand_iou_thres,
        wrist_phone_pixel_thres=args.wrist_phone_pixel_thres,
        detect_cores=args.detect_cores,
        pose_cores=args.pose_cores,
        decode=args.decode,
        decode_fallback=not args.strict_decode,
        codec=args.codec,
        encoder=args.encoder,
        bitrate=args.bitrate,
        decode_queue=args.decode_queue,
        encode_queue=args.encode_queue,
        frame_budget=args.frame_budget,
        frame_stride=args.frame_stride,
        no_save_vis=not args.save_vis,
        display=args.display,
        quiet=args.quiet,
    )
    return args, cfg


def parse_simple_args(argv=None):
    """参考链路（RKNNLite 单线程）参数。"""
    p = argparse.ArgumentParser(description="RKNNLite 参考实现（非零拷贝，单线程）")
    p.add_argument("--pose-model", default=str(POSE_MODEL))
    p.add_argument("--detect-model", default=str(DETECT_MODEL))
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--det-conf", type=float, default=0.35)
    p.add_argument("--person-conf", type=float, default=0.25)
    p.add_argument("--kpt-conf", type=float, default=0.30)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--phone-hand-pixel-thres", type=float, default=PHONE_HAND_PIXEL_THRES)
    p.add_argument("--phone-hand-iou-thres", type=float, default=PHONE_HAND_IOU_THRES)
    p.add_argument("--wrist-phone-pixel-thres", type=float, default=WRIST_PHONE_PIXEL_THRES)
    p.add_argument("--pose-input-layout", choices=("nhwc", "nchw"), default="nhwc")
    p.add_argument("--detect-input-layout", choices=("nhwc", "nchw"), default="nhwc")
    p.add_argument("--no-save-vis", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# 加速链路
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    args, cfg = parse_engine_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "frame_results.jsonl"
    source_name = Path(str(args.source)).name if not str(args.source).isdigit() else f"camera{args.source}"

    writer = None
    out_video = output_dir / f"{Path(str(args.source)).stem}_accel.mp4"
    if not cfg.no_save_vis:
        width, height, fps = probe_video_size(args.source)
        writer = open_writer(out_video, width, height, fps,
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
        print(f"Visualization video saved: {out_video}")
    return 0


# --------------------------------------------------------------------------
# 参考链路
# --------------------------------------------------------------------------
def main_simple(argv=None) -> int:
    from .lite_runner import RKNNRunner
    from .simple import process_images, process_video

    args = parse_simple_args(argv)
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


def main_dispatch(argv=None) -> int:
    """`--simple` 走参考链路，否则走加速链路。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--simple" in argv:
        argv.remove("--simple")
        return main_simple(argv)
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main_dispatch())
