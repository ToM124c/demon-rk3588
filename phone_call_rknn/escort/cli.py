"""护送模式命令行：视频 / 图片集推理，输出 JSONL 与可选可视化视频。

    python -m phone_call_rknn.escort --source test-escort-3.mp4 --save-vis
    python -m phone_call_rknn.escort --source imgs/ --save-vis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from ..common.media import image_files, is_video
from ..hw.gst import open_writer
from .config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    ESCORT_HAND_MODEL,
    ESCORT_POSE_MODEL,
    EscortConfig,
)
from .engine import EscortPipeline


def parse_args(argv=None):
    base = EscortConfig()
    parser = argparse.ArgumentParser(
        prog="escort",
        description="交警护送合规检测：hand 检测 + pose 关键点，RGA 单次 CSC + NPU 零拷贝",
    )
    parser.add_argument("--pose-model", default=str(ESCORT_POSE_MODEL), help="pose RKNN 路径或目录")
    parser.add_argument("--hand-model", "--detect-model", dest="hand_model", default=str(ESCORT_HAND_MODEL),
                        help="hand 检测 RKNN 路径或目录（兼容旧参数名 --detect-model）")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="输入视频、图片或图片目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--imgsz", type=int, default=base.imgsz)
    parser.add_argument("--det-conf", type=float, default=base.det_conf)
    parser.add_argument("--person-conf", type=float, default=base.person_conf)
    parser.add_argument("--kpt-conf", type=float, default=base.kpt_conf)
    parser.add_argument("--iou", type=float, default=base.iou)
    parser.add_argument("--hand-max-pixel-dist", type=float, default=base.hand_max_pixel_dist,
                        help="手检测框中心到手腕的最大像素距离（<=0 关闭该限制）")
    parser.add_argument("--contact-pixel-thres", type=float, default=base.contact_pixel_thres,
                        help="警察手到嫌疑人肘部的最大像素距离")
    parser.add_argument("--cores", default=base.cores, help="worker 绑定的 NPU 核，如 2 或 0,1")
    parser.add_argument("--decode", choices=("vpu", "cpu"), default=base.decode)
    parser.add_argument("--codec", choices=("h264", "h265", "hevc"), default=base.codec)
    parser.add_argument("--frame-stride", type=int, default=base.frame_stride, help="每 N 帧推理一帧")
    parser.add_argument("--no-zero-copy", action="store_true", help="关闭 dma-buf 零拷贝")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化视频/图片")
    parser.add_argument("--quiet", action="store_true", help="不逐帧打印 JSON")
    parser.add_argument("--log-interval", type=int, default=30, help="视频进度打印间隔")
    args = parser.parse_args(argv)

    cfg = replace(
        base,
        pose_model=args.pose_model,
        hand_model=args.hand_model,
        imgsz=args.imgsz,
        det_conf=args.det_conf,
        person_conf=args.person_conf,
        kpt_conf=args.kpt_conf,
        iou=args.iou,
        hand_max_pixel_dist=args.hand_max_pixel_dist,
        contact_pixel_thres=args.contact_pixel_thres,
        cores=args.cores,
        zero_copy=not args.no_zero_copy,
        decode=args.decode,
        codec=args.codec,
        frame_stride=args.frame_stride,
        no_save_vis=not args.save_vis,
        quiet=args.quiet,
    )
    return args, cfg


def write_record(jsonl_file, record, quiet):
    line = json.dumps(record, ensure_ascii=False)
    jsonl_file.write(line + "\n")
    jsonl_file.flush()
    if not quiet:
        print(line)


def _probe_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return width, height, fps


def process_video(pipeline, cfg, source_path, output_dir, jsonl_file, log_interval=30):
    width, height, fps = _probe_video(source_path)
    writer = None
    out_video = output_dir / f"{source_path.stem}_escort.mp4"
    if not cfg.no_save_vis:
        writer = open_writer(out_video, width, height, fps, encoder="vpu", fallback=True)

    total = compliant = 0
    try:
        for result in pipeline.iter_video(source_path, source_name=source_path.name, writer=writer):
            write_record(jsonl_file, result.record, cfg.quiet)
            total += 1
            compliant += int(result.compliant)
            if total == 1 or total % max(log_interval, 1) == 0:
                print(f"frame={result.frame_idx}, label={result.label}, reason={result.record['reason']}")
    finally:
        if writer is not None:
            writer.release()

    if writer is not None:
        print(f"Visualization video saved: {out_video}")
    print(f"Video done: total_frames={total}, compliant={compliant}, non_compliant={total - compliant}")


def process_images(pipeline, cfg, source_path, output_dir, jsonl_file):
    paths = image_files(source_path)
    if not paths:
        raise FileNotFoundError(f"No supported images found: {source_path}")
    vis_dir = output_dir / "vis_images"
    if not cfg.no_save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    compliant = 0
    for frame_idx, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        result = pipeline.run_frame(frame, frame_idx, path.name)
        write_record(jsonl_file, result.record, cfg.quiet)
        compliant += int(result.compliant)
        if not cfg.no_save_vis:
            vis_image = result.vis if result.vis is not None else result.visualize()
            cv2.imwrite(str(vis_dir / f"{path.stem}_escort.jpg"), vis_image)
    print(f"Images done: total={len(paths)}, compliant={compliant}, non_compliant={len(paths) - compliant}")


def main(argv=None) -> int:
    args, cfg = parse_args(argv)
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "frame_results.jsonl"

    pipeline = EscortPipeline(cfg).start()
    try:
        with result_path.open("w", encoding="utf-8") as jsonl_file:
            if is_video(source_path):
                process_video(pipeline, cfg, source_path, output_dir, jsonl_file, args.log_interval)
            else:
                process_images(pipeline, cfg, source_path, output_dir, jsonl_file)
        print(f"Frame results saved: {result_path}")
    finally:
        pipeline.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
