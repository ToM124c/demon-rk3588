"""命令行参数：加速链路（EngineConfig）与简单链路（SimpleArgs）各一套。"""

from __future__ import annotations

import argparse
from dataclasses import replace

from ..config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    DETECT_MODEL,
    PHONE_HAND_IOU_THRES,
    PHONE_HAND_PIXEL_THRES,
    POSE_MODEL,
    WRIST_PHONE_PIXEL_THRES,
    EngineConfig,
)


def parse_engine_args(argv=None):
    """加速链路参数：返回 (args, EngineConfig)。"""
    base = EngineConfig()
    p = argparse.ArgumentParser(
        prog="main_accel",
        description="RK3588 加速链路：VPU 硬解(dma-buf) + RGA 单次 CSC + RKNN 零拷贝/int8",
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
    p.add_argument("--decode", choices=("vpu", "cpu"), default=base.decode, help="vpu 硬解(dma-buf) / cpu 软解")
    p.add_argument("--codec", choices=("h264", "h265", "hevc"), default=base.codec)
    p.add_argument("--encoder", choices=("vpu", "mp4v"), default=base.encoder, help="输出编码")
    p.add_argument("--bitrate", type=int, default=None, help="硬编码码率 bps")
    p.add_argument("--decode-queue", type=int, default=base.decode_queue)
    p.add_argument("--encode-queue", type=int, default=base.encode_queue)
    p.add_argument("--frame-budget", type=int, default=base.frame_budget, help="整条流水线同时持有的帧数")
    p.add_argument("--frame-stride", type=int, default=base.frame_stride, help="每 N 帧推理一帧")
    p.add_argument("--no-zero-copy", action="store_true", help="关闭 dma-buf 零拷贝（回退 numpy 路径）")
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
        zero_copy=not args.no_zero_copy,
        decode=args.decode,
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


def parse_args(argv=None):
    """简单链路（RKNNLite 单线程，逐帧 numpy）参数。"""
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
