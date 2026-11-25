import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from phone_call_rknn.accel_pipeline import (
    AccelConfig,
    create_thread_pools,
    iter_accel_video_results,
    open_video_writer_optional,
    parse_core_ids,
)
from phone_call_rknn.config import DEFAULT_OUTPUT_DIR, DEFAULT_SOURCE
from phone_call_rknn.records import write_record
from phone_call_rknn.timing import add_timing_sums, new_timing_sums, print_overall_speed


DEFAULT_ACCEL_OUTPUT_DIR = str(Path(DEFAULT_OUTPUT_DIR).with_name(Path(DEFAULT_OUTPUT_DIR).name))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hardware accelerated RKNNLite driver phone-call detection with detect + pose models."
    )
    parser.add_argument("--pose-model", default=AccelConfig.pose_model, help="关键点检测 RKNN 模型路径或目录")
    parser.add_argument("--detect-model", default=AccelConfig.detect_model, help="hand/phone/seatbelt RKNN 模型路径或目录")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="输入视频路径、RTSP URL 或摄像头编号")
    parser.add_argument("--output-dir", default=DEFAULT_ACCEL_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--imgsz", type=int, default=AccelConfig.imgsz, help="模型输入尺寸")
    parser.add_argument("--det-conf", type=float, default=AccelConfig.det_conf, help="hand/phone/seatbelt 检测置信度阈值")
    parser.add_argument("--person-conf", type=float, default=AccelConfig.person_conf, help="人体框置信度阈值")
    parser.add_argument("--kpt-conf", type=float, default=AccelConfig.kpt_conf, help="手腕关键点置信度阈值")
    parser.add_argument("--iou", type=float, default=AccelConfig.iou, help="NMS IoU 阈值")
    parser.add_argument("--phone-hand-pixel-thres", type=float, default=AccelConfig.phone_hand_pixel_thres, help="phone 和 hand 中心点像素距离阈值")
    parser.add_argument("--phone-hand-iou-thres", type=float, default=AccelConfig.phone_hand_iou_thres, help="phone 和 hand 检测框交并比阈值")
    parser.add_argument("--wrist-phone-pixel-thres", type=float, default=AccelConfig.wrist_phone_pixel_thres, help="phone 和 wrist 关键点像素距离阈值")
    parser.add_argument("--pose-input-layout", choices=("nhwc", "nchw"), default=AccelConfig.pose_input_layout, help="pose 输入排布")
    parser.add_argument("--detect-input-layout", choices=("nhwc", "nchw"), default=AccelConfig.detect_input_layout, help="detect 输入排布")
    parser.add_argument("--detect-cores", default=AccelConfig.detect_cores, help="detect 推理使用的 NPU core 列表，例如 0,1")
    parser.add_argument("--pose-cores", default=AccelConfig.pose_cores, help="pose 推理使用的 NPU core 列表，例如 2")
    parser.add_argument("--decode", choices=("vpu", "cpu"), default=AccelConfig.decode, help="输入解码：vpu 硬解 / cpu 软解")
    parser.add_argument("--decode-fallback", action="store_true", help="VPU 硬解打开失败时回退 cv2.VideoCapture")
    parser.add_argument("--codec", choices=("h264", "h265", "hevc"), default=AccelConfig.codec, help="VPU 硬解时的输入编码格式")
    parser.add_argument("--encoder", choices=("vpu", "mp4v"), default=AccelConfig.encoder, help="输出编码：vpu 硬编码 / mp4v 软编码")
    parser.add_argument("--encode-fallback", action="store_true", help="VPU 硬编打开失败时回退 mp4v")
    parser.add_argument("--bitrate", type=int, default=None, help="VPU 硬编码码率 bps")
    parser.add_argument("--decode-queue", type=int, default=AccelConfig.decode_queue, help="异步解码队列深度")
    parser.add_argument("--encode-queue", type=int, default=AccelConfig.encode_queue, help="异步编码队列深度")
    parser.add_argument("--frame-budget", type=int, default=AccelConfig.frame_budget, help="全流水线同时持有的帧数上限")
    parser.add_argument("--no-save-vis", action="store_true", help="不保存可视化视频；若未同时指定 --display，则跳过 draw_annotations")
    parser.add_argument("--display", action="store_true", help="实时显示可视化结果（会生成可视化帧）")
    parser.add_argument("--quiet", action="store_true", help="不在标准输出逐帧打印 JSON")
    return parser.parse_args()


def args_to_accel_config(args):
    return AccelConfig(
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
        pose_input_layout=args.pose_input_layout,
        detect_input_layout=args.detect_input_layout,
        detect_cores=args.detect_cores,
        pose_cores=args.pose_cores,
        decode=args.decode,
        decode_fallback=args.decode_fallback,
        codec=args.codec,
        encoder=args.encoder,
        encode_fallback=args.encode_fallback,
        bitrate=args.bitrate,
        decode_queue=args.decode_queue,
        encode_queue=args.encode_queue,
        frame_budget=args.frame_budget,
        no_save_vis=args.no_save_vis,
        display=args.display,
        quiet=args.quiet,
    )


def main():
    args = parse_args()
    accel_args = args_to_accel_config(args)
    parse_core_ids(accel_args.detect_cores)
    parse_core_ids(accel_args.pose_cores)

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "frame_results.jsonl"
    source_name = source_path.name if not str(args.source).isdigit() else f"camera{args.source}"

    import cv2

    cap_probe = cv2.VideoCapture(str(args.source) if args.decode == "cpu" else str(args.source))
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    writer, out_video = open_video_writer_optional(accel_args, output_dir, source_path, fps, width, height)
    if writer is not None and getattr(writer, "pipeline", None):
        print("VPU encode pipeline:", writer.pipeline)

    detect_pool, pose_pool = create_thread_pools(accel_args)
    timing_sums = new_timing_sums()
    danger_count = 0
    processed_count = 0
    start = time.time()

    try:
        with result_path.open("w", encoding="utf-8") as jsonl_file:
            for result in iter_accel_video_results(
                args.source,
                accel_args,
                detect_pool,
                pose_pool,
                source_name=source_name,
                writer=writer,
            ):
                write_record(jsonl_file, result["record"], accel_args.quiet)
                add_timing_sums(timing_sums, result["timing"])
                danger_count += int(result["danger"])
                processed_count += 1
                if accel_args.display and result["vis"] is not None:
                    cv2.imshow("driver-phone-call-accel", result["vis"])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        detect_pool.release()
        pose_pool.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"Frame results saved: {result_path}")
    print(f"Video done: total_frames={processed_count}, danger_frames={danger_count}")
    if processed_count > 0 and elapsed > 0:
        print(f"Wall speed: {processed_count / elapsed:.2f} FPS")
    print_overall_speed("Video", processed_count, timing_sums)
    if writer is not None:
        print(f"Visualization video saved: {out_video}")


if __name__ == "__main__":
    main()
