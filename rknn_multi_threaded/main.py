import cv2
import time
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rknn_multi_threaded.rknnpool import rknnPoolExecutor
from rknn_multi_threaded.func_RGA import myFunc
from rknn_multi_threaded.vpu_capture import open_vpu_capture
from rknn_multi_threaded.vpu_writer import open_video_writer
from rknn_multi_threaded.pipeline_io import AsyncPipeline

DEFAULT_SOURCE = "/mnt/jl/yolov26-rk3588/video/test_h264_8bit.mp4"
DEFAULT_MODEL = "/mnt/jl/yolov26-rk3588/model-pose-detect/0610-hand-phone.rknn"
DEFAULT_OUTPUT = "/mnt/jl/driver-phone-call-rknn/rknn_multi_threaded/runs/0628-detect/output.mp4"


def read_rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fp:
            for line in fp:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-threaded RKNN inference for 0610-hand-phone.rknn")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="输入视频路径，也可以传摄像头编号，如 0")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="RKNN 模型路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出视频路径")
    parser.add_argument("--threads", type=int, default=3, help="线程数，RK3588 推荐 3")
    parser.add_argument("--no-display", action="store_true", help="不弹窗显示")
    parser.add_argument("--no-encode", action="store_true", help="只推理不编码，用于压测或排查内存")
    parser.add_argument("--codec", choices=("h264", "h265", "hevc"), default="h264", help="VPU 硬解时的输入编码格式")
    parser.add_argument("--decode",choices=("vpu", "cpu"),default="vpu",help="输入解码：vpu 硬解 / cpu 软解（cv2.VideoCapture）")
    parser.add_argument("--decode-fallback", action="store_true", help="VPU 硬解打开失败时回退 cv2.VideoCapture")
    parser.add_argument("--encoder", choices=("vpu", "mp4v"), default="vpu", help="输出编码：vpu 硬编码 / mp4v 软编码")
    parser.add_argument("--encode-fallback", action="store_true", help="VPU 硬编打开失败时回退 mp4v")
    parser.add_argument("--bitrate", type=int, default=None, help="VPU 硬编码码率 bps")
    parser.add_argument("--decode-queue", type=int, default=2, help="解码异步队列深度")
    parser.add_argument("--encode-queue", type=int, default=2, help="编码异步队列深度")
    parser.add_argument("--frame-budget",type=int,default=5,help="全流水线同时在内存中的帧数上限（防 OOM，3.8GB 板子推荐 5）")
    return parser.parse_args()


def prefill_count(threads, frame_budget):
    return max(1, min(threads, frame_budget - 1))


def open_capture(source, codec="h264", decode="vpu", fallback=False):
    if decode == "cpu":
        source_str = str(source)
        return cv2.VideoCapture(int(source_str) if source_str.isdigit() else source_str)
    return open_vpu_capture(source, codec=codec, fallback=fallback)


def new_timing_sums():
    return {
        "decode_ms": 0.0,
        "get_ms": 0.0,
        "pre_ms": 0.0,
        "infer_ms": 0.0,
        "post_ms": 0.0,
        "draw_ms": 0.0,
        "put_ms": 0.0,
        "write_ms": 0.0,
    }


def add_timing_sums(timing_sums, timing, decode_ms, get_ms, put_ms, write_ms):
    timing_sums["decode_ms"] += decode_ms
    timing_sums["get_ms"] += get_ms
    timing_sums["pre_ms"] += timing["pre_ms"]
    timing_sums["infer_ms"] += timing["infer_ms"]
    timing_sums["post_ms"] += timing["post_ms"]
    timing_sums["draw_ms"] += timing["draw_ms"]
    timing_sums["put_ms"] += put_ms
    timing_sums["write_ms"] += write_ms


def print_timing_report(frame_count, elapsed_sec, timing_sums):
    fps = frame_count / elapsed_sec if elapsed_sec > 0 else 0.0
    avg = {key: timing_sums[key] / frame_count for key in timing_sums}
    wall_ms = 1000.0 / fps if fps > 0 else 0.0
    tracked_ms = sum(avg.values())
    print(
        "30帧平均帧率:\t{:.2f} 帧/s, "
        "decode={:.2f}ms, get={:.2f}ms, "
        "pre={:.2f}ms, infer={:.2f}ms, post={:.2f}ms, draw={:.2f}ms, "
        "put={:.2f}ms, write={:.2f}ms "
        "(合计 {:.2f}ms, 帧间隔 {:.2f}ms)".format(
            fps,
            avg["decode_ms"],
            avg["get_ms"],
            avg["pre_ms"],
            avg["infer_ms"],
            avg["post_ms"],
            avg["draw_ms"],
            avg["put_ms"],
            avg["write_ms"],
            tracked_ms,
            wall_ms,
        )
    )


def main():
    args = parse_args()
    cap = open_capture(
        args.source,
        codec=args.codec,
        decode=args.decode,
        fallback=args.decode_fallback,
    )
    if not cap.isOpened():
        raise RuntimeError("Cannot open source: {}".format(args.source))
    print("输入解码:\t", args.decode)
    if getattr(cap, "pipeline", None):
        print("VPU decode pipeline:", cap.pipeline)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = None
    output_path = None
    if not args.no_encode:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = open_video_writer(
            output_path,
            width=width,
            height=height,
            fps=fps,
            encoder=args.encoder,
            bitrate=args.bitrate,
            fallback=args.encode_fallback,
        )
        output_path = Path(getattr(writer, "output_path", output_path))
        if getattr(writer, "pipeline", None):
            print("VPU encode pipeline:", writer.pipeline)
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot create output video: {}".format(output_path))

    pipeline = AsyncPipeline(
        cap,
        writer,
        decode_queue=args.decode_queue,
        encode_queue=args.encode_queue,
        frame_budget=args.frame_budget,
    )
    pipeline.start()
    print("输出编码:\t", "off" if args.no_encode else args.encoder)
    print(
        "异步流水线:\tdecode_queue={}, encode_queue={}, frame_budget={}".format(
            args.decode_queue, args.encode_queue, args.frame_budget
        )
    )

    pool = rknnPoolExecutor(
        rknnModel=args.model,
        TPEs=args.threads,
        func=myFunc,
    )

    pending = 0
    decode_eof = False
    stopped_by_user = False
    for _ in range(prefill_count(args.threads, args.frame_budget)):
        ret, frame, _decode_ms = pipeline.decoder.get()
        if not ret:
            decode_eof = True
            break
        pool.put(frame)
        pending += 1

    frames, loopTime, initTime = 0, time.time(), time.time()
    report_frames = 0
    timing_sums = new_timing_sums()
    encoded_frames = 0
    try:
        while pending > 0:
            decode_ms = 0.0
            if not decode_eof and not stopped_by_user:
                ret, frame, decode_ms = pipeline.decoder.get()
                if ret:
                    pool.put(frame)
                    pending += 1
                else:
                    decode_eof = True

            get_start = time.perf_counter()
            worker_result, flag = pool.get()
            get_ms = (time.perf_counter() - get_start) * 1000.0
            pending -= 1
            if flag is False:
                break
            result, timing = worker_result

            frames += 1
            report_frames += 1

            put_ms = pipeline.encoder.submit(result)
            write_ms = pipeline.encoder.take_write_ms()
            add_timing_sums(timing_sums, timing, decode_ms, get_ms, put_ms, write_ms)

            if not args.no_display:
                cv2.imshow("0610-hand-phone", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by_user = True
                    break

            if frames % 30 == 0:
                print_timing_report(report_frames, time.time() - loopTime, timing_sums)
                print(
                    "内存占用:\t{:.1f} MB, 帧预算使用 {}/{}".format(
                        read_rss_mb(),
                        pipeline.budget.in_use,
                        pipeline.budget.slots,
                    )
                )
                loopTime = time.time()
                report_frames = 0
                timing_sums = new_timing_sums()
    finally:
        for write_ms in pipeline.encoder.drain_write_times():
            timing_sums["write_ms"] += write_ms
        encoded_frames = pipeline.encoder.close()
        pipeline.decoder.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        pool.release()

    total_time = time.time() - initTime
    if frames > 0 and total_time > 0:
        print("总平均帧率\t", frames / total_time)
    print("推理帧数:\t", frames)
    if args.no_encode:
        print("编码:\t\t已跳过 (--no-encode)")
    else:
        print("编码帧数:\t", encoded_frames)
        if frames > 0 and encoded_frames != frames:
            print("警告:\t推理与编码帧数不一致，输出视频可能丢帧或时长异常")
        print("输出视频:\t", output_path)
    if stopped_by_user:
        print("用户按 q 提前停止")


if __name__ == "__main__":
    main()
