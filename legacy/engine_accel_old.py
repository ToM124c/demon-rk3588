"""（历史实现，仅供基准对比）旧的加速管线：numpy 帧 + RKNNLite + 三次 RGA。

不要在生产代码里 import；`tests/bench_compare.py` 用它来量化重构收益。
"""

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from legacy.rknn_multi_threaded.pipeline_io import AsyncPipeline
from legacy.rknn_multi_threaded.rga_helper import get_rga_letterbox
from legacy.rknn_multi_threaded.vpu_capture import NV12Frame, open_vpu_capture
from legacy.rknn_multi_threaded.vpu_writer import open_video_writer

from phone_call_rknn.driver.config import (
    DETECT_MODEL,
    DETECT_NAMES,
    PHONE_HAND_IOU_THRES,
    PHONE_HAND_PIXEL_THRES,
    POSE_MODEL,
    WRIST_PHONE_PIXEL_THRES,
)
from phone_call_rknn.common.config import resolve_rknn_path
from phone_call_rknn.vision.postprocess import decode_detect_outputs, decode_pose_outputs
from phone_call_rknn.vision.preprocess import letterbox
from phone_call_rknn.driver.records import make_frame_record
from phone_call_rknn.driver.rules import closest_wrist_phone_match, evaluate_detect_rule, get_wrist_points
from phone_call_rknn.common.timing import round_timing, zero_runner_timing
from phone_call_rknn.driver.visualize import draw_annotations


CORE_MASKS = {
    0: RKNNLite.NPU_CORE_0,
    1: RKNNLite.NPU_CORE_1,
    2: RKNNLite.NPU_CORE_2,
    -1: RKNNLite.NPU_CORE_0_1_2,
}


@dataclass
class AccelConfig:
    pose_model: str = str(POSE_MODEL)
    detect_model: str = str(DETECT_MODEL)
    imgsz: int = 640
    det_conf: float = 0.35
    person_conf: float = 0.25
    kpt_conf: float = 0.30
    iou: float = 0.45
    phone_hand_pixel_thres: float = PHONE_HAND_PIXEL_THRES
    phone_hand_iou_thres: float = PHONE_HAND_IOU_THRES
    wrist_phone_pixel_thres: float = WRIST_PHONE_PIXEL_THRES
    pose_input_layout: str = "nhwc"
    detect_input_layout: str = "nhwc"
    detect_cores: str = "0,1"
    pose_cores: str = "2"
    decode: str = "vpu"
    decode_fallback: bool = True
    codec: str = "h264"
    encoder: str = "vpu"
    encode_fallback: bool = False
    bitrate: Optional[int] = None
    decode_queue: int = 4
    encode_queue: int = 2
    frame_budget: int = 6
    no_save_vis: bool = False
    display: bool = False
    quiet: bool = True


def parse_core_ids(value):
    core_ids = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in {"all", "012", "0_1_2"}:
            core_ids.append(-1)
        else:
            core_ids.append(int(part))
    if not core_ids:
        raise ValueError("core list cannot be empty")
    for core_id in core_ids:
        if core_id not in CORE_MASKS:
            raise ValueError(f"unsupported NPU core id: {core_id}")
    return core_ids


def outputs_are_finite(outputs):
    return outputs is not None and all(np.all(np.isfinite(out)) for out in outputs)


def warmup_rknn(rknn, imgsz, layout, rounds=5):
    shape = (1, imgsz, imgsz, 3) if layout == "nhwc" else (1, 3, imgsz, imgsz)
    dummy = np.zeros(shape, dtype=np.uint8)
    for _ in range(rounds):
        rknn.inference(inputs=[dummy])


def init_rknn(model_path, core_id, imgsz, layout, name):
    rknn = RKNNLite(verbose=False)
    model_path = resolve_rknn_path(model_path)
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"Failed to load {name} RKNN model: {model_path}, ret={ret}")
    ret = rknn.init_runtime(core_mask=CORE_MASKS[core_id])
    if ret != 0:
        raise RuntimeError(f"Failed to init {name} RKNN runtime: {model_path}, core={core_id}, ret={ret}")
    warmup_rknn(rknn, imgsz, layout)
    return rknn


class RKNNThreadPool:
    def __init__(self, model_path, core_ids, imgsz, layout, name, func, args):
        self._rknns = [init_rknn(model_path, core_id, imgsz, layout, name) for core_id in core_ids]
        self._pool = ThreadPoolExecutor(max_workers=len(self._rknns), thread_name_prefix=f"{name}-rknn")
        self._futures = set()
        self._func = func
        self._args = args
        self._num = 0

    @property
    def workers(self):
        return len(self._rknns)

    @property
    def pending(self):
        return len(self._futures)

    def put(self, item):
        rknn = self._rknns[self._num % len(self._rknns)]
        future = self._pool.submit(self._func, rknn, item, self._args)  # pyright: ignore[reportCallIssue]
        self._futures.add(future)
        self._num += 1

    def poll_completed(self, block=False):
        if not self._futures:
            return None, False
        done, _ = wait(self._futures, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        if not done:
            return None, False
        future = done.pop()
        self._futures.discard(future)
        return future.result(), True

    def release(self):
        if self._futures:
            wait(self._futures)
            self._futures.clear()
        self._pool.shutdown(wait=True)
        for rknn in self._rknns:
            rknn.release()


def open_capture(source, codec, decode, fallback):
    source = str(source)
    if decode == "cpu":
        return cv2.VideoCapture(int(source) if source.isdigit() else source)
    return open_vpu_capture(source, codec=codec, fallback=fallback)


def frame_shape(frame):
    if isinstance(frame, NV12Frame):
        return (frame.height, frame.width, 3)
    return frame.shape


def nv12_to_bgr_cpu(frame):
    if frame.pixel_format != "nv12":
        raise RuntimeError("NV12_10LE40 preprocess requires RGA")
    return cv2.cvtColor(frame.data, cv2.COLOR_YUV2BGR_NV12)


def make_tensor(rgb, layout):
    if layout == "nchw":
        rgb = rgb.transpose(2, 0, 1)
    return np.ascontiguousarray(rgb[None], dtype=np.uint8)


def preprocess_bgr_cpu(frame, imgsz, layout, keep_source):
    canvas, scale, pad_x, pad_y = letterbox(frame, imgsz)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = make_tensor(rgb, layout)
    bgr = frame.copy() if keep_source else frame
    return tensor, scale, pad_x, pad_y, bgr


def preprocess_for_rknn(frame, imgsz, layout):
    rga = get_rga_letterbox(size=imgsz)
    if isinstance(frame, NV12Frame):
        if rga is not None:
            try:
                rgb, scale, pad_x, pad_y, bgr = rga.letterbox_from_nv12(
                    frame.data,
                    frame.width,
                    frame.height,
                    frame.pixel_format,
                )
                tensor = make_tensor(rgb, layout)
                return tensor, scale, pad_x, pad_y, bgr.copy()
            except RuntimeError:
                pass
        bgr = nv12_to_bgr_cpu(frame)
        return preprocess_bgr_cpu(bgr, imgsz, layout, keep_source=True)

    if rga is not None:
        try:
            rgb, scale, pad_x, pad_y = rga.letterbox(frame)
            tensor = make_tensor(rgb, layout)
            return tensor, scale, pad_x, pad_y, frame
        except RuntimeError:
            pass
    return preprocess_bgr_cpu(frame, imgsz, layout, keep_source=False)


def can_share_preprocess(args):
    return args.detect_input_layout == args.pose_input_layout


def make_timing(detect_timing, pose_timing, shared_preprocess_ms=0.0):
    total_preprocess_ms = shared_preprocess_ms + detect_timing["preprocess_ms"] + pose_timing["preprocess_ms"]
    total_postprocess_ms = detect_timing["postprocess_ms"] + pose_timing["postprocess_ms"]
    total_rknn_inference_ms = detect_timing["rknn_inference_ms"] + pose_timing["rknn_inference_ms"]
    total_runner_ms = total_preprocess_ms + total_postprocess_ms + total_rknn_inference_ms
    return {
        "pose": pose_timing,
        "detect": detect_timing,
        "total": {
            "shared_preprocess_ms": shared_preprocess_ms,
            "preprocess_ms": total_preprocess_ms,
            "postprocess_ms": total_postprocess_ms,
            "rknn_inference_ms": total_rknn_inference_ms,
            "total_runner_ms": total_runner_ms,
            "runner_fps": 1000.0 / total_runner_ms if total_runner_ms > 0 else 0.0,
        },
    }


def need_visualization(args):
    return (not args.no_save_vis) or args.display


def make_result(
    frame_idx,
    source_name,
    frame,
    persons,
    hand_dets,
    phone_dets,
    seatbelt_dets,
    wrists,
    danger,
    decision,
    matches,
    timing,
    need_vis,
):
    record = make_frame_record(
        frame_idx, source_name, persons, wrists, hand_dets, phone_dets, seatbelt_dets, danger, matches, decision
    )
    record["timing"] = round_timing(timing)
    vis = None
    if need_vis:
        assignments = {i: {"left": None, "right": None} for i in range(len(wrists))}
        vis = draw_annotations(frame, wrists, assignments, hand_dets, phone_dets, seatbelt_dets, danger)
    return {
        "frame_idx": frame_idx,
        "record": record,
        "vis": vis,
        "frame": frame,
        "timing": timing,
        "danger": danger,
    }


def detect_task(rknn, item, args):
    need_vis = need_visualization(args)
    frame_idx, frame, source_name = item
    preprocess_start = time.perf_counter()
    tensor, scale, pad_x, pad_y, bgr = preprocess_for_rknn(frame, args.imgsz, args.detect_input_layout)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
    share_preprocess = can_share_preprocess(args)

    inference_start = time.perf_counter()
    outputs = rknn.inference(inputs=[tensor])
    inference_ms = (time.perf_counter() - inference_start) * 1000.0
    detect_timing = {
        "preprocess_ms": preprocess_ms,
        "postprocess_ms": 0.0,
        "rknn_inference_ms": inference_ms,
        "total_runner_ms": preprocess_ms + inference_ms,
    }

    if not outputs_are_finite(outputs):
        timing = make_timing(detect_timing, zero_runner_timing())
        return make_result(frame_idx, source_name, bgr, [], [], [], [], [], False, "invalid_detect_outputs", [], timing, need_vis)

    postprocess_start = time.perf_counter()
    dets = decode_detect_outputs(outputs, scale, pad_x, pad_y, frame_shape(bgr),
                                 args.det_conf, args.iou, names=DETECT_NAMES)
    hand_dets = [det for det in dets if det["class"] == "hand"]
    phone_dets = [det for det in dets if det["class"] == "phone"]
    seatbelt_dets = [det for det in dets if det["class"] == "seatbelt"]
    danger, matches, decision, need_pose = evaluate_detect_rule(hand_dets, phone_dets, args)
    detect_timing["postprocess_ms"] = (time.perf_counter() - postprocess_start) * 1000.0
    detect_timing["total_runner_ms"] += detect_timing["postprocess_ms"]

    if need_pose:
        shared_preprocess_ms = preprocess_ms if share_preprocess else 0.0
        if share_preprocess:
            detect_timing["preprocess_ms"] = 0.0
            detect_timing["total_runner_ms"] -= preprocess_ms
        payload = {
            "frame_idx": frame_idx,
            "source_name": source_name,
            "frame": bgr,
            "hand_dets": hand_dets,
            "phone_dets": phone_dets,
            "seatbelt_dets": seatbelt_dets,
            "matches": list(matches),
            "detect_timing": detect_timing,
            "shared_preprocess_ms": shared_preprocess_ms,
        }
        if share_preprocess:
            payload["tensor"] = tensor
            payload["scale"] = scale
            payload["pad_x"] = pad_x
            payload["pad_y"] = pad_y
        return {
            "needs_pose": True,
            "payload": payload,
        }

    timing = make_timing(detect_timing, zero_runner_timing())
    return make_result(
        frame_idx, source_name, bgr, [], hand_dets, phone_dets, seatbelt_dets, [], danger, decision, matches, timing, need_vis
    )


def pose_task(rknn, payload, args):
    need_vis = need_visualization(args)
    frame = payload["frame"]
    shared_tensor = payload.get("tensor")
    if shared_tensor is not None:
        tensor = shared_tensor
        scale = payload["scale"]
        pad_x = payload["pad_x"]
        pad_y = payload["pad_y"]
        bgr = frame
        preprocess_ms = 0.0
    else:
        preprocess_start = time.perf_counter()
        tensor, scale, pad_x, pad_y, bgr = preprocess_for_rknn(frame, args.imgsz, args.pose_input_layout)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    inference_start = time.perf_counter()
    outputs = rknn.inference(inputs=[tensor])
    inference_ms = (time.perf_counter() - inference_start) * 1000.0
    pose_timing = {
        "preprocess_ms": preprocess_ms,
        "postprocess_ms": 0.0,
        "rknn_inference_ms": inference_ms,
        "total_runner_ms": preprocess_ms + inference_ms,
    }

    if not outputs_are_finite(outputs):
        timing = make_timing(payload["detect_timing"], pose_timing, payload.get("shared_preprocess_ms", 0.0))
        return make_result(
            payload["frame_idx"],
            payload["source_name"],
            bgr,
            [],
            payload["hand_dets"],
            payload["phone_dets"],
            payload["seatbelt_dets"],
            [],
            False,
            "invalid_pose_outputs",
            payload["matches"],
            timing,
            need_vis,
        )

    postprocess_start = time.perf_counter()
    persons = decode_pose_outputs(outputs, scale, pad_x, pad_y, frame_shape(bgr), args.person_conf, args.iou)
    wrists = get_wrist_points(persons, args.kpt_conf)
    matches = list(payload["matches"])
    pose_match = closest_wrist_phone_match(wrists, payload["phone_dets"])
    danger = False
    decision = "pose_no_valid_wrist"
    if pose_match is not None:
        matches.append(pose_match)
        danger = pose_match["distance_px"] < args.wrist_phone_pixel_thres
        decision = "pose_wrist_phone_hit" if danger else "pose_wrist_phone_miss"
    pose_timing["postprocess_ms"] = (time.perf_counter() - postprocess_start) * 1000.0
    pose_timing["total_runner_ms"] += pose_timing["postprocess_ms"]

    timing = make_timing(payload["detect_timing"], pose_timing, payload.get("shared_preprocess_ms", 0.0))
    return make_result(
        payload["frame_idx"],
        payload["source_name"],
        bgr,
        persons,
        payload["hand_dets"],
        payload["phone_dets"],
        payload["seatbelt_dets"],
        wrists,
        danger,
        decision,
        matches,
        timing,
        need_vis,
    )


def create_thread_pools(args):
    detect_cores = parse_core_ids(args.detect_cores)
    pose_cores = parse_core_ids(args.pose_cores)
    detect_pool = RKNNThreadPool(
        args.detect_model,
        detect_cores,
        args.imgsz,
        args.detect_input_layout,
        "detect",
        detect_task,
        args,
    )
    pose_pool = RKNNThreadPool(
        args.pose_model,
        pose_cores,
        args.imgsz,
        args.pose_input_layout,
        "pose",
        pose_task,
        args,
    )
    return detect_pool, pose_pool


def route_detect_result(detect_result, pose_pool, pipeline, args, state, ready):
    if detect_result.get("needs_pose"):
        pose_pool.put(detect_result["payload"])
        state["pending_pose"] += 1
        return
    release_no_vis_result(detect_result, pipeline, args, state)
    ready[detect_result["frame_idx"]] = detect_result


def route_pose_result(pose_result, pipeline, args, state, ready):
    release_no_vis_result(pose_result, pipeline, args, state)
    ready[pose_result["frame_idx"]] = pose_result


def poll_completed_results(detect_pool, pose_pool, pipeline, args, state, ready):
    got_work = False
    while True:
        detect_result, ok = detect_pool.poll_completed()
        if not ok:
            break
        state["pending_detect"] -= 1
        route_detect_result(detect_result, pose_pool, pipeline, args, state, ready)
        got_work = True

    while True:
        pose_result, ok = pose_pool.poll_completed()
        if not ok:
            break
        state["pending_pose"] -= 1
        route_pose_result(pose_result, pipeline, args, state, ready)
        got_work = True

    return got_work


def wait_for_one_result(detect_pool, pose_pool, pipeline, args, state, ready):
    if detect_pool.pending > 0:
        detect_result, ok = detect_pool.poll_completed(block=True)
        if ok:
            state["pending_detect"] -= 1
            route_detect_result(detect_result, pose_pool, pipeline, args, state, ready)
            return True
    if pose_pool.pending > 0:
        pose_result, ok = pose_pool.poll_completed(block=True)
        if ok:
            state["pending_pose"] -= 1
            route_pose_result(pose_result, pipeline, args, state, ready)
            return True
    return False


def release_skipped_frame(pipeline, frame, state):
    pipeline.encoder.submit(frame)
    pipeline.encoder.take_write_ms()
    state["released_count"] += 1


def try_submit_frame(pipeline, detect_pool, args, source_name, state, ready, frame_stride=1):
    budget_count = state["released_count"] if not need_visualization(args) else state["processed_count"]
    frames_in_pipeline = state["frame_idx"] - budget_count
    if state["decode_eof"] or frames_in_pipeline >= args.frame_budget:
        return
    ret, frame, _decode_ms = pipeline.decoder.get()
    if not ret:
        state["decode_eof"] = True
        return
    state["frame_idx"] += 1
    if frame_stride > 1 and (state["frame_idx"] - 1) % frame_stride != 0:
        release_skipped_frame(pipeline, frame, state)
        ready[state["frame_idx"]] = {"skipped": True}
        return
    detect_pool.put((state["frame_idx"], frame, source_name))
    state["pending_detect"] += 1


def release_result_frame(result, pipeline, state):
    if result.get("frame_released"):
        return
    pipeline.encoder.submit(result.get("vis"))
    pipeline.encoder.take_write_ms()
    result["frame_released"] = True
    state["released_count"] += 1


def release_no_vis_result(result, pipeline, args, state):
    if not need_visualization(args):
        release_result_frame(result, pipeline, state)


def pop_ready_results(ready, pipeline, args, state):
    while state["next_output_idx"] in ready:
        result = ready.pop(state["next_output_idx"])
        state["next_output_idx"] += 1
        if result.get("skipped"):
            continue
        state["processed_count"] += 1
        release_result_frame(result, pipeline, state)
        yield result


def open_video_writer_optional(args, output_dir, source_path, fps, width, height):
    if args.no_save_vis:
        return None, None
    out_video = output_dir / f"{source_path.stem}_phone_call_accel.mp4"
    writer = open_video_writer(
        out_video,
        width=width,
        height=height,
        fps=fps,
        encoder=args.encoder,
        bitrate=args.bitrate,
        fallback=args.encode_fallback,
    )
    return writer, Path(getattr(writer, "output_path", out_video))


def iter_accel_video_results(
    video_path,
    args: AccelConfig,
    detect_pool: RKNNThreadPool,
    pose_pool: RKNNThreadPool,
    source_name: Optional[str] = None,
    writer=None,
    frame_stride: int = 1,
) -> Iterator[dict]:
    video_path = Path(video_path)
    source_name = source_name or video_path.name

    cap = open_capture(video_path, args.codec, args.decode, args.decode_fallback)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    pipeline = AsyncPipeline(
        cap,
        writer,
        decode_queue=args.decode_queue,
        encode_queue=args.encode_queue,
        frame_budget=args.frame_budget,
    )
    state = {
        "frame_idx": 0,
        "next_output_idx": 1,
        "pending_detect": 0,
        "pending_pose": 0,
        "decode_eof": False,
        "processed_count": 0,
        "released_count": 0,
    }
    ready = {}
    frame_stride = max(1, int(frame_stride))

    try:
        pipeline.start()
        for _ in range(max(1, min(detect_pool.workers, args.frame_budget - 1))):
            try_submit_frame(pipeline, detect_pool, args, source_name, state, ready, frame_stride)

        while not state["decode_eof"] or detect_pool.pending > 0 or pose_pool.pending > 0 or ready:
            try_submit_frame(pipeline, detect_pool, args, source_name, state, ready, frame_stride)

            got_work = poll_completed_results(detect_pool, pose_pool, pipeline, args, state, ready)
            yield from pop_ready_results(ready, pipeline, args, state)

            if not got_work and (detect_pool.pending > 0 or pose_pool.pending > 0):
                wait_for_one_result(detect_pool, pose_pool, pipeline, args, state, ready)
                yield from pop_ready_results(ready, pipeline, args, state)

        yield from pop_ready_results(ready, pipeline, args, state)
    finally:
        pipeline.encoder.close()
        pipeline.decoder.stop()
        cap.release()
