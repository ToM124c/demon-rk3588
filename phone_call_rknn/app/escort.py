"""「护送/交警手势」模式：复用外部脚本的推理实现（非零拷贝，历史路径）。

这部分逻辑依赖 `/mnt/jl/yolov26-rk3588/inference-police-escort-rknn-hand-assign.py`
（不在本仓库内），因此单独隔离，默认不参与司机行为主链路。
"""

from __future__ import annotations

import importlib.util as importlib_util
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

from .service import (
    CODING_LEFT_POLICE_ACTION,
    CODING_RIGHT_POLICE_ACTION,
    CODING_SUSPECT_HAND,
    ESCORT_FRAME_STRIDE,
    ESCORT_ROOT,
    ESCORT_SCRIPT,
    MODE_ESCORT,
    VIDEO_BUCKET,
    ViolationPeriodTracker,
    base_event,
    completion_events,
    ensure_bucket,
    escort_codings_from_record,
    make_minio_client,
    ndjson_event,
    probe_video_fps,
    violation_period_event,
)

_escort_lock = threading.Lock()
_escort_module = None
_escort_pose_runner = None
_escort_detect_runner = None


@dataclass
class EscortArgs:
    pose_model: str
    detect_model: str
    imgsz: int = 640
    det_conf: float = 0.3
    person_conf: float = 0.25
    kpt_conf: float = 0.30
    iou: float = 0.45
    hand_max_pixel_dist: float = 180.0
    suspect_wrist_pixel_thres: float = 120.0
    contact_pixel_thres: float = 95.0
    pose_input_layout: str = "nhwc"
    detect_input_layout: str = "nhwc"
    no_save_vis: bool = True
    quiet: bool = True
    log_interval: int = 30


def load_escort_module():
    global _escort_module
    if _escort_module is not None:
        return _escort_module
    if not Path(ESCORT_SCRIPT).is_file():
        raise FileNotFoundError(f"Escort script not found: {ESCORT_SCRIPT}")
    sys.path.insert(0, str(ESCORT_ROOT))
    spec = importlib_util.spec_from_file_location("police_escort_inference", str(ESCORT_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load escort script: {ESCORT_SCRIPT}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _escort_module = module
    return _escort_module


def build_escort_args(module) -> EscortArgs:
    return EscortArgs(
        pose_model=os.getenv("ESCORT_POSE_MODEL", module.POSE_MODEL),
        detect_model=os.getenv("ESCORT_DETECT_MODEL", module.DETECT_MODEL),
        imgsz=int(os.getenv("ESCORT_IMGSZ", "640")),
        det_conf=float(os.getenv("ESCORT_DET_CONF", "0.3")),
        person_conf=float(os.getenv("ESCORT_PERSON_CONF", "0.25")),
        kpt_conf=float(os.getenv("ESCORT_KPT_CONF", "0.30")),
        iou=float(os.getenv("ESCORT_IOU", "0.45")),
        hand_max_pixel_dist=float(os.getenv("ESCORT_HAND_MAX_PIXEL_DIST", "180.0")),
        suspect_wrist_pixel_thres=float(os.getenv("ESCORT_SUSPECT_WRIST_PIXEL_THRES", "120.0")),
        contact_pixel_thres=float(os.getenv("ESCORT_CONTACT_PIXEL_THRES", "95.0")),
        pose_input_layout=os.getenv("ESCORT_POSE_INPUT_LAYOUT", "nhwc"),
        detect_input_layout=os.getenv("ESCORT_DETECT_INPUT_LAYOUT", "nhwc"),
        no_save_vis=True,
        quiet=True,
    )


def get_escort_runners():
    global _escort_pose_runner, _escort_detect_runner
    module = load_escort_module()
    args = build_escort_args(module)
    if _escort_pose_runner is None:
        _escort_pose_runner = module.util.RKNNRunner(args.pose_model, args.imgsz, args.pose_input_layout)
    if _escort_detect_runner is None:
        _escort_detect_runner = module.util.RKNNRunner(args.detect_model, args.imgsz, args.detect_input_layout)
    return module, _escort_pose_runner, _escort_detect_runner, args


def release_escort_runners() -> None:
    global _escort_pose_runner, _escort_detect_runner
    for runner in (_escort_pose_runner, _escort_detect_runner):
        if runner is not None:
            runner.release()
    _escort_pose_runner = None
    _escort_detect_runner = None


def run_escort_analysis(task_id: str) -> Iterator[str]:
    client = make_minio_client()
    with tempfile.TemporaryDirectory(prefix="police_escort_") as temp_dir:
        work_dir = Path(temp_dir)
        local_video = work_dir / Path(task_id).name
        client.fget_object(VIDEO_BUCKET, task_id, str(local_video))

        yield ndjson_event(base_event("stage_start", task_id, MODE_ESCORT))
        fps = probe_video_fps(local_video)
        cap = cv2.VideoCapture(str(local_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {local_video}")

        completion_seconds = 0.0
        candidate_codings = (CODING_SUSPECT_HAND, CODING_LEFT_POLICE_ACTION, CODING_RIGHT_POLICE_ACTION)
        tracker = ViolationPeriodTracker(candidate_codings, client, task_id, work_dir)
        frame_idx = 0
        try:
            with _escort_lock:
                module, pose_runner, detect_runner, args = get_escort_runners()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    video_seconds = (frame_idx - 1) / fps
                    completion_seconds = video_seconds
                    if (frame_idx - 1) % ESCORT_FRAME_STRIDE != 0:
                        continue
                    record, visualization_data, _ = module.run_one_frame(
                        frame, frame_idx, task_id, pose_runner, detect_runner, args, defer_visualization=True
                    )
                    for period in tracker.update(
                        escort_codings_from_record(record),
                        video_seconds,
                        frame_idx,
                        frame,
                        visualize_frame=lambda f, data=visualization_data: module.annotate_frame(
                            f,
                            data["roles"],
                            data["pose_people"],
                            data["hand_dets"],
                            data["frame_result"],
                            show_action_details=False,
                        ),
                    ):
                        yield violation_period_event(task_id, MODE_ESCORT, period)
        finally:
            cap.release()

        for period in tracker.finish():
            yield violation_period_event(task_id, MODE_ESCORT, period)
        yield from completion_events(task_id, MODE_ESCORT, completion_seconds)
