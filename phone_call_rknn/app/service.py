"""业务服务层：MinIO 取流、违规时段判定、NDJSON 事件流。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote, urlparse

import cv2
from minio import Minio

from ..config import EngineConfig, default_engine_config
from ..core.visualize import draw_annotations
from ..pipeline.engine import AccelEngine

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://192.17.110.253:30090")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "root")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "Litong123!")
VIDEO_BUCKET = os.getenv("MINIO_VIDEO_BUCKET", "detect-violating-videos")
IMAGE_BUCKET = os.getenv("MINIO_IMAGE_BUCKET", "detect-violating-images")

STAGE = 1
MODE_DRIVER = "driver"
MODE_ESCORT = "escort"
STAGE_NAME = "driver-behavior_escort_detection"
CODING_PHONE_CALL = 1
CODING_NO_SEATBELT = 2
CODING_SUSPECT_HAND = 3
CODING_LEFT_POLICE_ACTION = 4
CODING_RIGHT_POLICE_ACTION = 5

DRIVER_FRAME_STRIDE = int(os.getenv("DRIVER_FRAME_STRIDE", "15"))
ESCORT_FRAME_STRIDE = int(os.getenv("ESCORT_FRAME_STRIDE", "15"))
VIOLATION_MIN_FRAMES = int(os.getenv("VIOLATION_MIN_FRAMES", "5"))
ESCORT_ROOT = Path(os.getenv("ESCORT_ROOT", "/mnt/jl/yolov26-rk3588"))
ESCORT_SCRIPT = Path(os.getenv("ESCORT_SCRIPT", str(ESCORT_ROOT / "inference-police-escort-rknn-hand-assign.py")))


# --------------------------------------------------------------------------
# MinIO
# --------------------------------------------------------------------------
def make_minio_client() -> Minio:
    parsed = urlparse(MINIO_ENDPOINT)
    if parsed.scheme:
        endpoint, secure = parsed.netloc, parsed.scheme == "https"
    else:
        endpoint, secure = MINIO_ENDPOINT, False
    return Minio(endpoint, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=secure)


def public_object_url(bucket: str, object_name: str) -> str:
    return f"{MINIO_ENDPOINT.rstrip('/')}/{bucket}/{quote(object_name)}"


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def upload_video_to_minio(video_path, task_id: str) -> str:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    client = make_minio_client()
    ensure_bucket(client, VIDEO_BUCKET)
    ensure_bucket(client, IMAGE_BUCKET)
    client.fput_object(VIDEO_BUCKET, task_id, str(path), content_type="video/mp4")
    return task_id


def save_violation_frame(client: Minio, task_id: str, frame_idx: int, frame, work_dir: Path) -> str:
    image_name = f"{Path(task_id).stem}_{frame_idx}.jpg"
    image_path = work_dir / image_name
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to save violation frame: {image_path}")
    object_name = f"{Path(task_id).stem}/{image_name}"
    client.fput_object(IMAGE_BUCKET, object_name, str(image_path), content_type="image/jpeg")
    return public_object_url(IMAGE_BUCKET, object_name)


# --------------------------------------------------------------------------
# 事件
# --------------------------------------------------------------------------
def ndjson_event(payload) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def now_ms() -> int:
    return int(time.time() * 1000)


def format_video_time(seconds) -> str:
    total_millis = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_millis, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def normalize_mode(mode) -> str:
    normalized = (mode or MODE_DRIVER).strip().lower()
    if normalized not in (MODE_DRIVER, MODE_ESCORT):
        raise ValueError(f"Unsupported mode: {mode}")
    return normalized


def base_event(status: str, task_id: str, mode: str = MODE_DRIVER) -> dict:
    normalize_mode(mode)
    return {"status": status, "stage": STAGE, "stage_name": STAGE_NAME, "taskId": task_id, "timestamp": now_ms()}


def violation_period_event(task_id: str, mode: str, period: dict) -> str:
    event = base_event("violation", task_id, mode)
    event.update(
        {
            "coding": [period["coding"]],
            "img_url": period["img_url"],
            "wrong_action_time": "null",
            "wrong_action_duration": [
                {"start_time": period["start_time"], "end_time": period["end_time"]}
            ],
        }
    )
    return ndjson_event(event)


def completion_events(task_id: str, mode: str, completion_seconds: float):
    completion_time = format_video_time(completion_seconds)
    completed = base_event("completed", task_id, mode)
    completed["stage_completion_time"] = completion_time
    yield ndjson_event(completed)
    all_completed = base_event("all_completed", task_id, mode)
    all_completed["stage_completion_time"] = completion_time
    yield ndjson_event(all_completed)


# --------------------------------------------------------------------------
# 违规时段
# --------------------------------------------------------------------------
class ViolationPeriodTracker:
    def __init__(self, candidate_codings, minio_client: Minio, task_id: str, work_dir: Path):
        self.candidate_codings = candidate_codings
        self.minio_client = minio_client
        self.task_id = task_id
        self.work_dir = work_dir
        self.counts = {coding: 0 for coding in candidate_codings}
        self.starts = {}
        self.last_seen = {}
        self.image_urls = {}

    def update(self, frame_codings, video_seconds, frame_idx, frame, visualize_frame=None):
        current = set(frame_codings)
        closed = []
        for coding in self.candidate_codings:
            if coding in current:
                if self.counts[coding] == 0:
                    self.starts[coding] = (video_seconds, frame_idx, frame.copy(), visualize_frame)
                self.counts[coding] += 1
                self.last_seen[coding] = video_seconds
            else:
                period = self._close(coding)
                if period is not None:
                    closed.append(period)
        return closed

    def finish(self):
        closed = []
        for coding in self.candidate_codings:
            period = self._close(coding)
            if period is not None:
                closed.append(period)
        return closed

    def _close(self, coding):
        period = None
        if self.counts[coding] >= VIOLATION_MIN_FRAMES:
            start_seconds, frame_idx, frame, visualize_frame = self.starts[coding]
            end_seconds = self.last_seen[coding]
            img_url = self.image_urls.get(frame_idx)
            if img_url is None:
                if visualize_frame is not None:
                    frame = visualize_frame(frame)
                img_url = save_violation_frame(self.minio_client, self.task_id, frame_idx, frame, self.work_dir)
                self.image_urls[frame_idx] = img_url
            period = {
                "coding": coding,
                "start_time": format_video_time(start_seconds),
                "end_time": format_video_time(end_seconds),
                "img_url": img_url,
            }
        self.counts[coding] = 0
        self.starts.pop(coding, None)
        self.last_seen.pop(coding, None)
        return period


def frame_codings_from_record(record) -> list:
    codings = []
    if record["danger"]:
        codings.append(CODING_PHONE_CALL)
    if not record["driver_seatbelt_on"]:
        codings.append(CODING_NO_SEATBELT)
    return codings


def escort_codings_from_record(record) -> list:
    codings = []
    if not record["suspect_hand_ok"]:
        codings.append(CODING_SUSPECT_HAND)
    if not record["left_police_action_ok"]:
        codings.append(CODING_LEFT_POLICE_ACTION)
    if not record["right_police_action_ok"]:
        codings.append(CODING_RIGHT_POLICE_ACTION)
    return codings


def visualize_driver_frame(frame, record):
    wrists = record["wrists"]
    assignments = {index: {"left": None, "right": None} for index in range(len(wrists))}
    return draw_annotations(frame, wrists, assignments, record["hands"], record["phones"],
                            record["seatbelts"], record["danger"])


def probe_video_fps(video_path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)


# --------------------------------------------------------------------------
# 引擎单例（模型只加载一次）
# --------------------------------------------------------------------------
_engine_lock = threading.Lock()
_engine: Optional[AccelEngine] = None


def get_engine() -> AccelEngine:
    global _engine
    if _engine is None:
        cfg = default_engine_config(frame_stride=DRIVER_FRAME_STRIDE, no_save_vis=True, quiet=True)
        _engine = AccelEngine(cfg).start()
    return _engine


def release_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.release()
        _engine = None


# --------------------------------------------------------------------------
# 分析流程
# --------------------------------------------------------------------------
def analyze_driver_video(task_id: str) -> Iterator[str]:
    engine = get_engine()
    client = make_minio_client()
    with tempfile.TemporaryDirectory(prefix="driver_behavior_") as temp_dir:
        work_dir = Path(temp_dir)
        local_video = work_dir / Path(task_id).name
        client.fget_object(VIDEO_BUCKET, task_id, str(local_video))

        yield ndjson_event(base_event("stage_start", task_id, MODE_DRIVER))
        fps = probe_video_fps(local_video)
        completion_seconds = 0.0
        tracker = ViolationPeriodTracker((CODING_PHONE_CALL, CODING_NO_SEATBELT), client, task_id, work_dir)

        with _engine_lock:
            for result in engine.iter_video(local_video, source_name=task_id,
                                            frame_stride=engine.config.frame_stride):
                record = result.record
                video_seconds = (result.frame_idx - 1) / fps
                completion_seconds = video_seconds
                for period in tracker.update(
                    frame_codings_from_record(record),
                    video_seconds,
                    result.frame_idx,
                    result.bgr(),
                    visualize_frame=lambda frame, rec=record: visualize_driver_frame(frame, rec),
                ):
                    yield violation_period_event(task_id, MODE_DRIVER, period)

        for period in tracker.finish():
            yield violation_period_event(task_id, MODE_DRIVER, period)
        yield from completion_events(task_id, MODE_DRIVER, completion_seconds)


def analyze_escort_video(task_id: str) -> Iterator[str]:
    from .escort import run_escort_analysis

    yield from run_escort_analysis(task_id)


def analyze_video(task_id: str, mode: str = MODE_DRIVER) -> Iterator[str]:
    mode = normalize_mode(mode)
    if mode == MODE_ESCORT:
        yield from analyze_escort_video(task_id)
    else:
        yield from analyze_driver_video(task_id)


def stream_task(task_id: str, mode: str = MODE_DRIVER) -> Iterator[str]:
    try:
        yield from analyze_video(task_id, mode)
    except Exception as exc:  # noqa: BLE001 - 事件流需要把异常也变成一条 failed
        event = base_event("failed", task_id, mode if mode in (MODE_DRIVER, MODE_ESCORT) else MODE_DRIVER)
        event["message"] = str(exc)
        yield ndjson_event(event)
