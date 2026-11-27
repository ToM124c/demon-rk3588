"""司机模式的服务层：MinIO 取流 → 零拷贝引擎 → NDJSON 事件。"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Iterator, Optional

from ..common.config import env_int
from ..common.constants import MODE_DRIVER
from ..common.events import base_event, completion_events, ndjson_event, violation_period_event
from ..common.media import probe_video_fps
from ..common.storage import download_video, make_minio_client
from ..common.tracking import ViolationPeriodTracker
from .config import default_engine_config
from .engine import AccelEngine
from .visualize import draw_annotations

# 违规编码（与上游约定一致）
CODING_PHONE_CALL = 1
CODING_NO_SEATBELT = 2

DRIVER_FRAME_STRIDE = env_int("DRIVER_FRAME_STRIDE", 15)
VIOLATION_MIN_FRAMES = env_int("VIOLATION_MIN_FRAMES", 5)

_engine_lock = threading.Lock()
_engine: Optional[AccelEngine] = None


def get_engine() -> AccelEngine:
    """模型只加载一次（进程内单例）。"""
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


def frame_codings_from_record(record) -> list[int]:
    codings = []
    if record["danger"]:
        codings.append(CODING_PHONE_CALL)
    if not record["driver_seatbelt_on"]:
        codings.append(CODING_NO_SEATBELT)
    return codings


def visualize_driver_frame(frame, record):
    wrists = record["wrists"]
    assignments = {index: {"left": None, "right": None} for index in range(len(wrists))}
    return draw_annotations(frame, wrists, assignments, record["hands"], record["phones"],
                            record["seatbelts"], record["danger"])


def analyze_driver_video(task_id: str) -> Iterator[str]:
    client = make_minio_client()
    with tempfile.TemporaryDirectory(prefix="driver_behavior_") as temp_dir:
        work_dir = Path(temp_dir)
        local_video = download_video(client, task_id, work_dir / Path(task_id).name)

        yield ndjson_event(base_event("stage_start", task_id, MODE_DRIVER))
        fps = probe_video_fps(local_video)
        completion_seconds = 0.0
        tracker = ViolationPeriodTracker(
            (CODING_PHONE_CALL, CODING_NO_SEATBELT), client, task_id, work_dir,
            min_frames=VIOLATION_MIN_FRAMES,
        )

        with _engine_lock:
            engine = get_engine()
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


def analyze_video(task_id: str) -> Iterator[str]:
    """兼容旧接口名。"""
    yield from analyze_driver_video(task_id)
