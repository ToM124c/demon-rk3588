"""护送 / 交警手势模式的服务层：MinIO 取流 → 引擎 → NDJSON 事件。"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Iterator, Optional

from ..common.config import env_int
from ..common.constants import MODE_ESCORT
from ..common.events import base_event, completion_events, ndjson_event, violation_period_event
from ..common.media import probe_video_fps
from ..common.storage import download_video, make_minio_client
from ..common.tracking import ViolationPeriodTracker
from .config import (
    CODING_LEFT_POLICE_ACTION,
    CODING_RIGHT_POLICE_ACTION,
    CODING_SUSPECT_HAND,
    default_escort_config,
)
from .engine import EscortPipeline
from .visualize import annotate_frame

ESCORT_FRAME_STRIDE = env_int("ESCORT_FRAME_STRIDE", 15)
VIOLATION_MIN_FRAMES = env_int("VIOLATION_MIN_FRAMES", 5)

_pipeline_lock = threading.Lock()
_pipeline: Optional[EscortPipeline] = None


def get_escort_pipeline() -> EscortPipeline:
    """模型只加载一次（进程内单例）。"""
    global _pipeline
    if _pipeline is None:
        cfg = default_escort_config(
            frame_stride=ESCORT_FRAME_STRIDE,
            no_save_vis=True,
            quiet=True,
        )
        _pipeline = EscortPipeline(cfg).start()
    return _pipeline


def release_escort() -> None:
    global _pipeline
    if _pipeline is not None:
        _pipeline.release()
        _pipeline = None


def escort_codings_from_record(record) -> list[int]:
    codings = []
    if not record["suspect_hand_ok"]:
        codings.append(CODING_SUSPECT_HAND)
    if not record["left_police_action_ok"]:
        codings.append(CODING_LEFT_POLICE_ACTION)
    if not record["right_police_action_ok"]:
        codings.append(CODING_RIGHT_POLICE_ACTION)
    return codings


def _visualize_on(frame, result):
    """在给定 BGR 帧上按该帧的判定结果重绘（供违规截图使用）。"""
    return annotate_frame(
        frame,
        result.analysis["roles"],
        result.analysis["pose_people"],
        result.analysis["hand_dets"],
        result.frame_result,
        show_action_details=False,
    )


def analyze_escort_video(task_id: str) -> Iterator[str]:
    client = make_minio_client()
    with tempfile.TemporaryDirectory(prefix="police_escort_") as temp_dir:
        work_dir = Path(temp_dir)
        local_video = download_video(client, task_id, work_dir / Path(task_id).name)

        yield ndjson_event(base_event("stage_start", task_id, MODE_ESCORT))
        fps = probe_video_fps(local_video)
        completion_seconds = 0.0
        tracker = ViolationPeriodTracker(
            (CODING_SUSPECT_HAND, CODING_LEFT_POLICE_ACTION, CODING_RIGHT_POLICE_ACTION),
            client, task_id, work_dir, min_frames=VIOLATION_MIN_FRAMES,
        )

        with _pipeline_lock:
            pipeline = get_escort_pipeline()
            for result in pipeline.iter_video(local_video, source_name=task_id,
                                              frame_stride=ESCORT_FRAME_STRIDE):
                video_seconds = (result.frame_idx - 1) / fps
                completion_seconds = video_seconds
                for period in tracker.update(
                    escort_codings_from_record(result.record),
                    video_seconds,
                    result.frame_idx,
                    result.bgr(),
                    visualize_frame=lambda frame, res=result: _visualize_on(frame, res),
                ):
                    yield violation_period_event(task_id, MODE_ESCORT, period)

        for period in tracker.finish():
            yield violation_period_event(task_id, MODE_ESCORT, period)
        yield from completion_events(task_id, MODE_ESCORT, completion_seconds)


def analyze_video(task_id: str) -> Iterator[str]:
    """兼容旧接口名。"""
    yield from analyze_escort_video(task_id)
