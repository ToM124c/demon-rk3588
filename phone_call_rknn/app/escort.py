"""护送 / 交警手势模式的服务层：MinIO 取流 → 推理 → NDJSON 事件。

推理在 `phone_call_rknn.escort` 包里（仓库内，零拷贝），这里只负责
「取流 → 抽帧 → 违规时段聚合 → 事件」。
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Iterator, Optional

from ..escort import EscortPipeline, default_escort_config
from ..escort.visualize import annotate_frame
from .service import (
    CODING_LEFT_POLICE_ACTION,
    CODING_RIGHT_POLICE_ACTION,
    CODING_SUSPECT_HAND,
    ESCORT_FRAME_STRIDE,
    MODE_ESCORT,
    VIDEO_BUCKET,
    ViolationPeriodTracker,
    base_event,
    completion_events,
    escort_codings_from_record,
    make_minio_client,
    ndjson_event,
    probe_video_fps,
    violation_period_event,
)

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


# 兼容旧名字（api_server 曾用 release_escort_runners）
release_escort_runners = release_escort


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


def run_escort_analysis(task_id: str) -> Iterator[str]:
    client = make_minio_client()
    with tempfile.TemporaryDirectory(prefix="police_escort_") as temp_dir:
        work_dir = Path(temp_dir)
        local_video = work_dir / Path(task_id).name
        client.fget_object(VIDEO_BUCKET, task_id, str(local_video))

        yield ndjson_event(base_event("stage_start", task_id, MODE_ESCORT))
        fps = probe_video_fps(local_video)
        completion_seconds = 0.0
        tracker = ViolationPeriodTracker(
            (CODING_SUSPECT_HAND, CODING_LEFT_POLICE_ACTION, CODING_RIGHT_POLICE_ACTION),
            client, task_id, work_dir,
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
