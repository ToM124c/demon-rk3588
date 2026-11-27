"""违规时段聚合：连续命中 N 个采样帧才算一次违规，时段结束时落一张截图。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from minio import Minio

from .events import format_video_time
from .storage import save_violation_frame


class ViolationPeriodTracker:
    def __init__(
        self,
        candidate_codings: Iterable[int],
        minio_client: Minio,
        task_id: str,
        work_dir: Path,
        min_frames: int = 5,
    ):
        self.candidate_codings = tuple(candidate_codings)
        self.minio_client = minio_client
        self.task_id = task_id
        self.work_dir = work_dir
        self.min_frames = int(min_frames)
        self.counts = {coding: 0 for coding in self.candidate_codings}
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

    def _close(self, coding) -> Optional[dict]:
        period = None
        if self.counts[coding] >= self.min_frames:
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
