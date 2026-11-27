"""NDJSON 事件：两个模式的事件格式完全一致，只有 coding 含义不同。"""

from __future__ import annotations

import json
import time
from typing import Iterator

from .constants import MODE_DRIVER, STAGE, STAGE_NAME, normalize_mode


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


def base_event(status: str, task_id: str, mode: str = MODE_DRIVER) -> dict:
    normalize_mode(mode)
    return {
        "status": status,
        "stage": STAGE,
        "stage_name": STAGE_NAME,
        "taskId": task_id,
        "timestamp": now_ms(),
    }


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


def completion_events(task_id: str, mode: str, completion_seconds: float) -> Iterator[str]:
    completion_time = format_video_time(completion_seconds)
    completed = base_event("completed", task_id, mode)
    completed["stage_completion_time"] = completion_time
    yield ndjson_event(completed)

    all_completed = base_event("all_completed", task_id, mode)
    all_completed["stage_completion_time"] = completion_time
    yield ndjson_event(all_completed)


def failure_event(task_id: str, mode: str, message: str) -> str:
    event = base_event("failed", task_id, mode)
    event["message"] = message
    return ndjson_event(event)
