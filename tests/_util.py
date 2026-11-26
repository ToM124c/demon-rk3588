"""测试公用工具：找测试视频、跳过缺失资源。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phone_call_rknn.config import DEFAULT_SOURCE  # noqa: E402


def test_video() -> Path:
    path = Path(os.environ.get("TEST_VIDEO", str(DEFAULT_SOURCE)))
    if not path.is_file():
        raise SystemExit(f"[SKIP] 测试视频不存在: {path}（用 TEST_VIDEO=/path/to.mp4 指定）")
    return path


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def check(condition: bool, message: str) -> bool:
    print(("PASS " if condition else "FAIL ") + message)
    return bool(condition)
