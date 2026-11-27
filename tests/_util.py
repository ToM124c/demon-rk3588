"""测试公用工具：找测试视频、假 MinIO、跳过缺失资源。"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phone_call_rknn.driver.config import DEFAULT_SOURCE  # noqa: E402
from phone_call_rknn.escort.config import DEFAULT_SOURCE as DEFAULT_ESCORT_SOURCE  # noqa: E402


def test_video() -> Path:
    """司机行为测试视频（TEST_VIDEO）。"""
    path = Path(os.environ.get("TEST_VIDEO", str(DEFAULT_SOURCE)))
    if not path.is_file():
        raise SystemExit(f"[SKIP] 测试视频不存在: {path}（用 TEST_VIDEO=/path/to.mp4 指定）")
    return path


def escort_video() -> Path:
    """护送模式测试视频（TEST_ESCORT_VIDEO）。"""
    path = Path(os.environ.get("TEST_ESCORT_VIDEO", str(DEFAULT_ESCORT_SOURCE)))
    if not path.is_file():
        raise SystemExit(f"[SKIP] 护送测试视频不存在: {path}（用 TEST_ESCORT_VIDEO=/path/to.mp4 指定）")
    return path


class FakeMinio:
    """只实现服务用到的几个方法，避免测试依赖真实 MinIO。"""

    def __init__(self, source: Path):
        self.source = Path(source)
        self.uploaded = []

    def fget_object(self, bucket, name, path):
        shutil.copyfile(self.source, path)

    def fput_object(self, bucket, name, path, content_type=None):
        self.uploaded.append((bucket, name))
        return None

    def bucket_exists(self, bucket):
        return True

    def make_bucket(self, bucket):
        return None


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def check(condition: bool, message: str) -> bool:
    print(("PASS " if condition else "FAIL ") + message)
    return bool(condition)
