"""验证事件流：用假 MinIO 跑 analyze_driver_video，检查 NDJSON 事件结构。

不需要真实 MinIO；只验证业务编排与事件格式。

用法：python tests/test_service_events.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from _util import check, hr, test_video

from phone_call_rknn.app import service


class FakeMinio:
    def __init__(self, source: Path):
        self.source = source
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


def main() -> int:
    video = test_video()
    fake = FakeMinio(video)
    service.make_minio_client = lambda: fake           # 注入假 MinIO
    service.DRIVER_FRAME_STRIDE = 15

    events = []
    for line in service.analyze_driver_video("unit-test.mp4"):
        events.append(json.loads(line))
        if len(events) >= 6:
            break

    hr("事件流")
    for event in events:
        print({k: event.get(k) for k in ("status", "taskId", "coding", "img_url")})

    ok = check(bool(events), "至少产生一个事件")
    ok &= check(events[0]["status"] == "stage_start", "首事件是 stage_start")
    ok &= check(all("stage" in e and "stage_name" in e and "timestamp" in e for e in events), "事件字段完整")
    ok &= check(all(e["taskId"] == "unit-test.mp4" for e in events), "taskId 透传")
    violations = [e for e in events if e["status"] == "violation"]
    if violations:
        v = violations[0]
        ok &= check(isinstance(v["coding"], list) and v["img_url"], "violation 事件带 coding/img_url")
        ok &= check(bool(fake.uploaded), "违规截图已上传到图片桶")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
