"""验证护送模式事件流：假 MinIO + 真实推理，检查 NDJSON 事件结构。

用法：python tests/test_escort_service.py
"""

from __future__ import annotations

import json

from _util import FakeMinio, check, escort_video, hr

from phone_call_rknn.driver import service as driver_service
from phone_call_rknn.escort import service as escort_service
from phone_call_rknn.escort.config import default_escort_config
from phone_call_rknn.escort.engine import EscortPipeline


def main() -> int:
    video = escort_video()
    fake = FakeMinio(video)

    # 单例管线 + 假 MinIO（避免连真实对象存储）
    driver_service.make_minio_client = lambda: fake
    escort_service.make_minio_client = lambda: fake
    escort_service._pipeline = EscortPipeline(
        default_escort_config(frame_stride=60, no_save_vis=True, quiet=True)
    ).start()
    escort_service.ESCORT_FRAME_STRIDE = 60

    events = []
    try:
        for line in escort_service.analyze_escort_video("unit-test.mp4"):
            events.append(json.loads(line))
            if len(events) >= 6:
                break
    finally:
        escort_service.release_escort()

    hr("护送事件流")
    for event in events:
        print({k: event.get(k) for k in ("status", "taskId", "coding", "img_url")})

    ok = check(bool(events), "至少产生一个事件")
    ok &= check(events[0]["status"] == "stage_start", "首事件是 stage_start")
    ok &= check(all(e["taskId"] == "unit-test.mp4" for e in events), "taskId 透传")
    violations = [e for e in events if e["status"] == "violation"]
    if violations:
        coding = violations[0]["coding"]
        ok &= check(isinstance(coding, list) and coding[0] in (3, 4, 5), f"违规 coding 属于护送类型（{coding}）")
        ok &= check(bool(violations[0]["img_url"]), "violation 事件带截图 URL")
        ok &= check(bool(fake.uploaded), "截图已上传")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
