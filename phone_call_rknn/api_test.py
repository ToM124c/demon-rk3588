"""接口自测：上传视频到 MinIO 并跑一遍 NDJSON 流。

    python -m phone_call_rknn.api_test --video test.mp4 --mode driver --http
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .app.api_server import stream_task
from .common.constants import MODE_DRIVER, MODE_ESCORT
from .common.storage import upload_video_to_minio
from .driver.config import DEFAULT_SOURCE


@dataclass
class CliArgs:
    video: str
    task_id: str
    mode: str
    upload_only: bool
    skip_upload: bool
    http: bool
    base_url: str


def parse_cli_args(argv=None) -> CliArgs:
    parser = argparse.ArgumentParser(description="Upload test video to MinIO and test analyze API.")
    parser.add_argument("--video", default=DEFAULT_SOURCE)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--mode", choices=(MODE_DRIVER, MODE_ESCORT), default=MODE_DRIVER)
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    video_path = Path(args.video)
    return CliArgs(
        video=str(video_path),
        task_id=args.task_id or video_path.name,
        mode=args.mode,
        upload_only=args.upload_only,
        skip_upload=args.skip_upload,
        http=args.http,
        base_url=args.base_url,
    )


def run_stream_test(task_id: str, mode: str):
    events = violations = 0
    for line in stream_task(task_id, mode):
        events += 1
        if json.loads(line).get("status") == "violation":
            violations += 1
        print(line, end="")
    print(f"Stream test done: events={events}, violations={violations}")
    return events, violations


def run_http_test(base_url: str, task_id: str, mode: str):
    url = f"{base_url.rstrip('/')}/analyze/{quote(task_id)}?mode={quote(mode)}"
    events = violations = 0
    try:
        with urllib.request.urlopen(url, timeout=3600) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                events += 1
                if json.loads(line).get("status") == "violation":
                    violations += 1
                print(line)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    print(f"HTTP test done: events={events}, violations={violations}")
    return events, violations


def main_cli(argv=None) -> int:
    args = parse_cli_args(argv)
    if not args.skip_upload:
        upload_video_to_minio(args.video, args.task_id)
        print(f"Uploaded video: {args.task_id}")
    if args.upload_only:
        return 0
    if args.http:
        run_http_test(args.base_url, args.task_id, args.mode)
    else:
        run_stream_test(args.task_id, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
