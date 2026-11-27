"""FastAPI 入口（保持 uvicorn 命令不变）。

    uvicorn phone_call_rknn.api_server:app --host 0.0.0.0 --port 8000

业务逻辑在 `app/service.py`，这里只做 HTTP 层。
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from phone_call_rknn.app.escort import release_escort
from phone_call_rknn.app.service import MODE_DRIVER, release_engine, stream_task

app = FastAPI(title="Driver Behavior Detection API", version="2.0")


@app.on_event("shutdown")
def _on_shutdown() -> None:
    release_engine()
    release_escort()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/analyze/{taskId}")
async def stream_progress(taskId: str, mode: str = MODE_DRIVER):
    """按 NDJSON 流式返回检测事件（stage_start / violation / completed / failed）。"""
    return StreamingResponse(stream_task(taskId, mode), media_type="application/x-ndjson")


if __name__ == "__main__":
    from phone_call_rknn.app.api_test import main_cli

    main_cli()
