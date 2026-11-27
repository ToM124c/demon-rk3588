"""HTTP 层：按 `mode` 把请求分发到 driver / escort 两个业务域。

uvicorn 入口在仓库根模块 `phone_call_rknn.api_server`（薄 shim），
保持 `uvicorn phone_call_rknn.api_server:app` 命令不变。
"""

from __future__ import annotations

from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ..common.constants import MODE_DRIVER, MODE_ESCORT, MODES, normalize_mode
from ..common.events import failure_event
from ..driver.service import analyze_driver_video, release_engine
from ..escort.service import analyze_escort_video, release_escort

app = FastAPI(title="Driver Behavior Detection API", version="2.0")


def analyze_video(task_id: str, mode: str = MODE_DRIVER) -> Iterator[str]:
    """按模式选择分析流程（同步分发）。"""
    mode = normalize_mode(mode)
    if mode == MODE_ESCORT:
        yield from analyze_escort_video(task_id)
    else:
        yield from analyze_driver_video(task_id)


def stream_task(task_id: str, mode: str = MODE_DRIVER) -> Iterator[str]:
    """统一异常包装：任何失败都变成一条 `failed` 事件。"""
    try:
        yield from analyze_video(task_id, mode)
    except Exception as exc:  # noqa: BLE001 - 事件流需要把异常也返回给调用方
        safe_mode = mode if mode in MODES else MODE_DRIVER
        yield failure_event(task_id, safe_mode, str(exc))


@app.on_event("shutdown")
def _on_shutdown() -> None:
    release_engine()
    release_escort()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modes": list(MODES)}


@app.get("/analyze/{taskId}")
async def stream_progress(taskId: str, mode: str = MODE_DRIVER):
    """按 NDJSON 流式返回检测事件（stage_start / violation / completed / failed）。"""
    return StreamingResponse(stream_task(taskId, mode), media_type="application/x-ndjson")
