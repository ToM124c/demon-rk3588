"""uvicorn 入口 shim：`uvicorn phone_call_rknn.api_server:app`（命令保持不变）。

HTTP 层在 `phone_call_rknn/app/api_server.py`，业务在 `driver/` 与 `escort/`。
"""

from .app.api_server import analyze_video, app, stream_task  # noqa: F401

__all__ = ["app", "analyze_video", "stream_task"]

if __name__ == "__main__":
    from .api_test import main_cli

    raise SystemExit(main_cli())
