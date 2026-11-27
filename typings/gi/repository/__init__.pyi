"""`gi.repository` 的类型桩：GStreamer 绑定没有官方 stubs，这里统一按 Any 处理。"""

from typing import Any

Gst: Any
GstApp: Any
GstVideo: Any
GstAllocators: Any
GLib: Any
GObject: Any

def __getattr__(name: str) -> Any: ...
