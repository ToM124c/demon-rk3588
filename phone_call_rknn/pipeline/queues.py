"""有界队列、帧预算与引用计数：整条流水线的背压都由这里提供。"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional


class BoundedQueue:
    """可关闭的有界队列（线程安全的阻塞背压）。

    关闭后：
    - `put()` 立即返回 False（调用方负责释放资源）；
    - `get()` 在排空后立即返回 None，**不依赖哨兵元素**，避免队列满时哨兵丢失
      导致消费者永久阻塞。
    """

    def __init__(self, maxsize: int = 4):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = threading.Event()

    def put(self, item, timeout: Optional[float] = None) -> bool:
        """阻塞放入；队列已关闭时返回 False。timeout 为 None 表示一直等到能放入。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._closed.is_set():
            try:
                self._q.put(item, timeout=0.05)
                return True
            except queue.Full:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
        return False

    def get(self, timeout: Optional[float] = None):
        """取出元素；关闭且排空后返回 None。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._closed.is_set() and self._q.empty():
                return None
            if deadline is None:
                wait = 0.05
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                wait = min(0.05, remaining)
            try:
                return self._q.get(timeout=wait)
            except queue.Empty:
                continue

    def get_nowait(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed.set()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def qsize(self) -> int:
        return self._q.qsize()


class FrameBudget:
    """限制整条流水线同时持有的帧数（= 同时占用的解码 dma-buf 数量）。"""

    def __init__(self, slots: int = 6):
        if slots < 2:
            raise ValueError("frame budget must be >= 2")
        self.slots = int(slots)
        self._sem = threading.Semaphore(self.slots)
        self._lock = threading.Lock()
        self.in_use = 0

    def acquire(self, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            ok = self._sem.acquire()
        else:
            ok = self._sem.acquire(timeout=timeout)
        if ok:
            with self._lock:
                self.in_use += 1
        return ok

    def release(self) -> None:
        with self._lock:
            self.in_use = max(0, self.in_use - 1)
        self._sem.release()

    @property
    def available(self) -> int:
        with self._lock:
            return max(0, self.slots - self.in_use)


class Lease:
    """一次帧的引用计数：最后一个使用者负责归还预算并释放帧。"""

    __slots__ = ("frame", "budget", "_refs", "_lock", "_on_release", "_frame_released")

    def __init__(self, frame, budget: FrameBudget, on_release: Optional[Callable[[], None]] = None, refs: int = 1):
        self.frame = frame
        self.budget = budget
        self._refs = int(refs)
        self._lock = threading.Lock()
        self._on_release = on_release
        self._frame_released = False

    def retain(self, n: int = 1) -> "Lease":
        with self._lock:
            self._refs += n
        return self

    def release(self) -> None:
        with self._lock:
            self._refs -= 1
            last = self._refs <= 0
        if not last:
            return
        if not self._frame_released:
            self._frame_released = True
            try:
                self.frame.release()
            finally:
                self.budget.release()
        if self._on_release is not None:
            self._on_release()

    @property
    def released(self) -> bool:
        return self._refs <= 0


class AsyncWriter:
    """后台编码线程：appsrc/VideoWriter 写入放在独立线程，避免拖慢推理。"""

    _EOF = object()

    def __init__(self, writer, queue_size: int = 2):
        self.writer = writer
        self._q: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._error: Optional[Exception] = None
        self.frame_count = 0
        self._write_ms_total = 0.0

    def start(self) -> "AsyncWriter":
        self._thread = threading.Thread(target=self._run, name="encode-worker", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is self._EOF:
                break
            t0 = time.perf_counter()
            try:
                self.writer.write(item)
            except Exception as exc:      # pragma: no cover - 编码器异常
                self._error = exc
                break
            self._write_ms_total += (time.perf_counter() - t0) * 1000.0
            self.frame_count += 1
            del item

    def submit(self, frame) -> bool:
        if self._closed or frame is None:
            return False
        self._q.put(frame)
        return True

    @property
    def avg_write_ms(self) -> float:
        return self._write_ms_total / self.frame_count if self.frame_count else 0.0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(self._EOF)
        if self._thread is not None:
            self._thread.join(timeout=120.0)
        if self._error is not None:
            raise RuntimeError(f"writer failed: {self._error}") from self._error
