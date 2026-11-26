import queue
import threading
import time

_EOF = object()


class FrameBudget:
    """Limit how many video frames may exist in the whole pipeline at once."""

    def __init__(self, slots):
        if slots < 2:
            raise ValueError("frame budget must be at least 2")
        self._slots = int(slots)
        self._sem = threading.Semaphore(self._slots)
        self._lock = threading.Lock()
        self.in_use = 0

    def acquire(self):
        self._sem.acquire()
        with self._lock:
            self.in_use += 1

    def release(self):
        with self._lock:
            self.in_use = max(0, self.in_use - 1)
        self._sem.release()

    @property
    def slots(self):
        return self._slots

    @property
    def available(self):
        # Semaphore has no public getter; in_use is good enough for logging.
        return max(0, self._slots - self.in_use)


class AsyncFrameDecoder:
    """Background thread: cap.read() -> frame queue."""

    def __init__(self, cap, queue_size=2, budget=None):
        self.cap = cap
        self._queue = queue.Queue(maxsize=queue_size)
        self._budget = budget
        self._thread = None
        self._stop = threading.Event()
        self._exc = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="decode-worker", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                if self._budget is not None:
                    self._budget.acquire()
                t0 = time.perf_counter()
                ret, frame = self.cap.read()
                decode_ms = (time.perf_counter() - t0) * 1000.0
                if not ret:
                    if self._budget is not None:
                        self._budget.release()
                    self._queue.put((False, None, decode_ms))
                    break
                self._queue.put((True, frame, decode_ms))
        except Exception as exc:
            self._exc = exc
            self._queue.put((False, None, 0.0))

    def _raise_if_failed(self):
        if self._exc is not None:
            raise self._exc

    def get(self):
        self._raise_if_failed()
        item = self._queue.get()
        self._raise_if_failed()
        return item

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._drain_releases()

    def _drain_releases(self):
        if self._budget is None:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            return
        while True:
            try:
                ret, frame, _decode_ms = self._queue.get_nowait()
            except queue.Empty:
                break
            if ret:
                self._budget.release()
                del frame

#不编码时，释放帧预算
class NullFrameEncoder:
    """Discard inference results without encoding; release frame budget immediately."""

    def __init__(self, budget=None):
        self._budget = budget
        self._closed = False
        self._frame_count = 0

    def start(self):
        pass

    def submit(self, frame):
        if self._closed:
            raise RuntimeError("NullFrameEncoder is closed")
        self._frame_count += 1
        del frame
        if self._budget is not None:
            self._budget.release()
        return 0.0

    def take_write_ms(self):
        return 0.0

    def drain_write_times(self):
        return []

    def close(self):
        self._closed = True
        return self._frame_count

    @property
    def frame_count(self):
        return self._frame_count


class AsyncFrameEncoder:
    """Background thread: frame queue -> writer.write()."""

    def __init__(self, writer, queue_size=2, budget=None):
        self.writer = writer
        self._queue = queue.Queue(maxsize=queue_size)
        self._timing = queue.Queue()
        self._budget = budget
        self._thread = None
        self._exc = None
        self._closed = False
        self._frame_count = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="encode-worker", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                frame = self._queue.get()
                if frame is _EOF:
                    break
                t0 = time.perf_counter()
                self.writer.write(frame)
                self._frame_count += 1
                self._timing.put((time.perf_counter() - t0) * 1000.0)
                del frame
                if self._budget is not None:
                    self._budget.release()
        except Exception as exc:
            self._exc = exc

    def _raise_if_failed(self):
        if self._exc is not None:
            raise self._exc

    def submit(self, frame):
        if self._closed:
            raise RuntimeError("AsyncFrameEncoder is closed")
        self._raise_if_failed()
        t0 = time.perf_counter()
        self._queue.put(frame)
        return (time.perf_counter() - t0) * 1000.0

    def take_write_ms(self):
        try:
            return self._timing.get_nowait()
        except queue.Empty:
            return 0.0

    def drain_write_times(self):
        times = []
        while True:
            try:
                times.append(self._timing.get_nowait())
            except queue.Empty:
                break
        return times

    def close(self):
        if self._closed:
            return self._frame_count
        self._closed = True
        self._queue.put(_EOF)
        if self._thread is not None:
            self._thread.join(timeout=120.0)
        self._drain_releases()
        self._raise_if_failed()
        return self._frame_count

    def _drain_releases(self):
        if self._budget is None:
            return
        while True:
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                break
            if frame is not _EOF:
                del frame
                self._budget.release()

    @property
    def frame_count(self):
        return self._frame_count


class AsyncPipeline:
    def __init__(self, cap, writer=None, decode_queue=2, encode_queue=2, frame_budget=5):
        self.budget = FrameBudget(frame_budget)
        self.decoder = AsyncFrameDecoder(cap, queue_size=decode_queue, budget=self.budget)
        if writer is None:
            self.encoder = NullFrameEncoder(budget=self.budget)
        else:
            self.encoder = AsyncFrameEncoder(writer, queue_size=encode_queue, budget=self.budget)

    def start(self):
        self.decoder.start()
        self.encoder.start()

    def stop(self):
        self.decoder.stop()
        self.encoder.close()
