"""加速推理引擎：硬解 dma-buf -> RGA 单次 CSC -> NPU 零拷贝 -> int8 后处理。

与旧 `accel_pipeline` 的差异：
- 解码侧直接拿 **dma-buf fd**，帧不再落到 numpy（零拷贝起点）
- RGA 一次 `improcess` 完成 NV12->RGB + 缩放 + letterbox 偏移（单次 CSC）
- RKNN 走 `rknn_set_io_mem` 绑 dma-buf，`want_float=0` 拿原始 int8/fp16
- 每个 NPU 核一个 worker 线程，线程内独占 ctx/输入张量/RGA 上下文，无锁
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import cv2
import numpy as np

from ..config import EngineConfig, resolve_rknn_path
from ..core.postprocess import decode_detect_quantized, decode_pose_half
from ..core.records import make_frame_record
from ..core.rules import closest_wrist_phone_match, evaluate_detect_rule, get_wrist_points
from ..core.timing import zero_runner_timing
from ..core.visualize import draw_annotations
from ..hw.frame import Frame, frame_to_bgr
from ..hw.gst import open_capture
from ..hw.rga import Im2d, Letterboxer, RK_FORMAT_BGR_888
from ..hw.rknn import (CORE_MASKS, TENSOR_NHWC, TENSOR_UINT8, RknnContext)
from .queues import AsyncWriter, BoundedQueue, FrameBudget, Lease

_SKIP = object()


# --------------------------------------------------------------------------
# 结果对象
# --------------------------------------------------------------------------
@dataclass
class FrameResult:
    frame_idx: int
    source: str
    record: dict
    timing: dict
    danger: bool
    decision: str
    vis: Optional[np.ndarray] = None
    _lease: Optional[Lease] = field(default=None, repr=False)
    _bgr: Optional[np.ndarray] = field(default=None, repr=False)
    _im2d: Optional[Im2d] = field(default=None, repr=False)

    def bgr(self, im2d: Optional[Im2d] = None) -> np.ndarray:
        """按需把这一帧转成 BGR（可视化/取证）；推理链路本身不产生 BGR。"""
        if self._bgr is None:
            if self._lease is None:
                raise RuntimeError("frame already released")
            self._bgr = frame_to_bgr(self._lease.frame, im2d=im2d or self._im2d or Im2d())
        return self._bgr

    @property
    def frame(self) -> np.ndarray:
        return self.bgr()

    def release(self) -> None:
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.release()

    # 兼容旧的 dict 访问方式
    def __getitem__(self, key):
        if key == "frame":
            return self.bgr()
        return getattr(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default


# --------------------------------------------------------------------------
# 任务
# --------------------------------------------------------------------------
@dataclass
class _Job:
    frame_idx: int
    source: str
    lease: Lease


@dataclass
class _PoseJob(_Job):
    hand_dets: list
    phone_dets: list
    seatbelt_dets: list
    matches: list
    detect_timing: dict


def parse_cores(value, default=(0,)) -> List[int]:
    cores = []
    for part in str(value).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in {"all", "012", "0_1_2"}:
            cores.append(-1)
        else:
            core = int(part)
            if core not in (0, 1, 2):
                raise ValueError(f"unsupported NPU core id: {core}")
            cores.append(core)
    return cores or list(default)


# --------------------------------------------------------------------------
# 单核 worker
# --------------------------------------------------------------------------
class _Worker(threading.Thread):
    def __init__(self, engine: "AccelEngine", name: str, model_path, core_id: int,
                 jobs: BoundedQueue, handler, imgsz: int, warmup_rounds: int):
        super().__init__(name=name, daemon=True)
        self.engine = engine
        self.model_path = model_path
        self.core_id = core_id
        self.jobs = jobs
        self.handler = handler
        self.imgsz = imgsz
        self.warmup_rounds = warmup_rounds
        self.ctx: Optional[RknnContext] = None
        self.tensor = None
        self.letterboxer: Optional[Letterboxer] = None
        self.im2d: Optional[Im2d] = None
        self.ready = threading.Event()

    # ---- 生命周期 -------------------------------------------------
    def run(self):
        try:
            mask = CORE_MASKS[self.core_id]
            self.ctx = RknnContext(self.model_path, core_mask=mask, name=self.name)
            self.tensor = self.ctx.bind_owned_input(pass_through=False, dtype=TENSOR_UINT8, fmt=TENSOR_NHWC)
            self.im2d = Im2d()
            self.letterboxer = Letterboxer(self.imgsz, im2d=self.im2d)
            self.letterboxer.bind_fd(self.tensor.fd)
            for _ in range(max(1, int(self.warmup_rounds))):
                self.ctx.run()
            self.ready.set()
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                try:
                    self.handler(self, job)
                except Exception as exc:      # pragma: no cover - 运行期异常
                    self.engine._fail(exc)
                    return
        except Exception as exc:              # pragma: no cover - 初始化异常
            self.engine._fail(exc)
            self.ready.set()
        finally:
            self.ready.set()
            if self.ctx is not None:
                self.ctx.release()
                self.ctx = None


def preprocess_frame(worker: _Worker, frame: Frame):
    """把一帧送进 worker 专属的 RKNN 输入 dma-buf（fd 优先，零拷贝）。"""
    lb = worker.letterboxer
    if frame.fd is not None:
        try:
            return lb.from_fd(frame.fd, frame.width, frame.height, frame.pixel_format, frame.wstride, frame.hstride)
        except RuntimeError:
            # 10bit 等 RGA 不支持的格式：退回 CPU 转 BGR 再走 RGA
            bgr = frame_to_bgr(frame, im2d=worker.im2d)
            src = worker.im2d.wrap_va(bgr, frame.width, frame.height, frame.width, frame.height, RK_FORMAT_BGR_888)
            try:
                return lb.from_buffer(src, frame.width, frame.height)
            finally:
                worker.im2d.release(src)
    if frame.va is not None:
        if frame.va.ndim == 3:
            src = worker.im2d.wrap_va(frame.va, frame.width, frame.height, frame.width, frame.height,
                                      RK_FORMAT_BGR_888)
            try:
                return lb.from_buffer(src, frame.width, frame.height)
            finally:
                worker.im2d.release(src)
        return lb.from_nv12(frame.va, frame.width, frame.height, frame.pixel_format)
    raise RuntimeError("frame has neither dma-buf fd nor CPU buffer")


def _finalize(engine: "AccelEngine", worker: _Worker, lease: Lease, frame_idx: int, source: str,
              frame: Frame, persons, wrists, hand_dets, phone_dets, seatbelt_dets,
              danger, decision, matches, timing):
    cfg = engine.config
    record = make_frame_record(frame_idx, source, persons, wrists, hand_dets, phone_dets,
                               seatbelt_dets, danger, matches, decision)
    record["timing"] = _round_timing(timing)

    vis = None
    if not cfg.no_save_vis or cfg.display:
        bgr = frame_to_bgr(frame, im2d=worker.im2d)
        assignments = {i: {"left": None, "right": None} for i in range(len(wrists))}
        vis = draw_annotations(bgr, wrists, assignments, hand_dets, phone_dets, seatbelt_dets, danger)
        if engine._writer is not None:
            engine._writer.submit(vis)
            if not cfg.display:
                vis = None

    result = FrameResult(frame_idx, source, record, timing, bool(danger), decision, vis=vis)
    engine._emit(result, lease)


def _round_timing(timing):
    return {name: {k: round(float(v), 3) for k, v in values.items()} for name, values in timing.items()}


def _make_timing(detect_timing, pose_timing, shared_preprocess_ms=0.0):
    total_pre = shared_preprocess_ms + detect_timing["preprocess_ms"] + pose_timing["preprocess_ms"]
    total_post = detect_timing["postprocess_ms"] + pose_timing["postprocess_ms"]
    total_infer = detect_timing["rknn_inference_ms"] + pose_timing["rknn_inference_ms"]
    total = total_pre + total_post + total_infer
    return {
        "pose": pose_timing,
        "detect": detect_timing,
        "total": {
            "shared_preprocess_ms": shared_preprocess_ms,
            "preprocess_ms": total_pre,
            "postprocess_ms": total_post,
            "rknn_inference_ms": total_infer,
            "total_runner_ms": total,
            "runner_fps": 1000.0 / total if total > 0 else 0.0,
        },
    }


def _detect_handler(worker: _Worker, job: _Job):
    engine = worker.engine
    cfg = engine.config
    frame = job.lease.frame

    t0 = time.perf_counter()
    _rgb, scale, pad_x, pad_y = preprocess_frame(worker, frame)
    t1 = time.perf_counter()
    outputs = worker.ctx.run()
    t2 = time.perf_counter()
    dets = decode_detect_quantized(
        outputs, worker.ctx.outputs, scale, pad_x, pad_y,
        (frame.height, frame.width, 3), cfg.det_conf, cfg.iou,
    )
    t3 = time.perf_counter()

    hand_dets = [d for d in dets if d["class"] == "hand"]
    phone_dets = [d for d in dets if d["class"] == "phone"]
    seatbelt_dets = [d for d in dets if d["class"] == "seatbelt"]
    danger, matches, decision, need_pose = evaluate_detect_rule(hand_dets, phone_dets, cfg)

    detect_timing = {
        "preprocess_ms": (t1 - t0) * 1000.0,
        "rknn_inference_ms": (t2 - t1) * 1000.0,
        "postprocess_ms": (t3 - t2) * 1000.0,
        "total_runner_ms": (t3 - t0) * 1000.0,
    }

    if need_pose:
        engine._submit_pose(_PoseJob(
            frame_idx=job.frame_idx,
            source=job.source,
            lease=job.lease.retain(),
            hand_dets=hand_dets,
            phone_dets=phone_dets,
            seatbelt_dets=seatbelt_dets,
            matches=list(matches),
            detect_timing=detect_timing,
        ))
        job.lease.release()
        return

    timing = _make_timing(detect_timing, zero_runner_timing())
    _finalize(engine, worker, job.lease, job.frame_idx, job.source, frame, [], [],
              hand_dets, phone_dets, seatbelt_dets, danger, decision, matches, timing)
    job.lease.release()


def _pose_handler(worker: _Worker, job: _PoseJob):
    engine = worker.engine
    cfg = engine.config
    frame = job.lease.frame

    t0 = time.perf_counter()
    _rgb, scale, pad_x, pad_y = preprocess_frame(worker, frame)
    t1 = time.perf_counter()
    outputs = worker.ctx.run()
    t2 = time.perf_counter()
    persons = decode_pose_half(
        outputs[0], scale, pad_x, pad_y,
        (frame.height, frame.width, 3), cfg.person_conf, cfg.iou,
    )
    wrists = get_wrist_points(persons, cfg.kpt_conf)
    matches = list(job.matches)
    pose_match = closest_wrist_phone_match(wrists, job.phone_dets)
    danger = False
    decision = "pose_no_valid_wrist"
    if pose_match is not None:
        matches.append(pose_match)
        danger = pose_match["distance_px"] < cfg.wrist_phone_pixel_thres
        decision = "pose_wrist_phone_hit" if danger else "pose_wrist_phone_miss"
    t3 = time.perf_counter()

    pose_timing = {
        "preprocess_ms": (t1 - t0) * 1000.0,
        "rknn_inference_ms": (t2 - t1) * 1000.0,
        "postprocess_ms": (t3 - t2) * 1000.0,
        "total_runner_ms": (t3 - t0) * 1000.0,
    }
    timing = _make_timing(job.detect_timing, pose_timing)
    _finalize(engine, worker, job.lease, job.frame_idx, job.source, frame, persons, wrists,
              job.hand_dets, job.phone_dets, job.seatbelt_dets, danger, decision, matches, timing)
    job.lease.release()


# --------------------------------------------------------------------------
# 引擎
# --------------------------------------------------------------------------
class AccelEngine:
    """持有 NPU worker 组，按视频流产出 `FrameResult`。"""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._detect_workers: List[_Worker] = []
        self._pose_workers: List[_Worker] = []
        self._detect_queue: Optional[BoundedQueue] = None
        self._pose_queue: Optional[BoundedQueue] = None
        self._results = BoundedQueue(maxsize=max(4, self.config.frame_budget * 2))
        self._error: Optional[Exception] = None
        self._error_lock = threading.Lock()
        self._outstanding = 0
        self._out_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._writer: Optional[AsyncWriter] = None
        self._pose_rr = 0
        self._detect_rr = 0
        self._im2d: Optional[Im2d] = None
        self._started = False

    # ---- 生命周期 -------------------------------------------------
    def start(self) -> "AccelEngine":
        if self._started:
            return self
        cfg = self.config
        detect_cores = parse_cores(cfg.detect_cores, default=(0,))
        pose_cores = parse_cores(cfg.pose_cores, default=(2,))
        self._detect_queue = BoundedQueue(max(1, cfg.decode_queue))
        self._pose_queue = BoundedQueue(max(1, cfg.decode_queue))
        self._detect_workers = [
            _Worker(self, f"detect-c{core}", cfg.detect_model, core, self._detect_queue,
                    _detect_handler, cfg.imgsz, cfg.npu_warmup)
            for core in detect_cores
        ]
        self._pose_workers = [
            _Worker(self, f"pose-c{core}", cfg.pose_model, core, self._pose_queue,
                    _pose_handler, cfg.imgsz, cfg.npu_warmup)
            for core in pose_cores
        ]
        for worker in self._detect_workers + self._pose_workers:
            worker.start()
        for worker in self._detect_workers + self._pose_workers:
            worker.ready.wait(timeout=60.0)
        if self._error is not None:
            self.release()
            raise self._error
        self._started = True
        return self

    def release(self) -> None:
        for q in (self._detect_queue, self._pose_queue):
            if q is not None:
                q.close()
        for worker in self._detect_workers + self._pose_workers:
            worker.join(timeout=30.0)
        self._detect_workers = []
        self._pose_workers = []
        self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.release()

    # ---- 内部 -----------------------------------------------------
    def _fail(self, exc: Exception) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc
        for q in (self._detect_queue, self._pose_queue):
            if q is not None:
                q.close()

    def _emit(self, result: FrameResult, lease: Lease) -> None:
        lease.retain()
        result._lease = lease
        if not self._results.put(result):
            lease.release()

    def _submit_pose(self, job: _PoseJob) -> None:
        if not self._pose_workers:
            raise RuntimeError("no pose worker configured")
        worker = self._pose_workers[self._pose_rr % len(self._pose_workers)]
        self._pose_rr += 1
        if not worker.jobs.put(job):
            raise RuntimeError("pose queue closed")

    def _lease_created(self) -> None:
        with self._out_lock:
            self._outstanding += 1
            self._idle.clear()

    def _lease_released(self) -> None:
        with self._out_lock:
            self._outstanding = max(0, self._outstanding - 1)
            if self._outstanding == 0:
                self._idle.set()

    # ---- 主流程 ---------------------------------------------------
    def iter_video(self, source, source_name: Optional[str] = None, writer=None,
                   frame_stride: Optional[int] = None, fps: Optional[float] = None) -> Iterator[FrameResult]:
        """逐帧产出 `FrameResult`；抽帧在**解码侧**完成，跳过的帧不建 Frame/不占预算。"""
        cfg = self.config
        stride = max(1, int(frame_stride or cfg.frame_stride))
        cap = open_capture(source, codec=cfg.codec, zero_copy=cfg.zero_copy,
                           fallback=True, cpu=(cfg.decode == "cpu"))
        name = source_name or str(source)
        budget = FrameBudget(cfg.frame_budget)
        raw = BoundedQueue(cfg.decode_queue)
        decode_done = threading.Event()

        if writer is not None:
            self._writer = AsyncWriter(writer, queue_size=cfg.encode_queue).start()

        def decode_loop():
            decoded = 0
            try:
                while not decode_done.is_set():
                    if not budget.acquire(timeout=0.1):
                        continue
                    decoded += 1
                    if stride > 1 and (decoded - 1) % stride != 0:
                        # 抽帧：只 pull 掉，不取 fd、不建 Frame（最省 CPU/GIL）
                        alive = cap.drop_one()
                        budget.release()
                        if not alive:
                            break
                        continue
                    frame = cap.read()
                    if frame is None:
                        budget.release()
                        break
                    lease = Lease(frame, budget, on_release=self._lease_released)
                    self._lease_created()
                    if not raw.put((decoded, lease)):
                        lease.release()
                        break
            except Exception as exc:      # pragma: no cover - 解码器异常
                self._fail(exc)
            finally:
                decode_done.set()

        decoder = threading.Thread(target=decode_loop, name="decode-worker", daemon=True)
        decoder.start()

        submitted = collections.deque()      # 提交顺序（帧号可能不连续）
        ready = {}
        try:
            while True:
                if self._error is not None:
                    raise self._error
                progressed = False
                while True:
                    item = raw.get_nowait()
                    if item is None:
                        break
                    frame_idx, lease = item
                    submitted.append(frame_idx)
                    worker = self._detect_workers[self._detect_rr % len(self._detect_workers)]
                    self._detect_rr += 1
                    if not worker.jobs.put(_Job(frame_idx, name, lease)):
                        lease.release()
                        raise RuntimeError("detect queue closed")
                    progressed = True
                while True:
                    result = self._results.get_nowait()
                    if result is None:
                        break
                    ready[result.frame_idx] = result
                    progressed = True
                while submitted and submitted[0] in ready:
                    result = ready.pop(submitted.popleft())
                    try:
                        yield result
                    finally:
                        result.release()
                if decode_done.is_set() and not submitted and not ready:
                    break
                if not progressed:
                    # 没有可取的任务/结果时阻塞等一个结果，避免空转抢 GIL
                    result = self._results.get(timeout=0.005)
                    if result is not None:
                        ready[result.frame_idx] = result
        finally:
            decode_done.set()
            raw.close()
            decoder.join(timeout=10.0)
            cap.release()
            if self._writer is not None:
                self._writer.close()
                self._writer = None
