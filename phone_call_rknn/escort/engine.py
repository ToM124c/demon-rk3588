"""护送模式的零拷贝推理管线。

与司机模式（`pipeline/engine.py`）的区别：每帧**同时**跑 hand 检测和 pose，
两个模型共享同一块输入 dma-buf（一次 RGA），因此每个 worker 内部是顺序执行、
不需要 detect→pose 的任务路由。

    dma-buf 解码 ──RGA 一次──► 共享输入 dma-buf ──► hand 模型 ──┐
                                              └──► pose 模型 ──┴─► 角色/手部/合规判定
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

from ..common.timing import make_timing, round_timing
from ..hw.frame import Frame, frame_to_bgr
from ..hw.gst import open_capture
from ..hw.rga import Im2d, Letterboxer
from ..hw.rknn import CORE_MASKS, TENSOR_NHWC, TENSOR_UINT8, RknnContext
from ..pipeline.frames import preprocess_frame
from ..pipeline.queues import AsyncWriter, BoundedQueue, FrameBudget, Lease
from ..vision.postprocess import decode_detect_quantized, decode_pose_half
from .config import HAND_CLASSES, LABEL_COMPLIANT, EscortConfig
from .rules import (
    assign_hands_to_roles,
    assign_roles_by_right_shoulder,
    evaluate_frame,
    extract_pose_people,
    make_frame_record,
)
from .visualize import annotate_frame


def parse_cores(value) -> List[int]:
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
    return cores or [2]


@dataclass
class EscortResult:
    frame_idx: int
    source: str
    record: dict
    label: str
    frame_result: dict
    analysis: dict = field(default_factory=dict)   # 供可视化复用的中间结果
    vis: Optional[np.ndarray] = None
    _lease: Optional[Lease] = field(default=None, repr=False)
    _frame_bgr: Optional[np.ndarray] = field(default=None, repr=False)
    _im2d: Optional[Im2d] = field(default=None, repr=False)

    @property
    def compliant(self) -> bool:
        return self.label == LABEL_COMPLIANT

    def bgr(self, im2d: Optional[Im2d] = None) -> np.ndarray:
        if self._frame_bgr is not None:
            return self._frame_bgr
        if self._lease is None:
            raise RuntimeError("frame already released")
        self._frame_bgr = frame_to_bgr(self._lease.frame, im2d=im2d or self._im2d or Im2d())
        return self._frame_bgr

    @property
    def frame(self) -> np.ndarray:
        return self.bgr()

    def visualize(self, show_action_details: bool = True) -> np.ndarray:
        """在取证帧上画注释（延迟到真正需要时才算）。"""
        return annotate_frame(self.bgr(), self.analysis["roles"], self.analysis["pose_people"],
                              self.analysis["hand_dets"], self.frame_result,
                              show_action_details=show_action_details)

    def release(self) -> None:
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.release()

    def __getitem__(self, key):
        if key == "frame":
            return self.bgr()
        return getattr(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default


@dataclass
class _Job:
    frame_idx: int
    source: str
    lease: Lease


class _EscortWorker(threading.Thread):
    """一个 worker 独占两个 context + 一块共享输入张量 + 一个 RGA 上下文。"""

    def __init__(self, pipeline: "EscortPipeline", name: str, core_id: int):
        super().__init__(name=name, daemon=True)
        self.pipeline = pipeline
        self.core_id = core_id
        self.jobs = BoundedQueue(max(1, pipeline.config.decode_queue))
        self.ready = threading.Event()
        self.detect: Optional[RknnContext] = None
        self.pose: Optional[RknnContext] = None
        self.tensor = None
        self.letterboxer: Optional[Letterboxer] = None
        self.pose_letterboxer: Optional[Letterboxer] = None
        self.im2d: Optional[Im2d] = None

    @property
    def shared_input(self) -> bool:
        return self.pose_letterboxer is None

    def run(self):
        cfg = self.pipeline.config
        try:
            mask = CORE_MASKS[self.core_id]
            self.im2d = Im2d()
            detect = RknnContext(cfg.hand_model, core_mask=mask, name=f"{self.name}-hand")
            pose = RknnContext(cfg.pose_model, core_mask=mask, name=f"{self.name}-pose")
            self.detect = detect
            self.pose = pose
            self.tensor = self.detect.bind_owned_input(pass_through=False, dtype=TENSOR_UINT8, fmt=TENSOR_NHWC)
            self.letterboxer = Letterboxer(cfg.imgsz, im2d=self.im2d)
            self.letterboxer.bind_fd(self.tensor.fd)

            pose_bytes = self.pose.declared_input_size(TENSOR_UINT8)
            if pose_bytes == self.tensor.size:
                # 两个模型输入声明相同（640x640x3 uint8）：共用同一块 dma-buf
                self.pose.bind_shared_input(self.tensor.fd, self.tensor.size, pass_through=False,
                                            dtype=TENSOR_UINT8, fmt=TENSOR_NHWC,
                                            virt_addr=self.tensor.mem.virt_addr)
            else:
                pose_tensor = self.pose.bind_owned_input(pass_through=False, dtype=TENSOR_UINT8, fmt=TENSOR_NHWC)
                self.pose_letterboxer = Letterboxer(cfg.imgsz, im2d=self.im2d)
                self.pose_letterboxer.bind_fd(pose_tensor.fd)

            for _ in range(max(1, cfg.npu_warmup)):
                detect.run()
                pose.run()
            self.ready.set()

            while True:
                job = self.jobs.get()
                if job is None:
                    break
                try:
                    self._analyze(job)
                except Exception as exc:      # pragma: no cover - 运行期异常
                    self.pipeline._fail(exc)
                    return
        except Exception as exc:              # pragma: no cover - 初始化异常
            self.pipeline._fail(exc)
        finally:
            self.ready.set()
            for ctx in (self.detect, self.pose):
                if ctx is not None:
                    ctx.release()
            self.detect = self.pose = None

    # ---- 单帧分析 -------------------------------------------------
    def _analyze(self, job: _Job, emit: bool = True) -> EscortResult:
        cfg = self.pipeline.config
        detect = self.detect
        pose = self.pose
        if detect is None or pose is None:
            raise RuntimeError("escort worker contexts are not initialised")
        frame = job.lease.frame
        shape = (frame.height, frame.width, 3)

        t0 = time.perf_counter()
        _rgb, scale, pad_x, pad_y = preprocess_frame(self.letterboxer, frame, self.im2d)
        shared_pre_ms = (time.perf_counter() - t0) * 1000.0
        extra_pre_ms = 0.0
        if self.pose_letterboxer is not None:
            t = time.perf_counter()
            _rgb2, p_scale, p_pad_x, p_pad_y = preprocess_frame(self.pose_letterboxer, frame, self.im2d)
            extra_pre_ms = (time.perf_counter() - t) * 1000.0
        else:
            p_scale, p_pad_x, p_pad_y = scale, pad_x, pad_y

        t1 = time.perf_counter()
        det_outputs = detect.run()
        t2 = time.perf_counter()
        pose_outputs = pose.run()
        t3 = time.perf_counter()

        dets = decode_detect_quantized(
            det_outputs, detect.outputs, scale, pad_x, pad_y, shape, cfg.det_conf, cfg.iou,
            names=HAND_CLASSES,
        )
        hand_dets = [det for det in dets if det["class"] == "hand"]
        t4 = time.perf_counter()

        persons = decode_pose_half(
            pose_outputs[0], p_scale, p_pad_x, p_pad_y, shape, cfg.person_conf, cfg.iou
        )
        pose_people = extract_pose_people(persons, cfg.kpt_conf)
        roles = assign_roles_by_right_shoulder(pose_people)
        assignments = assign_hands_to_roles(roles, hand_dets, cfg.hand_max_pixel_dist)
        frame_result = evaluate_frame(roles, assignments, hand_dets, cfg)
        t5 = time.perf_counter()

        det_infer_ms = (t2 - t1) * 1000.0
        pose_infer_ms = (t3 - t2) * 1000.0
        det_post_ms = (t4 - t3) * 1000.0
        pose_post_ms = (t5 - t4) * 1000.0
        detect_timing = {
            "preprocess_ms": extra_pre_ms,
            "rknn_inference_ms": det_infer_ms,
            "postprocess_ms": det_post_ms,
            "total_runner_ms": det_infer_ms + det_post_ms + extra_pre_ms,
        }
        pose_timing = {
            "preprocess_ms": 0.0,
            "rknn_inference_ms": pose_infer_ms,
            "postprocess_ms": pose_post_ms,
            "total_runner_ms": pose_infer_ms + pose_post_ms,
        }
        timing = round_timing(make_timing(detect_timing, pose_timing, shared_pre_ms))
        record = make_frame_record(job.frame_idx, job.source, roles, hand_dets, frame_result, timing)

        analysis = {"roles": roles, "pose_people": pose_people, "hand_dets": hand_dets,
                    "frame_result": frame_result}
        vis = None
        if not cfg.no_save_vis:
            bgr = frame_to_bgr(frame, im2d=self.im2d)
            vis = annotate_frame(bgr, roles, pose_people, hand_dets, frame_result)
            if self.pipeline._writer is not None:
                self.pipeline._writer.submit(vis)

        result = EscortResult(job.frame_idx, job.source, record, frame_result["label"],
                              frame_result, analysis, vis)
        if emit:
            # 异步路径：把结果交给 dispatcher（它负责按帧序 yield 并释放 lease）
            self.pipeline._emit(result, job.lease)
            job.lease.release()
        return result


class EscortPipeline:
    """护送模式入口：多核 worker + 有界队列 + 结果重排。"""

    def __init__(self, config: Optional[EscortConfig] = None):
        self.config = config or EscortConfig()
        self._workers: List[_EscortWorker] = []
        self._results = BoundedQueue(maxsize=max(4, self.config.frame_budget * 2))
        self._writer: Optional[AsyncWriter] = None
        self._error: Optional[Exception] = None
        self._error_lock = threading.Lock()
        self._outstanding = 0
        self._out_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._rr = 0
        self._started = False

    # ---- 生命周期 -------------------------------------------------
    def start(self) -> "EscortPipeline":
        if self._started:
            return self
        cfg = self.config
        cores = parse_cores(cfg.cores)
        self._workers = [_EscortWorker(self, f"escort-c{core}", core) for core in cores]
        for worker in self._workers:
            worker.start()
        for worker in self._workers:
            worker.ready.wait(timeout=60.0)
        if self._error is not None:
            self.release()
            raise self._error
        self._started = True
        return self

    def release(self) -> None:
        for worker in self._workers:
            worker.jobs.close()
        for worker in self._workers:
            worker.join(timeout=30.0)
        self._workers = []
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
        for worker in self._workers:
            worker.jobs.close()

    def _emit(self, result: EscortResult, lease: Lease) -> None:
        lease.retain()
        result._lease = lease
        if not self._results.put(result):
            lease.release()

    def _lease_created(self) -> None:
        with self._out_lock:
            self._outstanding += 1
            self._idle.clear()

    def _lease_released(self) -> None:
        with self._out_lock:
            self._outstanding = max(0, self._outstanding - 1)
            if self._outstanding == 0:
                self._idle.set()

    # ---- 单张图片（同步，直接复用第一个 worker 的上下文）------------
    def run_frame(self, frame: np.ndarray, frame_idx: int = 1, source_name: str = "image") -> EscortResult:
        if not self._started:
            self.start()
        worker = self._workers[0]
        img = Frame(width=frame.shape[1], height=frame.shape[0], pixel_format="bgr", va=frame, _keepalive=frame)
        budget = FrameBudget(2)
        budget.acquire()
        lease = Lease(img, budget, on_release=self._lease_released)
        self._lease_created()
        job = _Job(frame_idx, source_name, lease)
        # 同步路径：不进结果队列（图片批量时队列会满），直接复用手里已有的 BGR
        result = worker._analyze(job, emit=False)
        result._frame_bgr = frame
        lease.release()
        return result

    # ---- 视频流 ---------------------------------------------------
    def iter_video(self, source, source_name: Optional[str] = None, writer=None,
                   frame_stride: Optional[int] = None) -> Iterator[EscortResult]:
        cfg = self.config
        stride = max(1, int(frame_stride or cfg.frame_stride))
        cap = open_capture(source, codec=cfg.codec, fallback=cfg.decode_fallback,
                           cpu=(cfg.decode == "cpu"))
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
            except Exception as exc:      # pragma: no cover
                self._fail(exc)
            finally:
                decode_done.set()

        decoder = threading.Thread(target=decode_loop, name="escort-decode", daemon=True)
        decoder.start()

        submitted = collections.deque()
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
                    worker = self._workers[self._rr % len(self._workers)]
                    self._rr += 1
                    if not worker.jobs.put(_Job(frame_idx, name, lease)):
                        lease.release()
                        raise RuntimeError("escort queue closed")
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
