# 架构说明

## 分层

```
┌──────────────────────────────────────────────────────────┐
│ app/        FastAPI / NDJSON 事件 / MinIO / 违规时段跟踪    │
├──────────────────────────────────────────────────────────┤
│ pipeline/   编排：解码线程 → 推理 worker → 结果重排 → 输出   │
├──────────────────────────────────────────────────────────┤
│ hw/         硬件后端：GStreamer(dma-buf) / RGA im2d / RKNN  │
├──────────────────────────────────────────────────────────┤
│ core/       与硬件无关：后处理、几何、规则、记录、可视化      │
└──────────────────────────────────────────────────────────┘
```

`core/` 不 import `hw/`，因此后处理可以在任意机器上单测。

## 数据流（零拷贝路径）

```
文件/RTSP
   │  H.264/H.265
   ▼
mppvideodec ──► video/x-raw(memory:DMABuf), NV12
   │                 │  appsink: GstSample
   │                 ▼
   │            Frame(fd, w, h, wstride, hstride, keepalive=sample)
   │                 │
   │                 ├──► RGA improcess ──► RKNN 输入 dma-buf (640x640 RGB)
   │                 │      （一次完成 缩放+CSC+letterbox 偏移）
   │                 │
   │                 └──（仅在需要 BGR 时）RGA imcvtcolor ──► numpy BGR
   ▼
NPU: rknn_run(input dma-buf) ──► int8 / fp16 原始输出
   │
   ▼
core.postprocess（量化域筛选 + 只反量化候选行）──► 判定 ──► 事件/可视化
```

**非零拷贝路径**（`ZERO_COPY=0` 或摄像头）：OpenCV `CAP_GSTREAMER` 的 appsink
返回 numpy NV12/BGR，`Frame.va` 有值，RGA 走 `importbuffer_virtualaddr`
（两侧都是 handle）。两条路径共用同一套 `Frame` 与 worker 代码。

## 线程模型

| 线程 | 数量 | 职责 |
|---|---|---|
| `decode-worker` | 1 | `cap.read()` / `cap.drop_one()`，按 stride 抽帧，占用帧预算 |
| `detect-c{core}` | = `DETECT_CORES` 个数 | 独占 RKNN ctx + 输入张量 + RGA 上下文；RGA → run → 后处理 |
| `pose-c{core}` | = `POSE_CORES` 个数 | 同上，仅在判定需要看手腕时被投递 |
| `encode-worker` | 0/1 | 写可视化视频（有 writer 时） |
| 调用线程（generator） | 1 | 分发任务、按帧号还原顺序、yield 结果 |

**一核一线程、线程内独占资源**，因此推理路径上没有任何锁；跨线程只通过两个
有界队列（`raw`、`_results`）和 worker 自己的任务队列通信。

## 背压与生命周期

- `FrameBudget(frame_budget)`：整条流水线最多同时持有 N 帧（也就是 N 个解码
  dma-buf、N 份预算）。解码线程在取帧前先 `acquire`，拿不到就等，天然限流。
- `Lease`：一次帧的引用计数。谁持有帧谁 `retain`，用完 `release`；最后一个
  释放者负责 `frame.release()`（放掉 GstSample）并归还预算。
- 结果乱序：worker 完成顺序不等于帧序，`submitted` 双端队列记录提交顺序，
  `ready` 字典按帧号缓存，只有队首就绪才 yield，保证下游看到单调递增的帧号。
- 抽帧在**解码侧**完成：`cap.drop_one()` 只 pull 掉 sample，不建 Frame、不占预算、
  不经过队列，所以 stride=15 时跳过帧的 Python 开销接近 0。

## 为什么用 ctypes 而不是 `rknnlite`

`rknnlite.api.RKNNLite` 只暴露 `inference(numpy)`：

- 拿不到 `want_float=0` 的原始 int8/fp16 输出（它永远返回 float32）；
- 无法用 `rknn_create_mem` / `rknn_set_io_mem` 绑定 dma-buf，输入必须是一块
  numpy（多一次拷贝）。

`hw/rknn.py` 直接调 `librknnrt.so` 的 C ABI，结构体布局按
`rknpu2/runtime/Linux/librknn_api/include/rknn_api.h` 逐字段对齐，并用 C 程序
`sizeof/offsetof` 校验过（`rknn_tensor_attr` = 376 字节）。

## 模型 IO 的特殊之处

这两个模型并不是「UINT8 进、FP32 出」：

| 模型 | 输入 | 输出 |
|---|---|---|
| detect `0704-PHS` | INT8 (scale=1/255, zp=-128)、NHWC、3 类 × 3 stride | 6 个 INT8 张量（4 回归 + 3 分类） |
| pose `yolo26n-pose` | **FP16**、NHWC | 1 个 FP16 张量 (56, 8400) |

所以不能简单 `pass_through=1`（那要求调用方直接把数据写成 INT8/FP16 并完成归一化）。
我们的做法是：内存是 dma-buf（零拷贝），但 `attr.pass_through=0` 并把 `type/fmt`
声明成 `UINT8/NHWC`，由 runtime 做「uint8 → 模型输入」的转换——这与
`rknn_inputs_set` 的语义一致，实验证明结果与 RKNNLite 逐元素相同。

## 护送模式（escort）的差异

`phone_call_rknn/escort/` 与司机模式共用 `hw/`（dma-buf、RGA、RKNN）与 `pipeline/queues.py`，
但编排不同：每帧**同时**需要 hand 检测和 pose，所以一个 worker 内部顺序跑两个模型，
并且**两个模型共享同一块输入 dma-buf**——一次 RGA 预处理同时喂两个模型：

```
dma-buf 帧 ──RGA improcess──► 输入 dma-buf（640x640 RGB, uint8）
                                  ├─► hand ctx.rknn_run  (core 2)
                                  └─► pose ctx.rknn_run  (core 2, 用 rknn_create_mem_from_fd 导入同一个 fd)
```

要点：`rknn_create_mem_from_fd` 必须传 `virt_addr`（传 NULL 会得到
`RKNN_ERR_MALLOC_FAIL`），我们直接把 owner 张量的 `mem.virt_addr` 传进去。

判定逻辑（角色分配、手部归属、合规规则）在 `escort/rules.py`，是不依赖硬件的纯函数，
`tests/test_escort_rules.py` 用构造数据覆盖了「三人在场」「手在腕内」「双手控双肘」等场景。

## 事件层

`app/service.py` 把 `FrameResult` 转成业务事件：

```
stage_start → violation(可能多条) → completed → all_completed
```

`ViolationPeriodTracker` 按 coding 累计连续命中帧数（`VIOLATION_MIN_FRAMES`），
在时段结束时才落盘一张违规截图（此时才调用 `result.bgr()` 触发一次
`NV12→BGR`），因此正常帧不会产生任何 CPU 色彩转换。
