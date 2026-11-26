# 优化记录：零拷贝 / RGA 单次 CSC / want_float=0

对应 `RK3588-应用开发学习指南.md` 第 9 节里收益最高的三条。所有数字都在
ROC-RK3588S-PC 上实机测得（720p H.264 24 fps，`TEST_VIDEO` 见 `tests/README.md`）。

## 0. 基线（重构前）

旧链路：`OpenCV CAP_GSTREAMER` → NV12 numpy → RGA cvtcolor(NV12→BGR) →
RGA resize → RGA cvtcolor(BGR→RGB) → `RKNNLite.inference(numpy)` → 浮点后处理。

| 阶段 | 耗时 |
|---|---|
| 预处理（3 次 RGA + 一次整帧 BGR） | 19–21 ms |
| 推理（含 runtime 反量化成 float32） | 45–46 ms |
| 后处理（NMS/解码） | 12–15 ms |
| 吞吐（stride=15，40 帧） | 4.0 FPS |

## 1. RGA 单次 CSC

`hw/rga.py::Letterboxer.from_buffer()` 用一次 `improcess` 完成
**缩放 + 色彩转换 + 目标矩形偏移**：

```python
im2d.process(src_nv12, dst_rgb,
             srect=Rect(0, 0, src_w, src_h),
             drect=Rect(left, top, new_w, new_h))   # letterbox 偏移写进 640x640 画布
```

要点与坑：

- **`improcess` 的 usage 必须是 0**。实测 `usage=1` 时 RGA 认为带旋转/混合算子，
  输出是噪声（与参考图相关性 ≈ −0.09）；`usage=0` 与 `imcvtcolor` 逐像素一致。
- **RGA 不允许 fd 与 handle 混用**：`wrapbuffer_fd` 的 buffer 只能配 fd 目标；
  配 numpy 目标时报 `improcess failed: status=0`。`Letterboxer` 因此同时维护
  fd 目标与 handle 目标，按 src 类型选择（`Wrapped.kind`）。
- 边框灰色只在**几何变化时**填一次（`imfill`），之后每帧只写中间区域。
- stride 用真实值（NV12 的 `wstride/hstride` 来自 dma-buf 大小或 GstVideoMeta）。

实测：

| 实现 | 耗时 |
|---|---|
| 旧三步（cvtcolor + resize + cvtcolor） | 14.9 ms |
| 单次 improcess（VA 目标） | 2.2 ms |
| 单次 improcess（fd 目标，引擎内） | 1.7 ms |
| 像素差 | 平均 0.185（缩放插值差异，颜色一致） |

## 2. want_float=0

`hw/rknn.py::RknnContext.run()` 用 `rknn_outputs_get(..., want_float=0)` 拿原始张量：

- detect 的 6 个输出是 **INT8**，直接 `np.ctypeslib.as_array().copy()`；
- pose 的输出是 **FP16**（模型输出未量化），按 float16 视图取出。

后处理在量化域里做筛选（`core/postprocess.py`）：

```python
cls_i = cls_q[:n_cls].astype(np.int16)
cls_idx = np.argmax(cls_i, axis=0)              # int 域 argmax == float 域 argmax
q_max = np.take_along_axis(cls_i, cls_idx[None], axis=0)[0]
scores = sigmoid((q_max - zp) * scale)          # 只对 6400 个格点算一次
mask = scores.reshape(-1) >= conf_thres
# 只对命中的行做回归头反量化与坐标还原
```

pose 同理：先在 fp16 的置信度列上筛选，再对候选行 `astype(float32)`，避免把
470k 个元素整体转成 float32。

验证（`tests/test_rknn_quantized.py`）：与 `RKNNLite.inference()` 的 float32
结果逐元素比较，**detect 6 个输出 max diff = 0，pose max diff = 0**。

| 指标 | 旧（float32 全量） | 新（int8/fp16 + 懒反量化） |
|---|---|---|
| 推理阶段 | 46.0 ms | 30.2 ms |
| 后处理阶段 | 12.1 ms | 10.3 ms |
| 结果 | — | 完全一致 |

## 3. dma-buf 零拷贝

三段都换成 fd：

1. **解码**：`hw/gst.py::DmaBufCapture` 用 caps
   `video/x-raw(memory:DMABuf),format=NV12` + `appsink`，从 `GstMemory`
   取 `dmabuf_memory_get_fd()`，帧以 `Frame(fd=...)` 传递（`GstSample` 保活，
   RGA 用完即放）。
2. **预处理**：RGA `wrapbuffer_fd_t(fd, ...)` 直接把硬解缓冲当源，目标写进
   RKNN 的输入 dma-buf。
3. **推理**：`rknn_create_mem` 分配输入 dma-buf，`rknn_set_io_mem` 绑定；
   `attr.pass_through=0 + type=UINT8 + fmt=NHWC` 让 runtime 做归一化
   （模型输入本身是 INT8/FP16，pass_through=1 会喂错数据）。

实测：整条链路每帧 **没有任何 CPU 帧拷贝**（可用 `tests/test_engine_pipeline.py`
的耗时反推：预处理 1.7 ms ≈ 纯 RGA 时间）。

## 4. 顺手修掉的瓶颈

- **抽帧下沉到解码侧**：`frame_stride>1` 时用 `cap.drop_one()` 丢掉未选中的帧，
  不建 `Frame`、不占预算、不进队列。之前跳帧仍走完整对象创建路径，等于把
  「抽帧」做成了「限流」。
- **按需 BGR**：推理链路不再生成整帧 BGR（旧实现每帧都转），仅在 `--save-vis`、
  显示、或违规截图时转换（`FrameResult.bgr()` 懒加载）。
- **有界队列 + 帧预算 + Lease**：解码不会跑飞，内存稳定在 `frame_budget` 帧；
  乱序结果按提交顺序还原。
- **绑定 npu 核**：detect 2 实例绑 core0/1、pose 1 实例绑 core2，各自线程内独占
  上下文，推理路径无锁。

## 5. 最终对比

`tests/bench_compare.py`（40 帧，stride=15，同一视频、同一模型）：

| 指标 | 旧管线 | 新引擎 | 变化 |
|---|---|---|---|
| 吞吐 | 4.0 FPS | 34.1 FPS | **8.5×** |
| 预处理 | 19.3 ms | 1.7 ms | **11×** |
| 推理 | 46.0 ms | 30.2 ms | −34% |
| 后处理 | 12.1 ms | 10.3 ms | −15% |
| 单帧 total | 77.4 ms | 42.2 ms | −45% |
| 判定一致率 | — | 39/40 = 97.5% | 差异帧同为 danger=True |

> 吞吐在不同轮次间有 16–34 FPS 的波动（DVFS/温度/同板其它负载），
> 上表是一次连续测量；趋势与阶段耗时稳定。

## 6. 还没做的

- 输出也走零拷贝（`rknn_set_io_mem` 绑输出 dma-buf，省掉 `outputs_get` 的 int8
  拷贝）——收益较小（int8 输出总量 <100KB），但 posenet 的 940KB fp16 值得一试。
- 抽帧时跳过解码（文件可用 `qtdemux` seek + 关键帧对齐，RTSP 不适用）。
- 编码侧去 CPU `videoconvert`（RGA 直接出 NV12 喂 `mpph264enc`）。
- 多路 RTSP 并发（当前一个引擎实例串行处理一条流）。
