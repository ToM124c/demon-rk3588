# Driver Phone Call RKNN

RK3588 上的「司机打电话 / 未系安全带」实时检测。输入视频经 **VPU 硬解 → RGA 单次
CSC 预处理 → NPU 推理（零拷贝 + int8 原始输出）→ 业务判定**，既可以作为 HTTP 服务
（NDJSON 流式事件 + MinIO 取流/存图），也可以命令行跑单条视频。

> 目标平台：ROC-RK3588S-PC / RK3588（Firefly 系统，Python 3.10，`librknnrt 2.3.x`）。

---

## 三条关键优化（相对旧实现）

| 优化 | 做法 | 实测收益（720p，stride=15，40 帧） |
|---|---|---|
| **RGA 单次 CSC** | 一次 `improcess` 完成 NV12→RGB + 缩放 + letterbox 偏移，取代「cvtcolor → resize → cvtcolor」三次操作 | 预处理 **19.3 ms → 1.7 ms**（单测 14.9 → 2.2 ms） |
| **want_float=0** | 直接拿 int8/fp16 原始输出，在量化域做阈值筛选 + 只反量化候选行 | 推理 **46.0 ms → 30.2 ms**；与 RKNNLite 浮点结果逐元素一致（max diff = 0） |
| **dma-buf 零拷贝** | `mppvideodec` 输出 `memory:DMABuf` → RGA 读 fd → 写进 RKNN 输入 dma-buf（`rknn_set_io_mem`），整条链路不落 CPU 内存 | 端到端吞吐 **4.0 → 34.1 FPS（8.5×）**，单帧 77.4 → 42.2 ms |

判定结果一致性：40 帧里 **39 帧完全一致（97.5%）**，唯一差异是同为 `danger=True`
的 hand 分支与 pose 分支切换（框位置差 1.5px 导致）。

> 具体原理、踩坑与验证方法见 [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)；
> 数据流与线程模型见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 目录结构

```
phone_call_rknn/
├── config.py            # 模型路径、阈值、EngineConfig（全部可用环境变量覆盖）
├── api_server.py        # FastAPI 入口（uvicorn phone_call_rknn.api_server:app）
├── main_accel.py        # 加速链路 CLI（推荐）
├── main.py              # 参考链路 CLI（RKNNLite 单线程，逐帧 numpy）
├── hw/                  # 硬件后端（只依赖 Rockchip 运行时）
│   ├── rknn.py          #   librknnrt 的 ctypes 绑定：零拷贝 IO + int8 输出
│   ├── rga.py           #   librga im2d：fd 包装、单次 improcess、letterbox
│   ├── gst.py           #   GStreamer：dma-buf appsink 采集 / mpph264enc 硬编
│   └── frame.py         #   Frame（fd/va/stride）与按需 BGR 转换
├── core/                # 与硬件无关：postprocess/geometry/rules/records/timing/visualize
├── pipeline/
│   ├── engine.py        #   零拷贝流水线编排（每核一线程）
│   ├── queues.py        #   有界队列 + 帧预算 + 引用计数 Lease + 异步编码
│   ├── simple.py        #   旧的逐帧流程（图片/视频，参考实现）
│   └── lite_runner.py   #   RKNNLite 封装（参考实现用）
└── app/
    ├── service.py       # MinIO 取流、违规时段判定、NDJSON 事件
    ├── escort.py        # 「护送/手势」模式（外部脚本，历史路径）
    ├── cli.py           # 命令行参数
    └── api_test.py      # 上传 + 调接口自测

tests/                   # 验证脚本与基准（见 tests/README.md）
legacy/                  # 历史实现留档：旧加速管线、rknn_multi_threaded 副本等
docs/                    # 架构与优化说明
```

---

## 快速开始

### 1. 环境

```bash
conda activate jl-rk3588
pip install -r requirements.txt          # fastapi/uvicorn/minio/numpy/opencv
python -c "from phone_call_rknn.hw.rknn import RknnContext; print('runtime ok')"
```

板端需要：`librknnrt.so`、`librga.so`、GStreamer `mppvideodec/mpph264enc`、
以及系统 `python3-gi`（conda 环境会自动把 `/usr/lib/python3/dist-packages`
加入 `sys.path` 复用系统 gi）。

### 2. HTTP 服务

```bash
# MODE: driver（默认，打电话/安全带）或 escort（护送/手势）
uvicorn phone_call_rknn.api_server:app --host 0.0.0.0 --port 8000

# 调用：taskId 是 MinIO 里的视频对象名
curl -N "http://127.0.0.1:8000/analyze/37ecab44-9e0a-4488-9ad1-0fd1cb832ef4.mp4?mode=driver"
```

事件流（`application/x-ndjson`）：`stage_start` → 若干 `violation`（含 `coding`、
`img_url`、`wrong_action_duration`）→ `completed` / `all_completed`，异常时为 `failed`。

### 3. 命令行

```bash
# 加速链路（推荐）：输出 frame_results.jsonl，可选可视化视频
python -m phone_call_rknn.main_accel \
  --source /path/video.mp4 --output-dir runs/demo \
  --frame-stride 15 --detect-cores 0,1 --pose-cores 2 --save-vis

# 参考链路（RKNNLite 单线程，图片或视频）
python -m phone_call_rknn.main --source /path/images --output-dir runs/ref --no-save-vis
```

### 4. 自测

```bash
export TEST_VIDEO=/path/to/720p.mp4
python tests/test_rga_single_pass.py     # RGA 单次 CSC 正确性/耗时
python tests/test_rknn_quantized.py      # want_float=0 与 RKNNLite 对齐
python tests/test_engine_pipeline.py     # 零拷贝引擎吞吐
python tests/test_decision_parity.py     # 新旧判定一致性
python tests/test_service_events.py      # 事件流（假 MinIO）
python tests/bench_compare.py            # 新旧管线对比
```

---

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DETECT_MODEL` / `POSE_MODEL` | `/mnt/jl/yolov26-rk3588/model/...` | RKNN 模型路径（或目录，自动找第一个 `.rknn`） |
| `DETECT_CORES` / `POSE_CORES` | `0,1` / `2` | NPU 绑核；`all` 表示三核合一 |
| `ZERO_COPY` | `1` | 关闭后走 OpenCV appsink + numpy 兜底 |
| `DECODE` | `vpu` | `vpu`（GStreamer 硬解）或 `cpu`（OpenCV 软解） |
| `CODEC` | `h264` | `h264` / `h265`；HEVC 走 10bit 时自动回退 CPU CSC |
| `FRAME_STRIDE` | `1` | 每 N 帧推理一帧（API 默认 15，见 `DRIVER_FRAME_STRIDE`） |
| `FRAME_BUDGET` | `6` | 整条流水线同时持有的帧数（同时限制解码 dma-buf 数量） |
| `DECODE_QUEUE` / `ENCODE_QUEUE` | `4` / `2` | 解码/编码队列深度（背压） |
| `DET_CONF` / `PERSON_CONF` / `KPT_CONF` / `NMS_IOU` | `0.35/0.25/0.30/0.45` | 阈值 |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 见代码 | 对象存储；**生产环境请用环境变量注入，不要提交明文** |
| `VIOLATION_MIN_FRAMES` | `5` | 连续多少个采样帧判为违规才报一个时段 |

---

## 已知限制

- **零拷贝依赖 `memory:DMABuf`**：`v4l2src` 摄像头通常拿不到 dma-buf，会自动回退
  OpenCV 路径（有一次 CPU 拷贝）；`vpu + h265` 若输出 10bit NV12，RGA 2.1 不支持
  10bit CSC，会回退到 CPU 转换（慢但正确）。
- **可视化/取证会走 CPU**：只有需要 BGR 时才做 `NV12→BGR`（~2.7 ms/帧），
  API 模式下仅违规截图时才触发。
- **抽帧不省解码**：`frame_stride>1` 仍要解码每一帧（文件/RTSP 无法按帧 seek），
  只是不建帧对象、不进推理。
- **`librknnrt` 与模型版本**：模型由 toolkit 2.3.2 编译，板端运行时 2.3.0，
  目前工作正常；升级模型时请同步升级 `librknnrt`（指南的「铁律」）。
- `escort` 模式仍依赖仓库外的
  `/mnt/jl/yolov26-rk3588/inference-police-escort-rknn-hand-assign.py`。

## 相关文档

- [RK3588 应用开发学习指南](RK3588-应用开发学习指南.md)
- [架构说明](docs/ARCHITECTURE.md)
- [优化记录（含验证数据）](docs/OPTIMIZATION.md)
- [测试与基准](tests/README.md)
- [legacy 说明](legacy/README.md)
