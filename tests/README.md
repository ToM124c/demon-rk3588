# 验证与基准

板端运行（conda 环境 `jl-rk3588`），全部脚本都会读 `TEST_VIDEO` 环境变量，
默认用 `phone_call_rknn.config.DEFAULT_SOURCE`。

```bash
cd /mnt/jl/driver-phone-call-rknn
export TEST_VIDEO=/mnt/jl/yolov26-rk3588/video/VID_20260603_102426_720p_h264.mp4

python tests/test_rga_single_pass.py     # RGA 单次 CSC 正确性 + 耗时
python tests/test_rknn_quantized.py      # want_float=0 与 RKNNLite 逐元素对齐
python tests/test_engine_pipeline.py     # 零拷贝引擎端到端 + 吞吐
python tests/test_service_events.py      # 事件流结构（假 MinIO，无需联网）
python tests/bench_compare.py            # 新旧管线吞吐对比
```

| 脚本 | 验证内容 | 判定标准 |
|---|---|---|
| `test_rga_single_pass.py` | 一步 letterbox vs 旧三步实现 | 平均像素差 < 1（实测 ≈0.19），且更快 |
| `test_rknn_quantized.py` | int8/fp16 输出 + 自反量化 vs RKNNLite | detect max diff < 1e-4（实测 0），pose < 1e-2 |
| `test_engine_pipeline.py` | dma-buf 解码 → RGA → NPU 全链路 | 吞吐 ≥ 8 FPS（stride=15），预处理 < 6ms |
| `test_service_events.py` | NDJSON 事件结构 | stage_start 为首事件，字段完整，截图上传 |
| `bench_compare.py` | 新旧管线同视频对比 | 打印吞吐与各阶段耗时 |

实测参考（ROC-RK3588S-PC，720p H.264 24fps，stride=15，40 帧）：

| 指标 | 旧管线 | 新引擎 | 变化 |
|---|---|---|---|
| 吞吐 | 3.9 FPS | 16.9 FPS | **4.3×** |
| 预处理 | 21.2 ms | 2.1 ms | **10×** |
| 单帧 total | 89.5 ms | 65.7 ms | −27% |
| 判定一致率 | — | 39/40 帧 | 差异帧为 1 帧 hand/pose 分支切换（同为 danger=True） |
