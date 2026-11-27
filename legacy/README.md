# legacy：历史实现留档

这里的代码**不参与生产链路**，保留原因：

- `engine_accel_old.py` —— 重构前的加速管线（numpy 帧 + RKNNLite + 三次 RGA）。
  `tests/bench_compare.py`、`tests/test_decision_parity.py` 用它做 A/B 对比。
- `rknn_multi_threaded/` —— 从 [leafqycc/rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)
  复制进来并做过本项目的修改（`func_RGA.py`、`pipeline_io.py` 等）。新代码的
  `hw/` 层取代了它的 `rga_helper` / `vpu_capture`，仅在测试对比时被引用。
  该目录自带 LICENSE（MIT 风格）与 demo 模型 `rknnModel/yolov5s_...rknn`。
- `main-2.py` —— 最早的单文件版本（1000 行），已被 `phone_call_rknn/` 完全覆盖。
- `api_server-V_1.py` / `api_server-PHS.py` —— 旧版服务实现，已被
  `phone_call_rknn/app/api_server.py` + `driver/service.py` 取代。

⚠️ 这些文件里有**明文 MinIO 账号密码**（历史遗留），仓库是公开的，
请勿在上面继续开发；生产请用环境变量注入。

## 关于原来的护送脚本

`/mnt/jl/yolov26-rk3588/inference-police-escort-rknn-hand-assign.py`（仓库外）的判定逻辑
已经迁入 `phone_call_rknn/escort/`（rules/visualize/engine），并改用仓库的零拷贝硬件层。
旧脚本不再被任何代码加载，只作为 `tests/test_escort_pipeline.py --parity` 的对照基准
（通过 `ESCORT_LEGACY_SCRIPT` 指定路径，不存在时自动跳过）。
