# Driver Phone Call RKNN
# 1.激活虚拟环境
conda activate jl-rk3588
# 2.进入文件夹
cd /mnt/jl/driver-phone-call-rknn
# 3.启动服务，切换模式 MODE_ESCORT  MODE_DRIVER  重新启动
uvicorn phone_call_rknn.api_server:app --host 0.0.0.0 --port 8000
# 4.测试指令
测试escort：curl -N "http://192.16.110.243:8000/analyze/37ecab44-9e0a-4488-9ad1-0fd1cb832ef4.mp4"
测试安全带或者接打电话：curl -N "http://192.16.110.243:8000/analyze/37ecab44-9e0a-4488-9ad1-0fd1cb832ef4.mp4"

test-phone-seatbelt.mp4：  f8ae3ce3-e948-4f3b-97c9-53e196121eeb.mp4

这是从 `/mnt/jl/yolov26-rk3588/inference-driver-phone-call-rknn-3.py` 拆分出的模块化版本，默认模型路径、输入路径、输出路径和原脚本保持一致。

## 运行

```bash
cd /mnt/jl/driver-phone-call-rknn
python run.py
```

也可以继续传入原脚本支持的参数：

```bash
python run.py \
  --source /path/to/video.mp4 \
  --output-dir /path/to/output \
  --det-conf 0.35 \
  --person-conf 0.25 \
  --kpt-conf 0.30
```

## 结构

- `main.py`: 命令行入口。
- `phone_call_rknn/cli.py`: 参数解析。
- `phone_call_rknn/config.py`: 默认路径、类别、阈值和关键点索引。
- `phone_call_rknn/preprocess.py`: letterbox 和输入张量预处理。
- `phone_call_rknn/geometry.py`: 坐标转换、IoU、NMS、距离计算等通用函数。
- `phone_call_rknn/postprocess.py`: detect 和 pose 模型输出解码。
- `phone_call_rknn/rknn_runner.py`: RKNNLite 模型加载、推理和释放。
- `phone_call_rknn/rules.py`: hand/phone/wrist 危险判定逻辑。
- `phone_call_rknn/records.py`: 单帧 JSON 结果生成和写入。
- `phone_call_rknn/visualization.py`: 可视化绘制。
- `phone_call_rknn/timing.py`: 单帧和总体耗时统计。
- `phone_call_rknn/media.py`: 图片、视频输入识别。
- `phone_call_rknn/pipeline.py`: 单帧、图片集、视频推理流程。

