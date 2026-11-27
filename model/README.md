# 模型目录（不入库）

把 `.rknn` 模型放在这里，代码会**优先**使用本目录的文件
（`phone_call_rknn/config.py::model_path`：环境变量 > `model/` > 外部默认路径）。

需要这四个文件（文件名必须一致）：

| 文件 | 用途 | 说明 |
|---|---|---|
| `0704-PHS.rknn` | 司机行为检测 | hand / phone / seatbelt 三类，输入 INT8 640×640 |
| `yolo26n-pose-rk3588.rknn` | 司机姿态 | 17 关键点，输入 FP16 |
| `0710-escort-hand.rknn` | 护送 hand 检测 | 单类 hand |
| `0710-escort-pose-rk3588.rknn` | 护送姿态 | 17 关键点 |

默认外部路径是 `/mnt/jl/yolov26-rk3588/model/`；也可以用环境变量覆盖，例如：

```bash
export DETECT_MODEL=/path/0704-PHS.rknn
export POSE_MODEL=/path/yolo26n-pose-rk3588.rknn
export ESCORT_DETECT_MODEL=/path/0710-escort-hand.rknn
export ESCORT_POSE_MODEL=/path/0710-escort-pose-rk3588.rknn
```

> 二进制模型默认被 `.gitignore` 忽略（避免仓库膨胀）。如需入库：
> `git add -f model/*.rknn`，或改用 Git LFS。
