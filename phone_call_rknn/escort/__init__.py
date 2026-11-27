"""护送 / 交警手势合规检测（hand 检测 + pose 关键点）。

模块划分：
- `config.py`    模型路径、阈值、角色/颜色/标签常量、EscortConfig
- `rules.py`     纯逻辑：角色分配、手部分配、合规判定、单帧记录
- `visualize.py` 绘制：角色框、关键点、接触线、中文状态面板
- `pipeline.py`  零拷贝推理管线（dma-buf -> RGA 一次 -> 两个模型共享输入 -> NPU）
- `cli.py`       命令行入口（python -m phone_call_rknn.escort）
"""

from .config import EscortConfig, default_escort_config
from .pipeline import EscortPipeline, EscortResult

__all__ = ["EscortConfig", "default_escort_config", "EscortPipeline", "EscortResult"]
