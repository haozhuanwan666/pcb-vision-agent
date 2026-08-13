"""
PCB 视觉检测 Agent —— 项目整合包
=================================
融合传统机器视觉(OpenCV/Halcon) + 深度学习(YOLO) + AI Agent 的 PCB 缺陷检测系统。

模块结构:
- core   : 核心引擎（视觉算子 / 检测流水线 / 决策 Agent / 报表 / 批量 / 性能测试）
- web    : Gradio Web 界面
- utils  : 配置 / 日志 / 文件工具
- data   : 数据集目录（images/train, images/val, labels）
- models : 模型权重目录
- output : 输出目录（detected 异常图 / reports 报表 / logs 日志）
- docs   : 文档
"""

__version__ = "1.0.0"
__author__ = "PCB Vision Agent Team"

# 统一从包导出的高频接口，方便外部调用
from .core.detector import PcbDetector, DetectionResult
from .core.agent import VisionAgent
from .utils.config import AGENT_MODE, DEFAULT_CONF_THRESHOLD

__all__ = [
    "PcbDetector",
    "DetectionResult",
    "VisionAgent",
    "AGENT_MODE",
    "DEFAULT_CONF_THRESHOLD",
    "__version__",
]
