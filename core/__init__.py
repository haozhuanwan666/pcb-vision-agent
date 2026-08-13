"""
core 子包：PCB 视觉检测核心引擎
- vision_toolkit : 视觉算子工具箱（14 个函数）
- detector       : 检测引擎（质量评估/增强/传统/YOLO/Halcon 统一流水线）
- agent          : 视觉决策 Agent（simulate / llm 双模式）
- report         : Excel/JSON 报表
- batch          : 批量检测
- benchmark      : 性能测试
"""
from .detector import PcbDetector, DetectionResult
from .agent import VisionAgent, get_all_tools, create_local_llm

__all__ = [
    "PcbDetector",
    "DetectionResult",
    "VisionAgent",
    "get_all_tools",
    "create_local_llm",
]
