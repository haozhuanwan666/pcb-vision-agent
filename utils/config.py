"""
全局配置模块
============
集中管理项目路径、模型路径、检测参数、Agent 与 LLM 配置。
所有模块都应从本模块读取配置，避免硬编码路径。
"""
import os
from pathlib import Path

# ============================================================
# 路径
# ============================================================
# 包根目录: d:\vscode python\pcb_vision_agent
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# 工作区根目录: d:\vscode python（放置训练权重 runs/、数据集 dataset/ 等）
WORKSPACE_ROOT = PACKAGE_ROOT.parent

# 输出目录
OUTPUT_DIR = PACKAGE_ROOT / "output"
OUTPUT_DETECTED = OUTPUT_DIR / "detected"   # 异常工件图归档
OUTPUT_REPORTS = OUTPUT_DIR / "reports"     # Excel/JSON 报表
OUTPUT_LOGS = OUTPUT_DIR / "logs"           # 运行日志

# 数据目录（本包内置，用于放置测试/样例数据）
DATA_DIR = PACKAGE_ROOT / "data"
DATA_IMAGES = DATA_DIR / "images"
DATA_LABELS = DATA_DIR / "labels"
MODELS_DIR = PACKAGE_ROOT / "models"

# 训练好的 YOLO 权重（默认指向工作区的训练产物，不复制文件）
DEFAULT_YOLO_WEIGHTS = str(
    WORKSPACE_ROOT / "runs" / "train" / "pcb_defect-5" / "weights" / "best.pt"
)
# 兜底权重：若用户将权重放入 models/ 则优先使用
FALLBACK_YOLO_WEIGHTS = str(MODELS_DIR / "best.pt")
# 备用官方预训练权重（仅演示用，权重缺失时兜底）
DEMO_YOLO_WEIGHTS = str(WORKSPACE_ROOT / "yolov8n.pt")


def get_yolo_weights() -> str:
    """
    返回可用的 YOLO 权重路径。
    优先级：环境变量 YOLO_WEIGHTS > 训练产物 best.pt > 包内 models/best.pt > 官方 yolov8n.pt。
    均不存在时返回默认训练产物路径（调用方会打印告警）。
    """
    env = os.getenv("YOLO_WEIGHTS")
    for p in (env, DEFAULT_YOLO_WEIGHTS, FALLBACK_YOLO_WEIGHTS, DEMO_YOLO_WEIGHTS):
        if p and os.path.exists(p):
            return p
    return DEFAULT_YOLO_WEIGHTS


# ============================================================
# 检测默认参数
# ============================================================
DEFAULT_CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MIN_AREA = 30.0

# ============================================================
# Agent / LLM 配置
# ============================================================
# simulate(规则模拟) / llm(本地大模型)，可用环境变量 VISION_AGENT_MODE 切换
AGENT_MODE = os.getenv("VISION_AGENT_MODE", "simulate")

# Ollama OpenAI 兼容接口
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "ollama")

# 支持的图片扩展名
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
