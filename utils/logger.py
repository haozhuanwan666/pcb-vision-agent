"""
日志模块
========
统一控制台 + 文件双输出日志。
解决 Windows 控制台默认 GBK 编码无法输出 emoji 的问题。
"""
import logging
import sys
from datetime import datetime

from .config import OUTPUT_LOGS

# Windows 控制台默认 GBK 编码，统一改为 UTF-8 以输出 emoji
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_LOGGERS = {}


def get_logger(
    name: str = "pcb_vision_agent",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """获取（或创建）带控制台 + 文件输出的 Logger（单例缓存）"""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_to_file:
        OUTPUT_LOGS.mkdir(parents=True, exist_ok=True)
        log_file = OUTPUT_LOGS / f"app_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def info(msg: str) -> None:
    """便捷日志：INFO 级别"""
    get_logger().info(msg)


def warn(msg: str) -> None:
    """便捷日志：WARNING 级别"""
    get_logger().warning(msg)


def error(msg: str) -> None:
    """便捷日志：ERROR 级别"""
    get_logger().error(msg)
