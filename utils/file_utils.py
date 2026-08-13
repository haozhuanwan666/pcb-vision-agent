"""
文件与通用工具
==============
目录创建、时间戳、图片查找/读写等通用函数。
"""
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import IMAGE_EXTENSIONS


def ensure_dir(path) -> str:
    """确保目录存在并返回路径字符串"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def timestamp_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """当前时间戳字符串"""
    return datetime.now().strftime(fmt)


def find_images(
    folder: str, extensions: Tuple[str, ...] = IMAGE_EXTENSIONS
) -> List[str]:
    """查找文件夹内（非递归）所有图片，按文件名排序返回绝对/相对路径列表"""
    if not os.path.isdir(folder):
        return []
    files = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(extensions):
            files.append(os.path.join(folder, f))
    return files


def read_image(path: str) -> Optional[np.ndarray]:
    """安全读取图片，失败返回 None（BGR 格式）"""
    return cv2.imread(path)


def save_image(path: str, image: np.ndarray) -> bool:
    """保存图片并确保父目录存在"""
    ensure_dir(os.path.dirname(path) or ".")
    return bool(cv2.imwrite(path, image))


def load_json(path: str):
    """读取 JSON 文件"""
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
