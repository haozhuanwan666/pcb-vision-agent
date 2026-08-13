"""
检测引擎 Detector
=================
提供统一的高层检测接口，融合传统视觉 + YOLO + Halcon 亚像素，
自动做图像质量评估与增强，屏蔽底层算子细节。
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import vision_toolkit as vt
from ..utils.config import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, DEFAULT_MIN_AREA
from ..utils.logger import get_logger

logger = get_logger("pcb_vision_agent.detector")


@dataclass
class DetectionResult:
    """单张图片的完整检测结果"""

    image_path: str = ""
    quality: Dict = field(default_factory=dict)
    detections: List[Dict] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    enhanced: bool = False
    verdict: str = "合格"
    timestamp: str = ""

    def to_dict(self) -> Dict:
        """转为可序列化/入库的结构化字典"""
        class_stats: Dict[str, int] = {}
        for d in self.detections:
            cls = d.get("class_name", "未知")
            class_stats[cls] = class_stats.get(cls, 0) + 1
        return {
            "image_path": self.image_path,
            "quality_level": self.quality.get("quality_level", "good"),
            "quality_issues": self.quality.get("issues", []),
            "total_defects": len(self.detections),
            "defect_by_class": class_stats,
            "methods": self.methods,
            "enhanced": self.enhanced,
            "verdict": self.verdict,
            "timestamp": self.timestamp,
        }


class PcbDetector:
    """
    PCB 缺陷检测引擎
    集质量评估 / 图像增强 / 传统检测 / YOLO / Halcon 亚像素于一体。
    """

    def __init__(
        self,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        min_area: float = DEFAULT_MIN_AREA,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area = min_area

    # ---------- 单步算子封装 ----------
    def assess_quality(self, image: np.ndarray) -> Dict:
        """图像质量评估"""
        return vt.assess_image_quality(image)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        """
        根据质量评估自动预处理（过曝校正/对比度增强）
        返回: (处理后的图, 是否增强, 质量报告)
        """
        quality = vt.assess_image_quality(image)
        processed = image
        enhanced = False
        if "过曝" in quality["issues"]:
            processed = vt.correct_overexposure(processed)
            enhanced = True
        if "低对比度" in quality["issues"]:
            processed = vt.enhance_contrast(processed)
            enhanced = True
        return processed, enhanced, quality

    def detect_traditional(self, image: np.ndarray) -> List[Dict]:
        """传统视觉缺陷检测（阈值分割 + 形态学 + 轮廓）"""
        binary = vt.threshold_segment(image, "otsu")
        cleaned = vt.morphological_process(binary, "open", 3)
        return vt.find_defect_contours(cleaned, self.min_area)

    def detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """YOLO 深度学习缺陷检测"""
        return vt.yolo_detect(
            image, conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold
        )

    def detect_subpixel(self, image: np.ndarray) -> Optional[List[Dict]]:
        """Halcon 亚像素边缘检测（无 Halcon 时返回 None）"""
        return vt.subpixel_edge_detect(image)

    # ---------- 组合流水线 ----------
    def detect(self, image: np.ndarray, methods: Tuple[str, ...] = ("yolo", "traditional")) -> DetectionResult:
        """
        执行完整检测流水线：
        1. 质量评估 + 自动增强
        2. 按 methods 指定的方法检测（yolo / traditional / subpixel）
        3. 汇总判定结果
        """
        processed, enhanced, quality = self.preprocess(image)

        detections: List[Dict] = []
        used: List[str] = []
        if "yolo" in methods:
            detections = self.detect_yolo(processed)
            used.append("yolo")
        if "traditional" in methods and not detections:
            trad = self.detect_traditional(processed)
            if trad:
                detections = trad
                used.append("traditional")
        if "subpixel" in methods:
            edges = self.detect_subpixel(processed)
            if edges:
                used.append("subpixel")

        return DetectionResult(
            quality=quality,
            detections=detections,
            methods=used,
            enhanced=enhanced,
            verdict="合格" if len(detections) == 0 else "不合格",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def detect_file(
        self, image_path: str, methods: Tuple[str, ...] = ("yolo", "traditional")
    ) -> DetectionResult:
        """从文件读取图片并执行完整检测流水线"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        result = self.detect(image, methods)
        result.image_path = image_path
        return result

    def annotate(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果框。
        自动识别来源：YOLO 结果（含 class_id）用 YOLO 样式，
        传统视觉结果（无 class_id）用轮廓框样式。
        """
        if detections and "class_id" in detections[0]:
            return vt.draw_yolo_detections(image, detections)
        return vt.draw_defects(image, detections)
