"""
视觉算子工具箱 Vision Toolkit（core 包版本）
============================================
由项目早期 vision_toolkit.py 重构迁移而来，供 Agent 与检测引擎调用。
所有函数统一输入输出规范：输入 BGR numpy 图像，输出结构化结果。

函数清单（14 个）：
- 质量评估 : assess_image_quality
- 图像增强 : enhance_contrast / denoise_image / correct_overexposure / gamma_correction
- 传统检测 : threshold_segment / morphological_process / find_defect_contours / draw_defects
- YOLO     : load_yolo_model / yolo_detect / draw_yolo_detections
- Halcon   : subpixel_edge_detect / measure_defect_size_subpixel
"""
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.config import get_yolo_weights

# ============================================================
# 第一部分：图像质量评估
# ============================================================
def assess_image_quality(image: np.ndarray) -> Dict:
    """
    评估图像质量（亮度、对比度、清晰度）
    输入: BGR图像
    输出: 质量评估字典
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 亮度：像素平均值
    brightness = np.mean(gray)
    # 对比度：像素标准差
    contrast = np.std(gray)
    # 清晰度：拉普拉斯方差
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    # 质量等级判断
    quality = "good"
    issues = []
    if brightness > 220:
        quality = "poor"
        issues.append("过曝")
    elif brightness < 40:
        quality = "poor"
        issues.append("过暗")
    if contrast < 20:
        quality = "poor"
        issues.append("低对比度")
    if sharpness < 100:
        quality = "poor"
        issues.append("模糊")
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "quality_level": quality,
        "issues": issues,
    }


# ============================================================
# 第二部分：图像增强工具
# ============================================================
def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    CLAHE对比度增强（处理反光、低对比度场景）
    输入: BGR图像
    输出: 增强后的BGR图像
    """
    # 转LAB空间，只增强L通道（不改变颜色）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def denoise_image(image: np.ndarray, method: str = "gaussian", kernel: int = 5) -> np.ndarray:
    """
    图像去噪（多种模式）
    method: gaussian(高斯) / median(中值) / bilateral(双边) / nlmeans(非局部均值)
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    elif method == "median":
        return cv2.medianBlur(image, kernel)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, kernel, 75, 75)
    elif method == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        return image


def correct_overexposure(image: np.ndarray) -> np.ndarray:
    """
    过曝校正（降低高光区域亮度）
    """
    # 转HSV降低V通道
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 高光区域压缩
    v = v.astype(np.float32)
    mask = v > 200
    v[mask] = 200 + (v[mask] - 200) * 0.5
    v = np.clip(v, 0, 255).astype(np.uint8)
    hsv_corrected = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)
    return result


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma校正（调节整体亮度）
    gamma < 1: 变亮; gamma > 1: 变暗
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


# ============================================================
# 第三部分：传统缺陷检测工具
# ============================================================
def threshold_segment(
    image: np.ndarray, method: str = "adaptive", thresh_value: int = 80
) -> np.ndarray:
    """
    阈值分割
    method: global(全局) / adaptive(自适应) / otsu
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "global":
        _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = gray
    return binary


def morphological_process(
    binary: np.ndarray, operation: str = "open", kernel_size: int = 3
) -> np.ndarray:
    """
    形态学操作
    operation: open(开运算) / close(闭运算) / dilate(膨胀) / erode(腐蚀)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "open":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif operation == "dilate":
        return cv2.dilate(binary, kernel, iterations=1)
    elif operation == "erode":
        return cv2.erode(binary, kernel, iterations=1)
    else:
        return binary


def find_defect_contours(binary: np.ndarray, min_area: float = 10.0) -> List[Dict]:
    """
    提取缺陷轮廓，返回缺陷列表
    每个缺陷: {area, bbox, center, contour}
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            defects.append(
                {
                    "area": round(area, 2),
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "contour": cnt,
                }
            )
    # 按面积从大到小排序
    defects.sort(key=lambda x: x["area"], reverse=True)
    return defects


def draw_defects(image: np.ndarray, defects: List[Dict]) -> np.ndarray:
    """
    在图像上绘制缺陷标注
    """
    result = image.copy()
    for i, d in enumerate(defects, 1):
        x, y, w, h = d["bbox"]
        # 矩形框
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 标签
        label = f"#{i} {d['area']:.0f}px"
        cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 中心点
        cv2.circle(result, d["center"], 3, (0, 255, 255), -1)
    return result


# ============================================================
# 第四部分：YOLO深度学习检测模块
# ============================================================
# 全局模型缓存（单例，避免重复加载）
_yolo_model = None


def load_yolo_model(weights_path: Optional[str] = None):
    """
    加载YOLO模型（单例模式，避免重复加载）
    weights_path 为空时自动从配置解析可用权重。
    """
    global _yolo_model
    if weights_path is None:
        weights_path = get_yolo_weights()
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO

        if os.path.exists(weights_path):
            _yolo_model = YOLO(weights_path)
            print(f"✅ YOLO模型加载成功: {weights_path}")
        else:
            print(f"⚠️ 权重文件不存在: {weights_path}")
            print("  将自动使用官方预训练权重（仅演示用）")
            _yolo_model = YOLO(get_yolo_weights())
    except ImportError:
        print("❌ ultralytics未安装")
        return None
    return _yolo_model


def yolo_detect(
    image: np.ndarray, conf_threshold: float = 0.5, iou_threshold: float = 0.45
) -> List[Dict]:
    """
    YOLO缺陷检测
    返回: 检测结果列表 [{class_name, confidence, bbox, area}]
    """
    model = load_yolo_model()
    if model is None:
        return []
    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            class_name = model.names[cls_id]
            area = (x2 - x1) * (y2 - y1)
            detections.append(
                {
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "bbox": (x1, y1, x2 - x1, y2 - y1),  # x, y, w, h
                    "area": area,
                }
            )
    return detections


def draw_yolo_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    绘制YOLO检测结果
    """
    result = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
    for det in detections:
        x, y, w, h = det["bbox"]
        color = colors[det["class_id"] % len(colors)]
        # 矩形框
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        # 标签背景
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x, y - th - 5), (x + tw, y), color, -1)
        # 标签文字
        cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return result


# ============================================================
# 第五部分：Halcon亚像素测量模块（可选）
# ============================================================
_HALCON_AVAILABLE = False
try:
    import halcon as h

    _HALCON_AVAILABLE = True
except ImportError:
    pass


def halcon_available() -> bool:
    """是否安装了 Halcon"""
    return _HALCON_AVAILABLE


def subpixel_edge_detect(
    image: np.ndarray, sigma: float = 1.0, threshold: float = 20.0
) -> Optional[List[Dict]]:
    """
    Halcon亚像素边缘检测（微小缺陷测量）
    无Halcon环境返回None，自动降级使用OpenCV。
    """
    if not _HALCON_AVAILABLE:
        print("ℹ️ Halcon不可用，使用OpenCV替代")
        return None
    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)
    # 创建Halcon图像（传内存指针）
    h_img = h.gen_image1("byte", gray.shape[1], gray.shape[0], gray.ctypes.data)
    # 亚像素边缘提取
    edges = h.edges_sub_pix(h_img, "canny", sigma, threshold, 20)
    # 提取边缘特征
    edges_list = []
    if h.count_obj(edges) > 0:
        for i in range(1, h.count_obj(edges) + 1):
            edge_i = h.select_obj(edges, i)
            length = h.length_xld(edge_i)[0]
            edges_list.append(
                {"edge_id": i, "length_pixels": round(length, 3), "type": "subpixel"}
            )
    return edges_list


def measure_defect_size_subpixel(
    image: np.ndarray, defect_bbox: Tuple[int, int, int, int]
) -> Dict:
    """
    亚像素精度测量缺陷尺寸
    """
    if not _HALCON_AVAILABLE:
        # 降级：用OpenCV普通像素测量
        x, y, w, h = defect_bbox
        return {
            "width_pixels": w,
            "height_pixels": h,
            "area_pixels": w * h,
            "precision": "pixel_level",
        }
    # Halcon亚像素测量逻辑（完整实现可按需补充）
    return {"precision": "subpixel"}
