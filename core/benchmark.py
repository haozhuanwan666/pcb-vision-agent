"""
性能测试 Benchmark
==================
量化评估：推理速度、检测准确率、传统 vs YOLO 对比。
由项目早期 web_app.py 的 run_performance_test 逻辑重构而来。
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2

from . import vision_toolkit as vt
from .detector import PcbDetector
from ..utils.config import WORKSPACE_ROOT
from ..utils.file_utils import ensure_dir, find_images
from .report import save_json_report


def run_performance_test(
    dataset_path: Optional[str] = None,
    sample_for_compare: int = 20,
    conf_threshold: float = 0.5,
) -> Optional[Dict]:
    """
    性能测试：推理速度 / 准确率 / 传统 vs YOLO 对比。
    dataset_path 默认使用工作区的 pcb_dataset/images/val。
    返回性能报告字典，并保存 JSON 到 output/reports。
    """
    if dataset_path is None:
        dataset_path = str(WORKSPACE_ROOT / "pcb_dataset" / "images" / "val")

    print("\n" + "=" * 60)
    print("⚡ 性能量化测试")
    print("=" * 60)

    if not os.path.isdir(dataset_path):
        print(f"⚠️ 测试数据集不存在: {dataset_path}")
        return None

    image_files = find_images(dataset_path)
    if not image_files:
        print("❌ 没有找到测试图片")
        return None

    print(f"\n测试图片数量: {len(image_files)} 张")

    # ===== 1. 推理速度测试 =====
    print("\n【1. 推理速度测试】")
    times: List[float] = []
    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)
        start = time.time()
        _ = vt.yolo_detect(img, conf_threshold=conf_threshold)
        times.append(time.time() - start)
    avg_time = sum(times) / len(times) * 1000  # 转毫秒
    fps = 1000 / avg_time if avg_time > 0 else 0
    print(f" 平均推理时间: {avg_time:.2f} ms/张")
    print(f" 推理速度: {fps:.1f} FPS")

    # ===== 2. 准确率统计 =====
    print("\n【2. 检测准确率统计】")
    total_detections = 0
    high_conf_count = 0
    confidences: List[float] = []
    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)
        detections = vt.yolo_detect(img, conf_threshold=0.25)
        for d in detections:
            total_detections += 1
            confidences.append(d["confidence"])
            if d["confidence"] >= 0.8:
                high_conf_count += 1
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    high_conf_ratio = high_conf_count / total_detections * 100 if total_detections else 0
    print(f" 总检测数: {total_detections}")
    print(f" 平均置信度: {avg_conf:.4f}")
    print(f" 高置信度占比(≥0.8): {high_conf_ratio:.1f}%")

    # ===== 3. 传统方法 vs YOLO 对比（抽样） =====
    print("\n【3. 传统视觉 vs YOLO 对比】")
    detector = PcbDetector(conf_threshold=conf_threshold)
    trad_counts: List[int] = []
    yolo_counts: List[int] = []
    for img_file in image_files[:sample_for_compare]:
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)
        trad_counts.append(len(detector.detect_traditional(img)))
        yolo_counts.append(len(detector.detect_yolo(img)))
    avg_trad = sum(trad_counts) / len(trad_counts) if trad_counts else 0
    avg_yolo = sum(yolo_counts) / len(yolo_counts) if yolo_counts else 0
    print(f" 传统方法平均检出: {avg_trad:.1f} 个/张")
    print(f" YOLO方法平均检出: {avg_yolo:.1f} 个/张")

    # ===== 4. 生成性能报告 =====
    performance_report: Dict = {
        "test_images": len(image_files),
        "inference": {"avg_time_ms": round(avg_time, 2), "fps": round(fps, 1)},
        "accuracy": {
            "total_detections": total_detections,
            "avg_confidence": round(avg_conf, 4),
            "high_conf_ratio_pct": round(high_conf_ratio, 1),
        },
        "comparison": {
            "traditional_avg": round(avg_trad, 1),
            "yolo_avg": round(avg_yolo, 1),
        },
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_file = save_json_report(performance_report, "性能测试报告")
    print(f"\n✅ 性能测试完成")
    print(f" 报告已保存: {report_file}")
    return performance_report
