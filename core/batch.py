"""
批量检测 Batch
==============
对文件夹内所有图片执行检测，汇总生成 Excel/JSON 报表，异常工件图归档。
由项目早期 web_app.py 的 batch_detect 逻辑重构而来。
"""
import os
from typing import Dict, List, Optional, Tuple

import cv2

from . import vision_toolkit as vt
from .detector import PcbDetector
from .report import generate_excel_report, save_json_log
from ..utils.config import OUTPUT_DETECTED
from ..utils.file_utils import ensure_dir, find_images, timestamp_str


def batch_detect(
    folder_path: str,
    conf_threshold: float = 0.5,
    detector: Optional[PcbDetector] = None,
    save_defect_images: bool = True,
) -> Tuple[str, Optional[str], float]:
    """
    批量检测文件夹中的所有图片。
    返回: (汇总文本, Excel报表路径, 良率%)
    """
    if not folder_path or not os.path.isdir(folder_path):
        return "请选择有效的文件夹路径", None, 0.0

    image_files = find_images(folder_path)
    if not image_files:
        return "文件夹中没有找到图片", None, 0.0

    if detector is None:
        detector = PcbDetector(conf_threshold=conf_threshold)

    results: List[Dict] = []
    total = len(image_files)
    print(f"\n📦 开始批量检测，共 {total} 张图片...")

    for idx, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        print(f" [{idx}/{total}] 检测: {filename}")
        det = detector.detect_file(img_path)
        row = det.to_dict()
        row["filename"] = filename
        results.append(row)

        # 异常工件图归档
        if save_defect_images and row["total_defects"] > 0:
            img = cv2.imread(img_path)
            annotated = detector.annotate(img, det.detections)
            ensure_dir(OUTPUT_DETECTED)
            out = str(OUTPUT_DETECTED / f"defect_{timestamp_str()}_{idx}.jpg")
            cv2.imwrite(out, annotated)

    # 生成 Excel 报表 + JSON 日志
    report_path = generate_excel_report(results)
    save_json_log(results, "batch")

    pass_count = sum(1 for r in results if r["total_defects"] == 0)
    fail_count = total - pass_count
    pass_rate = pass_count / total * 100

    summary = f"""
## 📊 批量检测完成
### 统计汇总
- 检测总数：**{total}** 张
- 合格数量：**{pass_count}** 张
- 不合格数量：**{fail_count}** 张
- 良率：**{pass_rate:.2f}%**
### 报表
Excel质检报告：`{report_path}`
JSON日志已保存至 output/logs/
"""
    return summary, report_path, pass_rate
