"""
传统固定参数 vs Agent 自适应 —— 检测对比实验
============================================
在验证集上对比两条流水线，用 Ground Truth 统计真实指标：
    传统固定参数 : 固定全局阈值 + 形态学 + 轮廓（不随图像质量调整）
    Agent 自适应  : 质量评估 → 过曝/低对比度自动增强 → YOLO 主检测 → 无检出时传统兜底

评估口径（IoU≥0.5 匹配，类别无关）：
    召回率/漏检率 / 精确率/误检 / F1

用法:
    python scripts/compare_pipelines.py                 # 全量验证集
    python scripts/compare_pipelines.py --limit 100     # 只看前 100 张（快速验证）
    python scripts/compare_pipelines.py --conf 0.5      # 指定 YOLO 置信度

输出:
    控制台汇总 + output/reports/对比实验_时间戳.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

import cv2

# 脚本位于 scripts/ 子目录，包父目录需要上溯 3 级（scripts -> 包 -> 工作区）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_vision_agent.core import vision_toolkit as vt  # noqa: E402
from pcb_vision_agent.utils.config import (  # noqa: E402
    WORKSPACE_ROOT,
)
from pcb_vision_agent.utils.file_utils import ensure_dir  # noqa: E402
from pcb_vision_agent.utils.logger import get_logger  # noqa: E402

logger = get_logger("compare")

DEFAULT_DATASET = str(WORKSPACE_ROOT / "pcb_dataset")
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]


# ------------------------------------------------------------
# Ground Truth 读取
# ------------------------------------------------------------
def load_gt(label_path: str, img_w: int, img_h: int):
    """
    读取 YOLO 格式标注: 每行 `class cx cy w h`（归一化）
    返回: [(class, x, y, w, h)]（像素坐标）
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:5])
            x = (cx - w / 2) * img_w
            y = (cy - h / 2) * img_h
            boxes.append((cls, x, y, w * img_w, h * img_h))
    return boxes


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def evaluate(det_boxes, gt_boxes, iou_thr: float = 0.5):
    """逐框 IoU≥0.5 匹配，返回 (tp, fp, fn)"""
    tp = fp = 0
    matched = [False] * len(gt_boxes)
    for d in det_boxes:
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gt_boxes):
            if matched[j]:
                continue
            v = iou(d, g[1:])
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            tp += 1
            matched[best_j] = True
        else:
            fp += 1
    fn = len(gt_boxes) - sum(matched)
    return tp, fp, fn


# ------------------------------------------------------------
# 两条流水线
# ------------------------------------------------------------
def pipeline_traditional_fixed(img):
    """
    固定参数传统视觉：Otsu 阈值 + 开运算 + 轮廓（固定算法，不做质量评估/增强/自适应）。
    作为"传统固定参数方案"的可信基线。
    """
    binary = vt.threshold_segment(img, "otsu")
    cleaned = vt.morphological_process(binary, "open", 3)
    defects = vt.find_defect_contours(cleaned, min_area=30)
    return [(d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]) for d in defects]


def pipeline_agent_adaptive(img, conf_threshold: float):
    """Agent 自适应：质量评估 → 自动增强 → YOLO 主检测 → 无检出传统兜底"""
    quality = vt.assess_image_quality(img)
    processed = img
    if "过曝" in quality["issues"]:
        processed = vt.correct_overexposure(processed)
    if "低对比度" in quality["issues"]:
        processed = vt.enhance_contrast(processed)
    dets = vt.yolo_detect(processed, conf_threshold=conf_threshold)
    boxes = [(d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]) for d in dets]
    if not boxes:
        binary = vt.threshold_segment(processed, "otsu")
        cleaned = vt.morphological_process(binary, "open", 3)
        trad = vt.find_defect_contours(cleaned, min_area=30)
        boxes = [(d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]) for d in trad]
    return boxes


# ------------------------------------------------------------
# 指标
# ------------------------------------------------------------
def summarize(tp, fp, fn):
    total = tp + fn
    return {
        "gt_total": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / total if total else 0.0,          # 召回率
        "miss_rate": fn / total if total else 0.0,       # 漏检率
        "precision": tp / (tp + fp) if tp + fp else 0.0, # 精确率
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="传统固定参数 vs Agent 自适应 对比实验")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="数据集根目录（含 images/val, labels/val）")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO 置信度阈值")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 张（0=全部）")
    args = ap.parse_args()

    img_dir = os.path.join(args.dataset, "images", "val")
    lab_dir = os.path.join(args.dataset, "labels", "val")
    if not os.path.isdir(img_dir):
        print(f"❌ 验证集不存在: {img_dir}")
        return

    images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".bmp")))
    if args.limit > 0:
        images = images[: args.limit]
    print(f"📁 测试图片: {len(images)} 张")

    agg_a = [0, 0, 0]  # tp, fp, fn
    agg_b = [0, 0, 0]
    for idx, name in enumerate(images, 1):
        img_path = os.path.join(img_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        base = os.path.splitext(name)[0]
        gt_boxes = load_gt(os.path.join(lab_dir, base + ".txt"), w, h)

        det_a = pipeline_traditional_fixed(img)
        det_b = pipeline_agent_adaptive(img, args.conf)
        ta = evaluate(det_a, gt_boxes)
        tb = evaluate(det_b, gt_boxes)
        for i in range(3):
            agg_a[i] += ta[i]
            agg_b[i] += tb[i]
        if idx % 100 == 0:
            print(f"  [{idx}/{len(images)}] 已处理")

    res_a = summarize(*agg_a)
    res_b = summarize(*agg_b)
    miss_drop = (res_a["miss_rate"] - res_b["miss_rate"]) / res_a["miss_rate"] * 100 if res_a["miss_rate"] > 0 else 0.0

    print("\n" + "=" * 64)
    print("📊 对比实验结果")
    print("=" * 64)
    for label, r in (("传统固定参数", res_a), ("Agent 自适应", res_b)):
        print(f"\n【{label}】")
        print(f"  GT缺陷总数: {r['gt_total']}  检出TP: {r['tp']}  误检FP: {r['fp']}  漏检FN: {r['fn']}")
        print(f"  召回率: {r['recall']*100:.2f}%   漏检率: {r['miss_rate']*100:.2f}%")
        print(f"  精确率: {r['precision']*100:.2f}%   F1: {r['f1']:.4f}")
    print(f"\n🎯 漏检率降低: {miss_drop:.1f}%")

    report = {
        "test_images": len(images),
        "conf_threshold": args.conf,
        "traditional_fixed": res_a,
        "agent_adaptive": res_b,
        "miss_rate_drop_pct": round(miss_drop, 2),
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ensure_dir(OUTPUT_DIR := os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "reports"))
    out = os.path.join(OUTPUT_DIR, f"对比实验_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 报告已保存: {out}")


if __name__ == "__main__":
    main()
