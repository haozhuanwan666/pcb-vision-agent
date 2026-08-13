"""
PcbDetector 检测引擎测试
=======================
覆盖：质量评估 / 自动预处理 / 完整流水线 / 文件读取异常 / 绘制分流。
"""
import cv2
import numpy as np
import pytest

from pcb_vision_agent.core.detector import DetectionResult, PcbDetector


@pytest.fixture(scope="module")
def detector():
    return PcbDetector(conf_threshold=0.5)


@pytest.fixture(scope="module")
def sample_image():
    """模拟一张含缺陷的低对比度图像"""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 110
    cv2.circle(img, (100, 100), 15, (70, 70, 70), -1)
    return img


def test_quality_assessment(detector, sample_image):
    quality = detector.assess_quality(sample_image)
    assert "quality_level" in quality
    assert "issues" in quality
    assert isinstance(quality["brightness"], float)
    assert isinstance(quality["contrast"], float)


def test_preprocess(detector, sample_image):
    processed, enhanced, quality = detector.preprocess(sample_image)
    assert processed.shape == sample_image.shape
    assert isinstance(enhanced, bool)
    assert quality["quality_level"] in ("good", "poor")


def test_detect_returns_result(detector, sample_image):
    result = detector.detect(sample_image, methods=("traditional",))
    assert isinstance(result, DetectionResult)
    assert result.verdict in ("合格", "不合格")
    data = result.to_dict()
    assert "quality_level" in data
    assert "total_defects" in data
    assert "verdict" in data


def test_detect_file(detector, test_scenarios):
    result = detector.detect_file(test_scenarios["正常质量"])
    assert result.image_path == test_scenarios["正常质量"]
    assert result.verdict in ("合格", "不合格")


def test_detect_file_missing(detector):
    with pytest.raises(FileNotFoundError):
        detector.detect_file("no_such_file.jpg")


def test_annotate_yolo_vs_traditional(detector):
    """回归测试：传统检测结果（无 class_id）不能走 draw_yolo_detections"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 120
    # 传统视觉缺陷（无 class_id）
    trad = [
        {
            "class_name": "缺陷",
            "bbox": (10, 10, 20, 20),
            "area": 400.0,
            "center": (20, 20),
            "contour": None,
        }
    ]
    out = detector.annotate(img, trad)
    assert out.shape == img.shape
    # YOLO 缺陷（含 class_id）
    yolo = [
        {
            "class_name": "short",
            "class_id": 1,
            "confidence": 0.9,
            "bbox": (10, 10, 20, 20),
            "area": 400,
        }
    ]
    out2 = detector.annotate(img, yolo)
    assert out2.shape == img.shape
