"""
报表模块测试
============
覆盖：Excel 质检报表 / JSON 日志落盘。
"""
import json
import os

from pcb_vision_agent.core.report import generate_excel_report, save_json_log


def sample_results():
    return [
        {
            "filename": "a.jpg",
            "quality_level": "good",
            "total_defects": 0,
            "defect_by_class": {},
            "verdict": "合格",
        },
        {
            "filename": "b.jpg",
            "quality_level": "poor",
            "total_defects": 2,
            "defect_by_class": {"short": 2},
            "verdict": "不合格",
        },
    ]


def test_excel_report(tmp_path):
    out = str(tmp_path / "report.xlsx")
    path = generate_excel_report(sample_results(), out)
    assert path == out
    assert os.path.exists(out)
    # 文件非空且为有效 xlsx（zip 魔数 PK）
    with open(out, "rb") as f:
        assert f.read(2) == b"PK"


def test_json_log(tmp_path):
    path = save_json_log(sample_results(), "test_batch")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["verdict"] == "合格"
