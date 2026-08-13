"""
批量检测测试
============
覆盖：无效文件夹 / 正常批量检测（生成 Excel 报表）。
"""
import os

from pcb_vision_agent.core.batch import batch_detect


def test_batch_detect_invalid_folder():
    summary, report, rate = batch_detect("no_such_folder")
    assert "请选择有效的文件夹路径" in summary
    assert report is None
    assert rate == 0.0


def test_batch_detect_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    summary, report, rate = batch_detect(str(empty))
    assert "没有找到图片" in summary
    assert report is None


def test_batch_detect_small(test_scenarios):
    """用测试场景所在文件夹跑一次完整批量检测"""
    folder = os.path.dirname(test_scenarios["正常质量"])
    summary, report_path, pass_rate = batch_detect(folder, conf_threshold=0.5)
    assert "批量检测完成" in summary
    assert 0.0 <= pass_rate <= 100.0
    assert report_path is not None
    assert os.path.exists(report_path)
