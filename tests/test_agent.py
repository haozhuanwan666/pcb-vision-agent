"""
VisionAgent 单元测试
====================
覆盖：测试场景生成 / 工具注册 / simulate 决策流程 / 报告生成 / 决策轨迹。
"""
import os

from pcb_vision_agent.core.agent import (
    LANGCHAIN_AVAILABLE,
    VISION_AGENT_SYSTEM_PROMPT,
    VisionAgent,
    generate_test_scenarios,
    get_all_tools,
)


# ------------------------------------------------------------
# 测试场景生成
# ------------------------------------------------------------
def test_generate_test_scenarios(test_scenarios):
    assert set(test_scenarios.keys()) == {"正常质量", "过曝", "低对比度"}
    for path in test_scenarios.values():
        assert os.path.exists(path)


# ------------------------------------------------------------
# 工具注册
# ------------------------------------------------------------
def test_get_all_tools():
    tools = get_all_tools()
    if LANGCHAIN_AVAILABLE:
        assert len(tools) == 6
        names = {t.name for t in tools}
        assert "assess_image_quality" in names
        assert "enhance_contrast" in names
        assert "correct_overexposure" in names
        assert "yolo_detect" in names
        assert "traditional_detect" in names
        assert "subpixel_detect" in names
    else:
        assert tools == []


def test_system_prompt_content():
    assert "assess_image_quality" in VISION_AGENT_SYSTEM_PROMPT
    assert "yolo_detect" in VISION_AGENT_SYSTEM_PROMPT
    assert "subpixel_detect" in VISION_AGENT_SYSTEM_PROMPT


# ------------------------------------------------------------
# Agent 初始化
# ------------------------------------------------------------
def test_agent_init_simulate():
    agent = VisionAgent(mode="simulate")
    assert agent.mode == "simulate"
    assert agent.action_history == []


# ------------------------------------------------------------
# simulate 决策流程
# ------------------------------------------------------------
def test_agent_run_simulate(test_scenarios):
    agent = VisionAgent(mode="simulate")
    result = agent.run(test_scenarios["正常质量"])
    assert "steps" in result
    assert "final_report" in result
    report = result["final_report"]
    assert "quality_level" in report
    assert report["verdict"] in ("合格", "不合格")
    # 第一步必须是质量评估
    assert result["steps"][0]["action"] == "assess_image_quality"


def test_agent_low_contrast_triggers_enhance(test_scenarios):
    """低对比度场景应触发 enhance_contrast 步骤"""
    agent = VisionAgent(mode="simulate")
    result = agent.run(test_scenarios["低对比度"])
    actions = [s["action"] for s in result["steps"]]
    assert "enhance_contrast" in actions


def test_agent_overexposure_triggers_correct(test_scenarios):
    """过曝场景应触发 correct_overexposure 步骤"""
    agent = VisionAgent(mode="simulate")
    result = agent.run(test_scenarios["过曝"])
    actions = [s["action"] for s in result["steps"]]
    assert "correct_overexposure" in actions


def test_agent_decision_trace(test_scenarios):
    agent = VisionAgent(mode="simulate")
    agent.run(test_scenarios["低对比度"])
    trace = agent.get_decision_trace()
    assert isinstance(trace, list)
    assert len(trace) >= 1
    assert trace[0][0] == "assess_image_quality"


# ------------------------------------------------------------
# 报告生成
# ------------------------------------------------------------
def test_generate_report_verdict():
    agent = VisionAgent(mode="simulate")
    quality = {"quality_level": "good", "issues": []}
    # 无缺陷 → 合格
    report = agent._generate_report(quality, [], enhanced=False)
    assert report["verdict"] == "合格"
    assert report["total_defects"] == 0
    assert report["defect_by_class"] == {}
    # 有缺陷 → 不合格 + 按类统计
    dets = [{"class_name": "short"}, {"class_name": "short"}, {"class_name": "open"}]
    report2 = agent._generate_report(quality, dets, enhanced=True)
    assert report2["verdict"] == "不合格"
    assert report2["defect_by_class"] == {"short": 2, "open": 1}
    assert report2["image_enhanced"] is True
    assert "timestamp" in report2
