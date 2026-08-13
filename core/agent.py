"""
视觉决策 Agent Vision Agent（core 包版本）
==========================================
由项目早期 vision_agent.py 重构迁移而来。
基于 LangChain 的智能视觉检测 Agent：
自主判断图像质量，自动选择处理算法。
双模式：本地大模型模式(llm) / 规则模拟模式(simulate)。
"""
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Type

import cv2
import numpy as np

from . import vision_toolkit as vt
from ..utils.config import (
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    WORKSPACE_ROOT,
)
from ..utils.logger import get_logger

logger = get_logger("pcb_vision_agent.agent")

# ============================================================
# 第一部分：LangChain Tool 封装
# ============================================================
try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain未安装，使用模拟模式")

    # 模拟模式：提供占位基类，保证模块在无LangChain时仍可导入
    class BaseModel:
        """占位：模拟模式下代替 pydantic 的 BaseModel"""

    class BaseTool:
        """占位：模拟模式下代替 LangChain 的 BaseTool"""

        name = ""
        description = ""
        args_schema = None

        def _run(self, *args, **kwargs):
            raise NotImplementedError

    def Field(*args, **kwargs):
        """占位：模拟模式下代替 pydantic 的 Field"""
        return None


class VisionBaseTool(BaseTool):
    """
    所有视觉工具的公共基类。
    兼容 ReAct 文本模式：模型输出的 Action Input 可能是 JSON 字符串
    （如 '{"image_path": "a.jpg"}'），这里先解析成 dict 再交给 args_schema 校验。
    """

    def _parse_input(self, tool_input, tool_call_id=None):
        if isinstance(tool_input, str):
            stripped = tool_input.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    tool_input = json.loads(stripped)
                except json.JSONDecodeError:
                    pass
        return super()._parse_input(tool_input, tool_call_id)


# 工具1：图像质量评估
class ImageQualityInput(BaseModel):
    image_path: str = Field(description="待评估的图片文件路径")


class AssessQualityTool(VisionBaseTool):
    name: str = "assess_image_quality"
    description: str = "评估图像质量，返回亮度、对比度、清晰度和存在的问题（过曝/过暗/低对比度/模糊）"
    args_schema: Type[BaseModel] = ImageQualityInput

    def _run(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return f"错误：无法读取图片 {image_path}"
        quality = vt.assess_image_quality(img)
        return json.dumps(quality, ensure_ascii=False)


# 工具2：图像增强（对比度增强）
class EnhanceInput(BaseModel):
    image_path: str = Field(description="待增强的图片路径")
    clip_limit: float = Field(default=2.0, description="CLAHE裁剪阈值，默认2.0")


class EnhanceContrastTool(VisionBaseTool):
    name: str = "enhance_contrast"
    description: str = "对图像进行CLAHE对比度增强，处理低对比度、反光场景"
    args_schema: Type[BaseModel] = EnhanceInput

    def _run(self, image_path: str, clip_limit: float = 2.0) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return "错误：无法读取图片"
        result = vt.enhance_contrast(img, clip_limit)
        output_path = image_path.replace(".", "_enhanced.")
        cv2.imwrite(output_path, result)
        return f"对比度增强完成，保存至：{output_path}"


# 工具3：过曝校正
class OverexposureInput(BaseModel):
    image_path: str = Field(description="待校正的图片路径")


class CorrectOverexposureTool(VisionBaseTool):
    name: str = "correct_overexposure"
    description: str = "校正过曝图像，压缩高光区域亮度"
    args_schema: Type[BaseModel] = OverexposureInput

    def _run(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return "错误：无法读取图片"
        result = vt.correct_overexposure(img)
        output_path = image_path.replace(".", "_corrected.")
        cv2.imwrite(output_path, result)
        return f"过曝校正完成，保存至：{output_path}"


# 工具4：YOLO缺陷检测
class YOLODetectInput(BaseModel):
    image_path: str = Field(description="待检测的图片路径")
    conf_threshold: float = Field(default=0.5, description="置信度阈值，默认0.5")


class YOLODetectTool(VisionBaseTool):
    name: str = "yolo_detect"
    description: str = "使用YOLO深度学习模型检测PCB缺陷，返回检测到的缺陷类别、置信度和位置"
    args_schema: Type[BaseModel] = YOLODetectInput

    def _run(self, image_path: str, conf_threshold: float = 0.5) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return "错误：无法读取图片"
        detections = vt.yolo_detect(img, conf_threshold=conf_threshold)
        result = {
            "total_defects": len(detections),
            "defects": [
                {
                    "class": d["class_name"],
                    "confidence": d["confidence"],
                    "area": d["area"],
                }
                for d in detections
            ],
        }
        return json.dumps(result, ensure_ascii=False)


# 工具5：传统视觉缺陷检测
class TraditionalDetectInput(BaseModel):
    image_path: str = Field(description="待检测的图片路径")
    min_area: float = Field(default=30.0, description="最小缺陷面积阈值")


class TraditionalDetectTool(VisionBaseTool):
    name: str = "traditional_detect"
    description: str = "使用传统机器视觉方法（阈值分割+轮廓提取）检测缺陷，适合规则明显的缺陷"
    args_schema: Type[BaseModel] = TraditionalDetectInput

    def _run(self, image_path: str, min_area: float = 30.0) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return "错误：无法读取图片"
        binary = vt.threshold_segment(img, "otsu")
        cleaned = vt.morphological_process(binary, "open", 3)
        defects = vt.find_defect_contours(cleaned, min_area)
        result = {
            "total_defects": len(defects),
            "method": "traditional_vision",
            "defects": [{"area": d["area"], "center": d["center"]} for d in defects],
        }
        return json.dumps(result, ensure_ascii=False)


# 工具6：亚像素二次检测（微小缺陷）
class SubpixelInput(BaseModel):
    image_path: str = Field(description="待检测的图片路径")


class SubpixelDetectTool(VisionBaseTool):
    name: str = "subpixel_detect"
    description: str = "使用Halcon亚像素边缘检测进行高精度二次检测，适用于微小划痕、细线缺陷"
    args_schema: Type[BaseModel] = SubpixelInput

    def _run(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return "错误：无法读取图片"
        edges = vt.subpixel_edge_detect(img)
        if edges is None:
            return "Halcon不可用，降级为OpenCV边缘检测（像素级精度）"
        result = {
            "method": "subpixel",
            "edge_count": len(edges),
            "edges": edges[:5],  # 只返回前5条
        }
        return json.dumps(result, ensure_ascii=False)


# 获取所有工具列表
def get_all_tools():
    """获取所有注册的视觉工具"""
    if not LANGCHAIN_AVAILABLE:
        return []
    return [
        AssessQualityTool(),
        EnhanceContrastTool(),
        CorrectOverexposureTool(),
        YOLODetectTool(),
        TraditionalDetectTool(),
        SubpixelDetectTool(),
    ]


# ============================================================
# 第二部分：Agent系统提示词
# ============================================================
VISION_AGENT_SYSTEM_PROMPT = """你是一名资深工业视觉检测工程师，负责PCB板缺陷质量检测。
你的工作是：根据图像质量情况，自主选择合适的视觉工具进行检测分析。
## 你的身份
- 专业的机器视觉检测工程师
- 熟悉传统图像处理算法和深度学习检测方法
- 能够根据图像工况智能选择最优处理方案
## 可用工具说明
1. assess_image_quality - 评估图像质量（亮度、对比度、清晰度、存在问题）
2. enhance_contrast - 对比度增强（低对比度、反光场景使用）
3. correct_overexposure - 过曝校正（图像整体过亮时使用）
4. yolo_detect - YOLO深度学习缺陷检测（主检测方法，优先使用）
5. traditional_detect - 传统视觉检测（辅助方法，规则明显的缺陷）
6. subpixel_detect - 亚像素高精度检测（微小缺陷二次确认时使用）
## 工作流程规范
第一步：必须先调用 assess_image_quality 评估图像质量
第二步：根据质量评估结果决定是否需要图像增强
- 如果存在"过曝"问题 → 调用 correct_overexposure
- 如果存在"低对比度"问题 → 调用 enhance_contrast
第三步：调用 yolo_detect 进行主检测
第四步：分析检测结果
- 如果检测到的缺陷置信度都很低（<0.6）→ 考虑调用 subpixel_detect 二次确认
- 如果YOLO未检测到缺陷但图像质量有问题 → 先增强再重新检测
第五步：输出最终检测报告
## 输出格式要求
最终答案必须包含：
1. 图像质量评估结论
2. 使用的检测方法
3. 检测到的缺陷数量和类型
4. 整体质量判定（合格/不合格）
请严格按照ReAct格式思考和行动，不要跳过步骤。
"""


# ============================================================
# 第三部分：Agent核心实现（双模式）
# ============================================================
class VisionAgent:
    """
    视觉决策Agent
    双模式：llm模式（需大模型） / simulate模式（规则模拟，无API也能跑）
    """

    def __init__(self, mode: str = "simulate", llm=None):
        """
        mode: 'llm' 使用LangChain大模型Agent（本地Ollama）
              'simulate' 使用规则模拟Agent（无API Key时使用）
        """
        self.mode = mode
        self.llm = llm
        self.tools = get_all_tools()
        self.action_history = []  # 记录Agent决策过程
        if mode == "llm":
            # 未显式传入模型时，自动连接本地 Ollama
            if self.llm is None:
                self.llm = create_local_llm()
            if self.llm is None:
                print("⚠️ 无法创建本地大模型，自动降级为simulate模式")
                self.mode = "simulate"
        print(f"🤖 视觉Agent初始化完成，运行模式: {self.mode}")

    def run(self, image_path: str) -> Dict:
        """
        执行完整的视觉检测Agent流程
        """
        print(f"\n{'='*60}")
        print(f"🔍 Agent开始检测: {image_path}")
        print(f"{'='*60}")
        self.action_history = []
        current_image = image_path
        if self.mode == "simulate":
            return self._run_simulate(current_image)
        else:
            return self._run_llm(current_image)

    def _run_simulate(self, image_path: str) -> Dict:
        """
        规则模拟模式：按照预设逻辑模拟Agent决策
        无大模型API时使用，用于演示和学习
        """
        result = {"image_path": image_path, "steps": [], "final_report": {}}
        # Step 1: 质量评估
        print("\n🧠 Thought: 首先评估图像质量")
        print("  Action: assess_image_quality")
        img = cv2.imread(image_path)
        quality = vt.assess_image_quality(img)
        print(f"  Observation: {quality}")
        result["steps"].append(
            {"step": 1, "action": "assess_image_quality", "result": quality}
        )
        self.action_history.append(("assess_image_quality", quality))
        current_img = img
        current_path = image_path
        # Step 2: 根据质量决定是否增强
        enhanced = False
        if "过曝" in quality["issues"]:
            print("\n🧠 Thought: 图像过曝，需要校正")
            print("  Action: correct_overexposure")
            current_img = vt.correct_overexposure(current_img)
            enhanced_path = image_path.replace(".", "_corrected.")
            cv2.imwrite(enhanced_path, current_img)
            current_path = enhanced_path
            enhanced = True
            print(f"  Observation: 过曝校正完成，保存至 {enhanced_path}")
            result["steps"].append(
                {"step": 2, "action": "correct_overexposure", "result": "校正完成"}
            )
            self.action_history.append(("correct_overexposure", "done"))
        if "低对比度" in quality["issues"]:
            print("\n🧠 Thought: 对比度偏低，需要增强")
            print("  Action: enhance_contrast")
            current_img = vt.enhance_contrast(current_img)
            enhanced_path = current_path.replace(".", "_enhanced.")
            cv2.imwrite(enhanced_path, current_img)
            current_path = enhanced_path
            enhanced = True
            print("  Observation: 对比度增强完成")
            result["steps"].append(
                {"step": len(result["steps"]) + 1, "action": "enhance_contrast", "result": "增强完成"}
            )
            self.action_history.append(("enhance_contrast", "done"))
        # Step 3: YOLO主检测
        print("\n🧠 Thought: 图像质量检查完成，开始YOLO缺陷检测")
        print("  Action: yolo_detect")
        detections = vt.yolo_detect(current_img, conf_threshold=0.5)
        print(f"  Observation: 检测到 {len(detections)} 个缺陷")
        for d in detections:
            print(f"   - {d['class_name']}: 置信度 {d['confidence']:.3f}")
        result["steps"].append(
            {
                "step": len(result["steps"]) + 1,
                "action": "yolo_detect",
                "result": f"检测到{len(detections)}个缺陷",
            }
        )
        self.action_history.append(("yolo_detect", detections))
        # Step 4: 低置信度缺陷二次检测（亚像素）
        low_conf = [d for d in detections if d["confidence"] < 0.6]
        if low_conf:
            print("\n🧠 Thought: 存在低置信度缺陷，调用亚像素检测二次确认")
            print("  Action: subpixel_detect")
            sub_result = vt.subpixel_edge_detect(current_img)
            if sub_result:
                print(f"  Observation: 亚像素检测发现 {len(sub_result)} 条边缘")
            else:
                print("  Observation: Halcon不可用，跳过亚像素检测")
            result["steps"].append(
                {
                    "step": len(result["steps"]) + 1,
                    "action": "subpixel_detect",
                    "result": "二次确认完成",
                }
            )
            self.action_history.append(("subpixel_detect", sub_result))
        # Step 5: 生成最终报告
        print("\n🧠 Thought: 检测完成，生成最终报告")
        report = self._generate_report(quality, detections, enhanced)
        result["final_report"] = report
        print(f"\n{'='*60}")
        print("📋 最终检测报告")
        print(f"{'='*60}")
        print(f" 图像质量: {quality['quality_level']}")
        print(f" 缺陷总数: {len(detections)}")
        print(f" 质量判定: {report['verdict']}")
        print(f" 处理步骤: {len(result['steps'])} 步")
        return result

    def _run_llm(self, image_path: str) -> Dict:
        """
        LLM模式：使用LangChain Agent + 本地大模型（Ollama）
        优先使用原生工具调用Agent（qwen2.5等支持）；
        若模型不支持工具调用（如 qwen:7b），自动回退到文本ReAct模式。
        """
        if not LANGCHAIN_AVAILABLE:
            return {"error": "LangChain未安装"}
        try:
            # LangChain 1.x 中 AgentExecutor / create_tool_calling_agent 已迁移到 langchain_classic
            from langchain_classic.agents import (
                AgentExecutor,
                create_react_agent,
                create_tool_calling_agent,
            )
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            tools = self.tools
            task = f"请检测这张PCB图片的缺陷: {image_path}"

            # 方式一：原生工具调用 Agent（需要模型支持 function calling）
            tc_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", VISION_AGENT_SYSTEM_PROMPT),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
            tc_executor = AgentExecutor(
                agent=create_tool_calling_agent(self.llm, tools, tc_prompt),
                tools=tools,
                verbose=True,
                max_iterations=10,
            )
            try:
                response = tc_executor.invoke({"input": task})
                return {
                    "mode": "llm",
                    "agent_style": "tool_calling",
                    "output": response["output"],
                    "steps": response.get("intermediate_steps", []),
                }
            except Exception as e:
                # 模型不支持工具调用（Ollama 报 "does not support tools"）→ 换 ReAct
                print(f"⚠️ 原生工具调用失败({e})，改用文本 ReAct 模式")

            # 方式二：文本 ReAct Agent（qwen:7b 等不支持工具调用的模型可用）
            react_system = f"""{VISION_AGENT_SYSTEM_PROMPT}

你只能使用以下可用工具（按需调用，可多次调用）：
{{tools}}

请严格按下面的格式思考并行动（可循环多轮）：
Thought: 你当前的思考
Action: 要调用的工具名，必须是 [{{tool_names}}] 之一
Action Input: 该工具的参数（JSON 格式）
Observation: 工具返回的结果
...（Thought/Action/Action Input/Observation 可循环多次）
Thought: 我已得到最终结论
Final Answer: 对用户问题的最终中文回答
"""
            react_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", react_system),
                    ("human", "任务：{input}\n\n{agent_scratchpad}"),
                ]
            )
            react_executor = AgentExecutor(
                agent=create_react_agent(self.llm, tools, react_prompt),
                tools=tools,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True,
            )
            response = react_executor.invoke({"input": task})
            return {
                "mode": "llm",
                "agent_style": "react",
                "output": response["output"],
                "steps": response.get("intermediate_steps", []),
            }
        except Exception as e:
            print(f"❌ LLM模式运行失败: {e}")
            print("  降级到simulate模式...")
            return self._run_simulate(image_path)

    def _generate_report(self, quality: Dict, detections: List, enhanced: bool) -> Dict:
        """生成最终检测报告"""
        total = len(detections)
        verdict = "合格" if total == 0 else "不合格"
        # 按类别统计
        class_stats = {}
        for d in detections:
            cls = d["class_name"]
            class_stats[cls] = class_stats.get(cls, 0) + 1
        return {
            "quality_level": quality["quality_level"],
            "quality_issues": quality["issues"],
            "image_enhanced": enhanced,
            "total_defects": total,
            "defect_by_class": class_stats,
            "verdict": verdict,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_decision_trace(self) -> List:
        """获取Agent决策轨迹（用于调试和展示）"""
        return self.action_history


# ============================================================
# 第四部分：本地大模型（Ollama）接入配置
# ============================================================
def create_local_llm(model: str = LOCAL_LLM_MODEL):
    """创建连接本地 Ollama 的 OpenAI 兼容客户端（langchain_openai）"""
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url=LOCAL_LLM_BASE_URL,
            api_key=LOCAL_LLM_API_KEY,
        )
    except ImportError:
        print("❌ langchain-openai 未安装，无法连接本地大模型")
        return None


# ============================================================
# 第五部分：测试场景生成
# ============================================================
def generate_test_scenarios(out_dir: Optional[str] = None) -> Dict[str, str]:
    """
    生成三个测试场景的图片，返回 {名称: 路径} 字典
    """
    if out_dir is None:
        out_dir = str(WORKSPACE_ROOT / "test_scenarios")
    os.makedirs(out_dir, exist_ok=True)
    # 场景1：正常质量图片
    img1 = np.ones((400, 400, 3), dtype=np.uint8) * 100
    cv2.circle(img1, (200, 200), 25, (50, 50, 50), -1)
    cv2.line(img1, (50, 100), (350, 100), (50, 50, 50), 2)
    p1 = os.path.join(out_dir, "normal.jpg")
    cv2.imwrite(p1, img1)
    # 场景2：过曝图片
    img2 = np.ones((400, 400, 3), dtype=np.uint8) * 240
    cv2.circle(img2, (200, 200), 25, (200, 200, 200), -1)
    p2 = os.path.join(out_dir, "overexposed.jpg")
    cv2.imwrite(p2, img2)
    # 场景3：低对比度 + 微小缺陷
    img3 = np.ones((400, 400, 3), dtype=np.uint8) * 110
    cv2.circle(img3, (200, 200), 20, (100, 100, 100), -1)
    cv2.line(img3, (100, 300), (300, 300), (100, 100, 100), 1)  # 微小划痕
    p3 = os.path.join(out_dir, "low_contrast.jpg")
    cv2.imwrite(p3, img3)
    print("✅ 三个测试场景已生成:")
    print(f" 1. normal.jpg - 正常质量        -> {p1}")
    print(f" 2. overexposed.jpg - 过曝场景   -> {p2}")
    print(f" 3. low_contrast.jpg - 低对比度  -> {p3}")
    return {"正常质量": p1, "过曝": p2, "低对比度": p3}


# ============================================================
# 自测入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("视觉决策Agent 自测")
    print("=" * 60)
    scenarios = generate_test_scenarios()
    RUN_MODE = os.getenv("VISION_AGENT_MODE", "simulate").lower()
    print(f"运行模式: {RUN_MODE}")
    agent = VisionAgent(mode=RUN_MODE)
    all_reports = []
    for name, path in scenarios.items():
        print(f"\n\n{'#'*60}")
        print(f"# {name}")
        print(f"{'#'*60}")
        all_reports.append(agent.run(path))
    print(f"\n\n{'='*60}")
    print("📊 场景测试汇总")
    print(f"{'='*60}")
    for name, r in zip(scenarios.keys(), all_reports):
        print(f"\n{name}:")
        if "final_report" in r:
            report = r["final_report"]
            print(f" 质量等级: {report['quality_level']}")
            print(f" 处理步数: {len(r['steps'])} 步")
            print(f" 缺陷数量: {report['total_defects']}")
            print(f" 判定结果: {report['verdict']}")
        elif "output" in r:
            print(f" Agent风格: {r.get('agent_style', 'llm')}")
            print(f" 结论: {(r.get('output') or '无输出')[:120]}...")
    print("\n🎉 Agent自测完成！")
