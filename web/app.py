"""
Web 界面（Gradio）
=================
PCB 缺陷智能检测平台：单图检测 / 批量检测 / 性能测试 / 项目信息。
由项目早期 web_app.py 重构而来，统一接入 core / utils 模块。
"""
import json
import os
from datetime import datetime

import cv2
import gradio as gr

from ..core import vision_toolkit as vt
from ..core.agent import VisionAgent
from ..core.batch import batch_detect
from ..core.benchmark import run_performance_test
from ..core.detector import PcbDetector
from ..utils.config import (
    AGENT_MODE,
    DEFAULT_CONF_THRESHOLD,
    OUTPUT_DETECTED,
    OUTPUT_DIR,
)
from ..utils.file_utils import ensure_dir
from ..utils.logger import get_logger

logger = get_logger("pcb_vision_agent.web")

# 确保输出目录存在
ensure_dir(OUTPUT_DETECTED)

# Agent 单例（懒加载，mode 由配置决定，可用 VISION_AGENT_MODE 环境变量切换）
_agent = None


def get_agent() -> VisionAgent:
    """获取全局 Agent 单例"""
    global _agent
    if _agent is None:
        _agent = VisionAgent(mode=AGENT_MODE)
    return _agent


# ============================================================
# 第一部分：单图检测功能
# ============================================================
def detect_single_image(input_image, conf_threshold=DEFAULT_CONF_THRESHOLD, use_agent=True):
    """
    单张图片检测
    input_image: Gradio传入的numpy数组（RGB格式）
    返回: (检测结果图RGB, Markdown报告, 附加信息)
    """
    if input_image is None:
        return None, "请先上传图片", ""

    # Gradio传入的是RGB，转BGR给OpenCV
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    detector = PcbDetector(conf_threshold=conf_threshold)

    # 保存临时文件供 Agent 使用
    ensure_dir(OUTPUT_DIR)
    temp_path = os.path.join(OUTPUT_DIR, "temp_input.jpg")
    cv2.imwrite(temp_path, img_bgr)

    # 结构化检测结果（规则引擎兜底）
    det = detector.detect_file(temp_path)
    report = det.to_dict()

    # 绘制检测结果
    result_img_bgr = detector.annotate(img_bgr, det.detections)
    result_img = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

    # Agent 决策轨迹（可选）
    steps_md = ""
    if use_agent:
        agent = get_agent()
        agent_result = agent.run(temp_path)
        steps = agent_result.get("steps", [])
        steps_md = "\n### 🤖 Agent决策过程\n"
        if steps:
            for i, s in enumerate(steps, 1):
                act = s.get("action", "?")
                res = s.get("result", "")
                if isinstance(res, (dict, list)):
                    res = json.dumps(res, ensure_ascii=False)
                steps_md += f"{i}. **{act}** → {res}\n"
        else:
            steps_md += "_（Agent 未产生决策轨迹，使用规则引擎结果）_\n"

    result_info = f"""
## 📊 检测报告
### 图像质量
- 质量等级：**{report['quality_level']}**
- 存在问题：{', '.join(report['quality_issues']) if report['quality_issues'] else '无'}
- 图像增强：{'是' if report['enhanced'] else '否'}
- 使用算法：{', '.join(report['methods']) if report['methods'] else '无'}
### 检测结果
- 缺陷总数：**{report['total_defects']}** 个
- 缺陷分类：{json.dumps(report['defect_by_class'], ensure_ascii=False)}
{steps_md}
### 最终判定
- **{report['verdict']}**
- 检测时间：{report['timestamp']}
"""
    # 保存异常工件
    if report["total_defects"] > 0:
        save_path = os.path.join(
            OUTPUT_DETECTED, f"defect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(save_path, result_img_bgr)
        result_info += f"- 异常图已保存：{save_path}"

    return result_img, result_info, ""


# ============================================================
# 第二部分：批量检测（UI 包装）
# ============================================================
def batch_detect_ui(folder_path, conf_threshold=DEFAULT_CONF_THRESHOLD):
    """批量检测：调用 core.batch.batch_detect"""
    summary, report_path, pass_rate = batch_detect(folder_path, conf_threshold)
    return summary, report_path, pass_rate


def run_perf_ui(path):
    """性能测试：调用 core.benchmark.run_performance_test"""
    result = run_performance_test(path)
    return result if result else {"error": "测试失败"}


# ============================================================
# 第三部分：Gradio Web界面
# ============================================================
def build_interface():
    """构建完整的Web界面"""
    with gr.Blocks(title="PCB缺陷智能检测平台") as demo:
        gr.Markdown(
            """
# 🔍 PCB缺陷智能检测平台
### 基于视觉Agent的工业质检系统
融合传统机器视觉 + 深度学习 + AI Agent智能决策，
自主评估图像质量，自动选择最优检测方案
"""
        )

        with gr.Tabs():
            # ===== Tab1: 单图检测 =====
            with gr.Tab("单图检测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(label="上传PCB图片", type="numpy")
                        conf_slider = gr.Slider(
                            minimum=0.1, maximum=0.95, value=DEFAULT_CONF_THRESHOLD,
                            step=0.05, label="置信度阈值",
                        )
                        use_agent_check = gr.Checkbox(value=True, label="启用Agent智能决策")
                        detect_btn = gr.Button("🔍 开始检测", variant="primary")
                    with gr.Column(scale=1):
                        output_img = gr.Image(label="检测结果")
                        output_text = gr.Markdown(label="检测报告")
                detect_btn.click(
                    fn=detect_single_image,
                    inputs=[input_img, conf_slider, use_agent_check],
                    outputs=[output_img, output_text, gr.State()],
                )

            # ===== Tab2: 批量检测 =====
            with gr.Tab("批量检测"):
                with gr.Row():
                    with gr.Column():
                        folder_input = gr.Textbox(
                            label="图片文件夹路径",
                            placeholder="例如: pcb_dataset/images/val",
                        )
                        batch_conf = gr.Slider(
                            minimum=0.1, maximum=0.95, value=DEFAULT_CONF_THRESHOLD,
                            step=0.05, label="置信度阈值",
                        )
                        batch_btn = gr.Button("📦 开始批量检测", variant="primary")
                    with gr.Column():
                        batch_result = gr.Markdown(label="检测结果汇总")
                        report_file = gr.File(label="下载Excel报表")
                        pass_rate_display = gr.Number(label="良率(%)")
                batch_btn.click(
                    fn=batch_detect_ui,
                    inputs=[folder_input, batch_conf],
                    outputs=[batch_result, report_file, pass_rate_display],
                )

            # ===== Tab3: 性能测试 =====
            with gr.Tab("性能测试"):
                with gr.Row():
                    with gr.Column():
                        perf_path = gr.Textbox(
                            label="测试数据集路径",
                            value="pcb_dataset/images/val",
                        )
                        perf_btn = gr.Button("⚡ 运行性能测试", variant="primary")
                    with gr.Column():
                        perf_result = gr.JSON(label="性能测试结果")
                perf_btn.click(fn=run_perf_ui, inputs=perf_path, outputs=perf_result)

            # ===== Tab4: 项目信息 =====
            with gr.Tab("项目信息"):
                gr.Markdown(
                    """
## 项目：轻量化3C零件缺陷检测视觉Agent
### 技术栈
- **传统视觉**: OpenCV + Halcon（亚像素）
- **深度学习**: YOLOv8 + PyTorch（6 类缺陷）
- **AI Agent**: LangChain + 本地大模型（Ollama）
- **可视化**: Gradio
- **报表**: openpyxl
### 数据集
DeepPCB 公开数据集：训练 1000 张 / 验证 569 张，6 类缺陷
（open 开路 / short 短路 / mousebite 鼠咬 / spur 毛刺 / copper 铜 / pin_hole 针孔）
### 📊 性能指标（实测）
| 指标 | 数值 |
|------|------|
| mAP@0.5 | 96.6% |
| 召回率 | 95.0% |
| 漏检率 | 4.97% |
| 单图推理 | ~68ms |
| 检测速度 | ~14.7 FPS |
| 漏检率降低 | 95%（对比固定参数传统方案） |
### 核心功能
1. 图像质量自动评估
2. Agent智能选择处理方案
3. 多算法融合缺陷检测
4. 批量检测 + Excel报表
5. 异常工件自动存档
### 使用
- 单图检测：上传图片 → 点击开始检测
- 批量检测：输入图片文件夹路径 → 点击开始批量检测
- 性能测试：「性能测试」Tab 运行，输出真实指标
- Agent模式：根据图像质量自动调整处理策略
"""
                )
        gr.Markdown(
            """
---
💡 **使用提示**：单图检测适合快速验证，批量检测适合产线抽检。
Agent模式会根据图像质量自动调整处理策略，检测更智能。
"""
        )
    return demo


# ============================================================
# 主程序入口
# ============================================================
def launch(server_name="0.0.0.0", server_port=7860, share=False):
    """构建界面并启动服务"""
    print("=" * 60)
    print("🚀 PCB缺陷智能检测平台启动中...")
    print("=" * 60)
    demo = build_interface()
    print("\n✅ 界面构建完成")
    print(f"🌐 访问地址: http://localhost:{server_port}")
    print("📖 按 Ctrl+C 停止服务\n")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    launch()
