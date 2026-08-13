"""
PCB 视觉检测 Agent —— 统一入口
===============================

用法（在 d:\\vscode python 目录下运行）：
    python -m pcb_vision_agent.main                      # 启动 Web 界面（默认）
    python -m pcb_vision_agent.main web                  # 启动 Web 界面
    python -m pcb_vision_agent.main single -i xx.jpg     # 单图检测
    python -m pcb_vision_agent.main batch -f 图片目录     # 批量检测
    python -m pcb_vision_agent.main perf                 # 性能测试
    python -m pcb_vision_agent.main demo                 # 生成测试场景并跑 Agent
    python -m pcb_vision_agent.main selftest             # 快速自检

通用参数：
    --conf 0.5              置信度阈值
    --mode simulate|llm     切换 Agent 模式（simulate 规则 / llm 本地大模型）
    --agent / --no-agent    单图检测时是否启用 Agent 决策
"""
import argparse
import os
import sys

# 兼容两种运行方式：
#   1) python -m pcb_vision_agent.main   （推荐，__package__ 非空）
#   2) python pcb_vision_agent/main.py   （脚本方式，__package__ 为空）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pcb_vision_agent.core.agent import VisionAgent, generate_test_scenarios
    from pcb_vision_agent.core.batch import batch_detect
    from pcb_vision_agent.core.benchmark import run_performance_test
    from pcb_vision_agent.core.detector import PcbDetector
    from pcb_vision_agent.web.app import build_interface, launch
    from pcb_vision_agent.utils.config import (
        AGENT_MODE,
        DEFAULT_CONF_THRESHOLD,
        OUTPUT_DETECTED,
    )
else:
    from .core.agent import VisionAgent, generate_test_scenarios
    from .core.batch import batch_detect
    from .core.benchmark import run_performance_test
    from .core.detector import PcbDetector
    from .web.app import build_interface, launch
    from .utils.config import (
        AGENT_MODE,
        DEFAULT_CONF_THRESHOLD,
        OUTPUT_DETECTED,
    )


def cmd_single(args):
    """单图检测"""
    detector = PcbDetector(conf_threshold=args.conf)
    agent = VisionAgent(mode=args.mode)
    print("=" * 60)
    print(f"🔍 单图检测: {args.image}")
    print("=" * 60)
    if args.agent:
        result = agent.run(args.image)
        if "final_report" in result:
            report = result["final_report"]
            print(f" 质量等级: {report['quality_level']}")
            print(f" 缺陷数量: {report['total_defects']}")
            print(f" 判定结果: {report['verdict']}")
        elif "output" in result:
            print(f" Agent风格: {result.get('agent_style', 'llm')}")
            print(f" 结论: {(result.get('output') or '无输出')}")
    det = detector.detect_file(args.image)
    print(f" 算法: {', '.join(det.methods)} | 增强: {det.enhanced} | 判定: {det.verdict}")
    print(f" 缺陷明细:")
    for d in det.detections:
        print(f"   - {d.get('class_name', '缺陷')}: 置信度 {d.get('confidence', 0):.3f}, 面积 {d.get('area', 0)}px")


def cmd_batch(args):
    """批量检测"""
    summary, report_path, pass_rate = batch_detect(args.folder, conf_threshold=args.conf)
    print(summary)


def cmd_perf(args):
    """性能测试"""
    run_performance_test(args.dataset)


def cmd_demo(args):
    """生成测试场景并跑 Agent"""
    os.environ["VISION_AGENT_MODE"] = args.mode
    scenarios = generate_test_scenarios()
    agent = VisionAgent(mode=args.mode)
    for name, path in scenarios.items():
        print(f"\n\n{'#'*60}\n# {name}\n{'#'*60}")
        agent.run(path)


def cmd_selftest(args):
    """快速自检：核心模块导入 + 检测流水线"""
    print("=" * 60)
    print("🧪 快速自检")
    print("=" * 60)
    # 生成一张测试图
    scenarios = generate_test_scenarios()
    path = scenarios["低对比度"]
    print("\n[1] 检测引擎流水线测试...")
    detector = PcbDetector(conf_threshold=args.conf)
    det = detector.detect_file(path)
    print(f" 质量: {det.quality['quality_level']} | 增强: {det.enhanced} "
          f"| 缺陷: {len(det.detections)} | 判定: {det.verdict}")
    print("\n[2] VisionAgent 初始化测试...")
    agent = VisionAgent(mode="simulate")
    result = agent.run(path)
    print(f" Agent 步数: {len(result.get('steps', []))} | "
          f"报告判定: {result.get('final_report', {}).get('verdict')}")
    print("\n[3] Web 界面构建测试...")
    demo = build_interface()
    print(f" Gradio Blocks 构建成功: {type(demo).__name__}")
    print("\n🎉 自检通过！")


def main():
    parser = argparse.ArgumentParser(
        prog="pcb_vision_agent",
        description="PCB 视觉检测 Agent —— 项目整合统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python -m pcb_vision_agent.main web\n"
               "  python -m pcb_vision_agent.main single -i test_scenarios/normal.jpg\n"
               "  python -m pcb_vision_agent.main batch -f pcb_dataset/images/val\n"
               "  python -m pcb_vision_agent.main demo",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_web = sub.add_parser("web", help="启动 Gradio Web 界面（默认）")
    p_web.add_argument("--port", type=int, default=7860, help="端口号")
    p_web.add_argument("--share", action="store_true", help="生成公网分享链接")

    p_single = sub.add_parser("single", help="单图检测")
    p_single.add_argument("-i", "--image", required=True, help="图片路径")
    p_single.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="置信度阈值")
    p_single.add_argument("--mode", default=AGENT_MODE, choices=["simulate", "llm"], help="Agent 模式")
    p_single.add_argument("--agent", dest="agent", action="store_true", default=True, help="启用 Agent 决策")
    p_single.add_argument("--no-agent", dest="agent", action="store_false", help="禁用 Agent 决策")

    p_batch = sub.add_parser("batch", help="批量检测")
    p_batch.add_argument("-f", "--folder", required=True, help="图片文件夹路径")
    p_batch.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="置信度阈值")

    p_perf = sub.add_parser("perf", help="性能测试")
    p_perf.add_argument("--dataset", default=None, help="测试数据集路径（默认 pcb_dataset/images/val）")

    p_demo = sub.add_parser("demo", help="生成测试场景并运行 Agent")
    p_demo.add_argument("--mode", default=AGENT_MODE, choices=["simulate", "llm"], help="Agent 模式")

    p_selftest = sub.add_parser("selftest", help="快速自检")
    p_selftest.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="置信度阈值")

    args = parser.parse_args()

    if args.cmd in (None, "web"):
        port = getattr(args, "port", 7860)
        share = getattr(args, "share", False)
        launch(server_port=port, share=share)
    elif args.cmd == "single":
        cmd_single(args)
    elif args.cmd == "batch":
        cmd_batch(args)
    elif args.cmd == "perf":
        cmd_perf(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    elif args.cmd == "selftest":
        cmd_selftest(args)


if __name__ == "__main__":
    main()
