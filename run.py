"""
PCB 缺陷智能检测系统 - 一键启动入口
==================================
用法:
    python run.py web        # 启动 Web 检测平台
    python run.py train      # 训练 YOLO 模型
    python run.py test       # 运行性能测试
    python run.py agent      # Agent 模式单图测试
"""
import os
import sys

# 将包所在工作区加入路径，保证以脚本方式运行也可导入 pcb_vision_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_vision_agent.utils.logger import get_logger  # noqa: E402

logger = get_logger("run")


def print_banner():
    print("=" * 60)
    print(" 🔍  PCB 缺陷智能检测平台")
    print("   基于视觉 Agent 的工业质检系统")
    print("=" * 60)
    print()


def cmd_web():
    """启动 Web 检测平台"""
    from pcb_vision_agent.web.app import launch

    launch(server_port=7860)


def cmd_train():
    """训练 YOLO 模型（本项目为整合包，训练脚本见早期 day*/p1_* 脚本）"""
    print("🚀 开始 YOLO 模型训练 ...")
    print("   本仓库为项目整合后的检测/推理代码，不含训练脚本。")
    print("   训练流程请参考 docs/deployment.md 及项目早期训练脚本。")


def cmd_test():
    """运行性能测试"""
    from pcb_vision_agent.core.benchmark import run_performance_test

    run_performance_test()


def cmd_agent():
    """Agent 模式单图测试"""
    from pcb_vision_agent.core.agent import VisionAgent, generate_test_scenarios

    scenarios = generate_test_scenarios()
    agent = VisionAgent(mode=os.getenv("VISION_AGENT_MODE", "simulate"))
    for name, path in scenarios.items():
        print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
        agent.run(path)


def main():
    print_banner()
    if len(sys.argv) < 2:
        print("用法:")
        print("  python run.py web       # 启动 Web 检测平台")
        print("  python run.py train     # 训练 YOLO 模型")
        print("  python run.py test      # 运行性能测试")
        print("  python run.py agent     # Agent 模式单图测试")
        print()
        print("示例: python run.py web")
        return

    cmd = sys.argv[1].lower()
    if cmd == "web":
        cmd_web()
    elif cmd == "train":
        cmd_train()
    elif cmd == "test":
        cmd_test()
    elif cmd == "agent":
        cmd_agent()
    else:
        print(f"❌ 未知命令: {cmd}")
        print("可用命令: web / train / test / agent")


if __name__ == "__main__":
    main()
