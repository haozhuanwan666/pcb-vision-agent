# 🔍 PCB 视觉检测 Agent（项目整合）

融合**传统机器视觉(OpenCV/Halcon) + 深度学习(YOLO) + AI Agent 智能决策**的 PCB 缺陷检测系统。
这是整个学习项目的最终整合，将前几天的成果（预处理 / 相机标定 / Halcon / YOLO 训练 / 视觉 Agent / Web 界面 / 批量报表）统一收敛为规范化的工程包。

---

## 📁 目录结构

```
pcb_vision_agent/
├── main.py               # 统一命令行入口
├── requirements.txt      # 依赖清单
├── core/                 # 核心引擎
│   ├── vision_toolkit.py # 视觉算子（14 个：质量评估/增强/传统/YOLO/Halcon）
│   ├── detector.py       # 检测引擎（质量评估→自动增强→多算法检测 流水线）
│   ├── agent.py          # 视觉决策 Agent（simulate 规则 / llm 大模型 双模式）
│   ├── report.py         # Excel/JSON 报表
│   ├── batch.py          # 批量检测
│   └── benchmark.py      # 性能测试
├── web/
│   └── app.py            # Gradio Web 界面
├── utils/
│   ├── config.py         # 全局配置（路径/模型/参数）
│   ├── logger.py         # 日志（控制台 + 文件）
│   └── file_utils.py     # 文件工具
├── data/                 # 数据集（images/train, images/val, labels）
├── models/               # 模型权重目录（可放置 best.pt）
├── output/               # 输出（detected 异常图 / reports 报表 / logs 日志）
└── docs/                 # 文档
```

---

## 🚀 快速开始

在 **`d:\vscode python`** 目录（包上级）下运行：

```bash
# 1. 启动 Web 界面（默认）
python -m pcb_vision_agent.main

# 2. 单图检测
python -m pcb_vision_agent.main single -i test_scenarios/normal.jpg

# 3. 批量检测（文件夹内所有图片 → Excel/JSON 报表 + 异常图归档）
python -m pcb_vision_agent.main batch -f pcb_dataset/images/val

# 4. 性能测试（推理速度 / 准确率 / 传统 vs YOLO 对比）
python -m pcb_vision_agent.main perf

# 5. 生成测试场景并运行 Agent（simulate 规则模式）
python -m pcb_vision_agent.main demo

# 6. 快速自检（核心模块 + 检测流水线 + Web 构建）
python -m pcb_vision_agent.main selftest
```

> 也可以直接脚本方式运行：`python pcb_vision_agent/main.py web`

---

## ⚙️ 配置说明

所有配置集中在 `utils/config.py`，可通过**环境变量**覆盖：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VISION_AGENT_MODE` | `simulate` | Agent 模式：`simulate`(规则) / `llm`(本地大模型) |
| `YOLO_WEIGHTS` | 自动探测 | 权重路径：训练产物 > `models/best.pt` > `yolov8n.pt` |
| `CONF_THRESHOLD` | `0.5` | 默认置信度阈值 |
| `LOCAL_LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI 兼容地址 |
| `LOCAL_LLM_MODEL` | `qwen2.5:7b` | 本地大模型名 |

### 接入本地大模型（LLM 模式）

```bash
# 1. 启动 Ollama 并拉取模型
ollama pull qwen2.5:7b

# 2. 切换 Agent 模式运行
$env:VISION_AGENT_MODE = "llm"
python -m pcb_vision_agent.main demo
```

模型不支持工具调用时（如 `qwen:7b`），Agent 会自动回退到文本 ReAct 模式；
模型或服务不可用时自动降级为 simulate 规则模式。

---

## 🧩 核心 API 示例

```python
from pcb_vision_agent import PcbDetector, VisionAgent

# 规则检测引擎（质量评估 → 自动增强 → YOLO/传统检测）
detector = PcbDetector(conf_threshold=0.5)
result = detector.detect_file("test_scenarios/normal.jpg")
print(result.to_dict())          # {'quality_level':..., 'total_defects':..., 'verdict':...}
print(result.detections)         # 缺陷列表
annotated = result_annotate      # detector.annotate(img, detections)

# 视觉决策 Agent（双模式）
agent = VisionAgent(mode="simulate")   # 或 mode="llm"
trace = agent.run("test_scenarios/overexposed.jpg")
print(trace["steps"])                  # Agent 决策轨迹
```

---

## 📊 输出产物

- `output/detected/` — 检出的异常工件标注图（自动归档）
- `output/reports/`  — Excel 质检报表（带样式、良率统计）+ 性能测试 JSON
- `output/logs/`     — 批量检测 JSON 日志 + 运行日志

---

## 🛠 技术栈

| 模块 | 技术 |
|---|---|
| 传统视觉 | OpenCV（阈值/形态学/轮廓）+ Halcon（亚像素，可选） |
| 深度学习 | YOLO（ultralytics）+ PyTorch |
| AI Agent | LangChain（LangChain 1.x） |
| Web | Gradio |
| 报表 | openpyxl |

## ✅ 自测

`python -m pcb_vision_agent.main selftest` 会依次验证：
1. 检测引擎完整流水线（质量评估 → 增强 → 缺陷判定）
2. VisionAgent 初始化与 simulate 决策
3. Gradio Web 界面构建
