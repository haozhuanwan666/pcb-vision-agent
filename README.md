# 🔍 PCB 缺陷智能检测平台

基于 LangChain 视觉 Agent 的 3C 工件多缺陷智能检测系统 | OpenCV + YOLOv8 + Agent

---

## 📋 项目简介

面向 3C 电子厂 PCB / 盖板外观检测场景，搭建**自主决策 AI 视觉 Agent**：

- **传统机器视觉**做图像预处理与质量评估（OpenCV / Halcon 亚像素）
- **YOLO 深度学习**做缺陷识别（YOLOv8 轻量化模型，6 类缺陷）
- **AI Agent** 自主切换图像处理算子、自动调参、生成检测报表（LangChain）

解决传统视觉程序无法自适应不同工件、人工反复调参的痛点。

---

## 🏗 系统架构

```
用户输入  →  图像质量评估  →  Agent 决策中心
                              ↓
          ┌──────────┬──────────┬──────────┐
          ↓          ↓          ↓          ↓
      图像增强    YOLO 检测   亚像素检测   传统检测
          └──────────┴─────┬────┴──────────┘
                            ↓
                      检测报告生成
                     /       |        \
                Excel 报表   异常存档   日志记录
```

**模块分层**：`core/`（核心算法）→ `web/`（界面）→ `utils/`（配置/日志/工具），
`data/`（数据集）、`models/`（权重）、`output/`（检测输出）、`docs/`（文档）分层明确。

---

## 🛠 技术栈

| 模块 | 技术 |
|------|------|
| 编程语言 | Python 3.10+ |
| 传统视觉 | OpenCV + Halcon（亚像素，可选） |
| 深度学习 | PyTorch + YOLOv8 |
| AI Agent | LangChain + 本地大模型（Ollama） |
| 可视化 | Gradio |
| 报表 | openpyxl |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型

将训练好的 `best.pt` 放入 `models/` 目录（未提供时自动回退官方 `yolov8n.pt` 演示）。

### 3. 启动系统

```bash
python run.py web        # 启动 Web 检测平台（浏览器访问 http://localhost:7860）
python run.py train      # 训练 YOLO 模型
python run.py test       # 运行性能测试
python run.py agent      # Agent 模式单图测试
```

> 也可用更细分的命令：`python -m pcb_vision_agent.main single|batch|perf|demo|selftest`

---

## 📁 项目结构

```
pcb_vision_agent/
├── run.py               # 一键启动入口
├── main.py              # 命令行入口（single/batch/perf/demo/selftest）
├── requirements.txt     # 依赖清单
├── core/                # 核心算法模块
│   ├── vision_toolkit.py   # 视觉算子工具箱（14 个算子）
│   ├── detector.py         # 检测引擎（质量评估→自动增强→多算法检测）
│   ├── agent.py            # 决策 Agent（simulate 规则 / llm 大模型）
│   ├── report.py           # Excel/JSON 报表
│   ├── batch.py            # 批量检测
│   └── benchmark.py        # 性能测试
├── web/
│   └── app.py           # Gradio 界面
├── utils/
│   ├── config.py        # 全局配置
│   ├── logger.py        # 日志
│   └── file_utils.py    # 文件工具
├── tests/               # 单元测试（20 用例）
├── data/ · models/ · output/ · docs/
```

---

## ✨ 核心功能

1. **智能质量评估**：自动评估图像亮度、对比度、清晰度
2. **Agent 自主决策**：根据图像质量自动选择处理策略
3. **多算法融合**：传统视觉 + 深度学习 + 亚像素检测
4. **批量检测**：文件夹批量处理，自动统计良率
5. **Excel 报表**：自动生成带样式的质检报告
6. **异常存档**：不合格工件自动保存留档

---

## 📊 性能指标

| 指标 | 数值（实测） |
|------|------|
| mAP@0.5 | 96.6%（DeepPCB 验证集训练实测） |
| 召回率 | 95.0%（569 张验证集） |
| 漏检率 | 4.97% |
| 单图推理 | ~68ms |
| 检测速度 | ~14.7 FPS |
| 漏检率降低 | 95%（对比固定参数传统方案，可复现：`scripts/compare_pipelines.py`） |

---

## ⚙️ 配置说明

所有配置集中在 `utils/config.py`，可通过**环境变量**覆盖：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VISION_AGENT_MODE` | `simulate` | Agent 模式：`simulate`(规则) / `llm`(本地大模型) |
| `YOLO_WEIGHTS` | 自动探测 | 权重路径：环境变量 > `models/best.pt` > `yolov8n.pt` |
| `CONF_THRESHOLD` | `0.5` | 默认置信度阈值 |
| `LOCAL_LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI 兼容地址 |
| `LOCAL_LLM_MODEL` | `qwen2.5:7b` | 本地大模型名 |

接入本地大模型（LLM 模式）：先 `ollama pull qwen2.5:7b`，再设 `VISION_AGENT_MODE=llm`。
模型不支持工具调用时自动回退文本 ReAct 模式；服务不可用时自动降级 simulate。

---

## 🧩 核心 API 示例

```python
from pcb_vision_agent import PcbDetector, VisionAgent

# 规则检测引擎（质量评估 → 自动增强 → YOLO/传统检测）
detector = PcbDetector(conf_threshold=0.5)
result = detector.detect_file("image.jpg")
print(result.to_dict())      # {'quality_level':..., 'total_defects':..., 'verdict':...}

# 视觉决策 Agent（双模式）
agent = VisionAgent(mode="simulate")   # 或 mode="llm"
trace = agent.run("image.jpg")
print(trace["steps"])                  # Agent 决策轨迹
```

---

## 📊 输出产物

- `output/detected/` — 检出的异常工件标注图（自动归档）
- `output/reports/` — Excel 质检报表（带样式、良率统计）+ 性能测试 JSON
- `output/logs/` — 批量检测 JSON 日志 + 运行日志

---

## 🎯 适配场景

- PCB 板外观缺陷检测
- 手机盖板瑕疵检测
- 3C 电子零件质检
- 工业产线在线检测
- 边缘设备离线部署

---

## ✅ 测试

```bash
python -m pytest tests -v      # 20 个单元测试
```

## 📝 License

MIT License
