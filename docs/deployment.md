# 部署与运行指南

## 1. 环境准备

| 依赖 | 版本参考 | 用途 |
|---|---|---|
| Python | 3.10+ | 运行环境 |
| opencv-python | ≥4.8 | 图像处理 |
| numpy | ≥1.24 | 数值计算 |
| ultralytics | ≥8.0 | YOLO 推理/训练 |
| langchain-core / langchain-classic / langchain-openai | 1.x | Agent 框架 |
| pydantic | ≥2.0 | 工具 Schema |
| gradio | ≥4.0 | Web 界面 |
| openpyxl | ≥3.1 | Excel 报表 |
| halcon | 可选 | 亚像素测量（缺失自动降级 OpenCV） |

```bash
pip install -r pcb_vision_agent/requirements.txt
```

> 本项目实测环境：conda `vision_agent`（Python 3.10.20），已装 langchain 1.3.14 / ultralytics / halcon。

## 2. 模型准备

YOLO 权重路径由 `utils/config.py::get_yolo_weights()` 自动探测，优先级：

1. 环境变量 `YOLO_WEIGHTS`（显式指定）
2. 训练产物 `runs/train/pcb_defect-5/weights/best.pt`（本工作区默认）
3. 包内 `pcb_vision_agent/models/best.pt`（拷贝权重到此处亦可）
4. 官方预训练 `yolov8n.pt`（兜底演示）

```bash
# 指定权重示例
$env:YOLO_WEIGHTS = "D:\pcb_dataset\...\best.pt"
```

## 3. 启动 Web 界面

```bash
cd "d:\vscode python"
python -m pcb_vision_agent.main web            # 默认 0.0.0.0:7860
python -m pcb_vision_agent.main web --port 8000 --share
```

浏览器访问 `http://localhost:7860`。

## 3.1 一键云端启动（推荐）

`start_cloud.bat`（项目根目录）可同时拉起 Web 服务 + cloudflared 公网隧道，
并自动打印公网访问地址：

- **双击 `start_cloud.bat`** 即可（自动使用 `vision_agent` 环境的 Python）
- 或命令行运行：`python tools\start_cloud.py`
- 按 **Ctrl+C** 一键停止并清理所有进程

行为说明：
- 若端口 7860 已有服务在运行，自动复用现有服务，只新建隧道
- 端口可用环境变量 `PCB_WEB_PORT` 覆盖
- 依赖 `tools/cloudflared.exe`（下载方式见下方「常见问题」）

> ⚠️ 免费临时隧道（trycloudflare）无 uptime 保证，链接在进程关闭/重启后失效，
> 每次启动会生成新的随机地址。需要永久域名请注册 Cloudflare 账号使用 named tunnel。


## 4. 接入本地大模型（LLM 模式）

1. 安装并启动 [Ollama](https://ollama.com)，拉取模型：
   ```bash
   ollama pull qwen2.5:7b
   ```
2. 切换 Agent 模式：
   ```bash
   $env:VISION_AGENT_MODE = "llm"
   python -m pcb_vision_agent.main demo
   ```

行为说明：
- 模型支持工具调用 → 走 `create_tool_calling_agent`
- 模型不支持（如 `qwen:7b`）→ 自动回退文本 ReAct
- LLM 服务不可用 → 自动降级 simulate

## 5. 命令行模式

```bash
python -m pcb_vision_agent.main single -i test_scenarios/normal.jpg
python -m pcb_vision_agent.main batch -f pcb_dataset/images/val
python -m pcb_vision_agent.main perf --dataset pcb_dataset/images/val
python -m pcb_vision_agent.main selftest
```

## 6. 运行单元测试

```bash
python -m pytest pcb_vision_agent/tests -v
```

覆盖：Agent 决策 / 检测引擎 / 报表 / 批量。20 个用例，含传统 vs YOLO 绘制分流回归。

## 7. 输出产物

| 目录 | 内容 |
|---|---|
| `output/detected/` | 异常工件标注图（自动归档） |
| `output/reports/` | Excel 质检报表 + 性能测试 JSON |
| `output/logs/` | 批量检测 JSON + 运行日志 |

## 8. 常见问题

| 问题 | 解决 |
|---|---|
| 控制台 emoji 乱码 | 已自动 `sys.stdout.reconfigure(utf-8)`；手工运行请设 `$env:PYTHONIOENCODING="utf-8"` |
| `KeyError: class_id` | 已修复：`annotate()` 按检测来源分流（请确保使用本包 detector） |
| 中文路径读图失败 | `cv2.imread` 不支持中文路径，请先用 `save_image` 复制到英文路径 |
| Excel 报表为空 | 未安装 openpyxl，`pip install openpyxl` |
| LLM 模式跑不通 | 检查 Ollama 是否运行、模型是否已 pull、`VISION_AGENT_MODE=llm` 是否设置 |
| Gradio `share=True` 无公网链接 | 需下载 frpc，但 `cdn-media.huggingface.co` 在部分网络被墙 → 改用 `start_cloud.bat`（cloudflared 隧道） |
| 下载 cloudflared.exe | GitHub 被墙时走代理：`curl -L -o tools/cloudflared.exe https://ghproxy.net/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe` |
