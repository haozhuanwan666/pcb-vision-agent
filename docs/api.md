# API 参考

## 1. PcbDetector（core/detector.py）

检测引擎，融合质量评估 / 增强 / 传统 / YOLO / Halcon。

```python
from pcb_vision_agent import PcbDetector

det = PcbDetector(conf_threshold=0.5, iou_threshold=0.45, min_area=30.0)
```

### 方法

| 方法 | 签名 | 说明 |
|---|---|---|
| `assess_quality` | `(image) -> Dict` | 质量评估 `{brightness, contrast, sharpness, quality_level, issues}` |
| `preprocess` | `(image) -> (image, enhanced, quality)` | 过曝校正 / 对比度增强 自动预处理 |
| `detect_traditional` | `(image) -> List[Dict]` | 传统法：阈值+形态学+轮廓 |
| `detect_yolo` | `(image) -> List[Dict]` | YOLO 深度学习检测 |
| `detect_subpixel` | `(image) -> Optional[List]` | Halcon 亚像素边缘（无 Halcon 返回 None） |
| `detect` | `(image, methods=("yolo","traditional")) -> DetectionResult` | 完整流水线 |
| `detect_file` | `(image_path, methods=...) -> DetectionResult` | 从文件检测（读不到抛 FileNotFoundError） |
| `annotate` | `(image, detections) -> image` | 绘制检测框（自动分流 YOLO/传统样式） |

### DetectionResult（dataclass）

字段：`image_path / quality / detections / methods / enhanced / verdict / timestamp`
方法：`to_dict() -> Dict`（可序列化，含 `defect_by_class` 统计）

## 2. VisionAgent（core/agent.py）

视觉决策 Agent，双模式。

```python
from pcb_vision_agent import VisionAgent

# simulate：规则模拟（无需 API）
agent = VisionAgent(mode="simulate")
result = agent.run("image.jpg")
# -> {"steps": [...], "final_report": {...}}   (simulate)
# -> {"mode": "llm", "agent_style": "...", "output": "...", "steps": [...]}  (llm)

# llm：本地大模型（Ollama），失败自动降级 simulate
agent2 = VisionAgent(mode="llm")
```

| 方法 | 说明 |
|---|---|
| `run(image_path) -> Dict` | 执行完整 Agent 决策流程 |
| `get_decision_trace() -> List` | 获取决策轨迹 `[(action, result), ...]` |

### 注册的 6 个工具

`assess_image_quality` / `enhance_contrast` / `correct_overexposure` /
`yolo_detect` / `traditional_detect` / `subpixel_detect`
（`get_all_tools()` 返回；无 LangChain 时返回空列表）

## 3. 视觉算子（core/vision_toolkit.py）

14 个算子，输入 BGR numpy 图像，输出结构化结果：

- 质量：`assess_image_quality`
- 增强：`enhance_contrast` / `denoise_image` / `correct_overexposure` / `gamma_correction`
- 传统：`threshold_segment` / `morphological_process` / `find_defect_contours` / `draw_defects`
- YOLO：`load_yolo_model` / `yolo_detect` / `draw_yolo_detections`
- Halcon：`subpixel_edge_detect` / `measure_defect_size_subpixel`（`halcon_available()` 探测）

## 4. 批量 / 报表 / 性能

```python
from pcb_vision_agent.core.batch import batch_detect
from pcb_vision_agent.core.report import generate_excel_report, save_json_log, save_json_report
from pcb_vision_agent.core.benchmark import run_performance_test

# 批量检测 → (汇总文本, Excel路径, 良率)
summary, report_path, rate = batch_detect("pcb_dataset/images/val")

# 报表
generate_excel_report(results, "out.xlsx")   # results: [{filename, quality_level, total_defects, defect_by_class, verdict}]
save_json_log(data, "batch")                 # -> output/logs/batch_时间戳.json
save_json_report(data, "性能测试报告")        # -> output/reports/...

# 性能测试（默认数据集 pcb_dataset/images/val）
perf = run_performance_test(dataset_path=None)
```

## 5. 配置（utils/config.py）

| 常量/函数 | 默认 | 说明 |
|---|---|---|
| `get_yolo_weights()` | 自动探测 | 权重：`YOLO_WEIGHTS` > `runs/train/pcb_defect-5/.../best.pt` > `models/best.pt` > `yolov8n.pt` |
| `AGENT_MODE` | `simulate` | 环境变量 `VISION_AGENT_MODE` |
| `DEFAULT_CONF_THRESHOLD` | `0.5` | 环境变量 `CONF_THRESHOLD` |
| `OUTPUT_DETECTED/REPORTS/LOGS` | `output/*` | 输出目录 |
| `LOCAL_LLM_*` | Ollama 默认 | `LOCAL_LLM_BASE_URL/MODEL/API_KEY` |
| `IMAGE_EXTENSIONS` | jpg/jpeg/png/bmp/tiff | 支持格式 |

## 6. 工具函数（utils/file_utils.py）

`ensure_dir` / `timestamp_str` / `find_images` / `read_image` / `save_image` / `load_json`

## 7. 日志（utils/logger.py）

```python
from pcb_vision_agent.utils.logger import get_logger, info, warn, error

log = get_logger("my_module")
log.info("...")          # 控制台 + output/logs/app_时间戳.log
```

## 8. Web 启动（web/app.py）

```python
from pcb_vision_agent.web.app import build_interface, launch

launch(server_port=7860)          # 启动服务
# demo = build_interface()        # 仅构建界面对象
```
