# 系统架构设计

## 1. 总体架构

本项目把整个学习过程（预处理 → 相机标定 → Halcon → YOLO 训练 → 视觉 Agent → Web/报表）整合为
**核心引擎层 + 决策智能层 + 应用层** 三层架构：

```mermaid
graph TB
    subgraph App["应用层"]
        CLI["main.py 统一入口<br/>(web/single/batch/perf/demo/selftest)"]
        WEB["web/app.py Gradio 界面<br/>(单图/批量/性能/信息)"]
    end

    subgraph Core["核心引擎层 core/"]
        DET["detector.py 检测引擎<br/>质量评估→自动增强→多算法检测"]
        AG["agent.py 视觉决策 Agent<br/>(simulate 规则 / llm 大模型)"]
        BATCH["batch.py 批量检测"]
        RPT["report.py Excel/JSON 报表"]
        BENCH["benchmark.py 性能测试"]
        TK["vision_toolkit.py 视觉算子库<br/>(14 个算子)"]
    end

    subgraph Util["支撑层 utils/"]
        CFG["config.py 全局配置"]
        LOG["logger.py 日志"]
        FU["file_utils.py 文件工具"]
    end

    CLI --> WEB
    CLI --> DET
    CLI --> AG
    CLI --> BATCH
    CLI --> BENCH

    WEB --> DET
    WEB --> AG
    WEB --> BATCH
    WEB --> BENCH

    DET --> TK
    AG --> TK
    BATCH --> DET
    BATCH --> RPT
    BENCH --> DET

    DET --> CFG
    AG --> CFG
    BATCH --> CFG
    RPT --> CFG
    BENCH --> CFG
    TK --> CFG
```

## 2. 模块职责

| 模块 | 职责 | 依赖 |
|---|---|---|
| `core/vision_toolkit.py` | 14 个视觉算子：质量评估 / CLAHE 增强 / 去噪 / 过曝校正 / Gamma / 阈值 / 形态学 / 轮廓 / YOLO 检测 / Halcon 亚像素 | cv2, numpy, ultralytics, halcon(可选) |
| `core/detector.py` | `PcbDetector` 高层引擎：质量评估 → 自动预处理 → YOLO/传统多算法检测 → 判定 | vision_toolkit |
| `core/agent.py` | `VisionAgent` 双模式 Agent；6 个 LangChain 工具；LLM(Ollama) 接入 | vision_toolkit, langchain |
| `core/report.py` | Excel 质检报表（带样式/良率）、JSON 日志/报表落盘 | openpyxl(可选) |
| `core/batch.py` | 文件夹批量检测 + 报表 + 异常图归档 | detector, report |
| `core/benchmark.py` | 推理速度 / 准确率 / 传统 vs YOLO 对比 | detector, report |
| `web/app.py` | Gradio 界面（单图/批量/性能/项目信息） | core 各模块 |
| `utils/config.py` | 全局路径/模型/参数，支持环境变量覆盖 | 无 |
| `utils/logger.py` | 控制台 + 文件双输出，UTF-8 修复 | config |
| `utils/file_utils.py` | 目录/时间戳/图片查找读写 | config |

## 3. 检测数据流

```mermaid
sequenceDiagram
    participant U as 用户/Web
    participant D as PcbDetector
    participant Q as 质量评估
    participant P as 预处理
    participant Y as YOLO/传统
    participant R as 报表

    U->>D: detect_file(image_path)
    D->>Q: assess_image_quality(image)
    Q-->>D: {brightness, contrast, sharpness, issues}
    D->>P: 存在"过曝"→correct_overexposure<br/>存在"低对比度"→enhance_contrast
    P-->>D: 增强后的图 + enhanced 标记
    D->>Y: detect_yolo() 主检测
    alt YOLO 无检出
        D->>Y: detect_traditional() 传统法兜底
    end
    Y-->>D: detections + methods
    D-->>U: DetectionResult(质量/缺陷/判定/时间)
    U->>R: to_dict() → Excel/JSON
```

## 4. Agent 决策流（双模式）

```mermaid
flowchart TD
    A[输入图片] --> B{mode?}
    B -->|simulate 规则| S1[1. assess_image_quality]
    B -->|llm 大模型| L1[LangChain Agent 自主决策]
    S1 --> S2{存在问题?}
    S2 -->|过曝| S3[correct_overexposure]
    S2 -->|低对比度| S4[enhance_contrast]
    S3 --> S5[yolo_detect 主检测]
    S4 --> S5
    S5 --> S6{低置信度缺陷?}
    S6 -->|是| S7[subpixel_detect 二次确认]
    S6 -->|否| S8[生成最终报告]
    S7 --> S8
    L1 --> L2{支持工具调用?}
    L2 -->|是| L3[tool_calling Agent]
    L2 -->|否| L4[文本 ReAct Agent]
    L3 --> L5[输出中文检测报告]
    L4 --> L5
    L5 -.失败降级.-> S1
```

## 5. 关键设计决策

1. **双模式 Agent**：`simulate` 无需 API 可离线演示；`llm` 连接本地 Ollama；
   模型不支持工具调用自动回退 ReAct；LLM 不可用自动降级 simulate。
2. **单例模型加载**：YOLO 模型模块级缓存（`_yolo_model`），避免重复加载耗时。
3. **规则引擎兜底**：`PcbDetector` 在 YOLO 无检出时用传统法兜底，比 Agent 单纯 YOLO 更灵敏。
4. **配置集中化**：路径/模型/参数全部收敛到 `utils/config.py`，环境变量可覆盖
   （`YOLO_WEIGHTS` / `VISION_AGENT_MODE` / `CONF_THRESHOLD` 等）。
5. **可选依赖优雅降级**：Halcon 缺失→OpenCV 替代；openpyxl 缺失→跳过报表；
   LangChain 缺失→占位基类保证模块可导入。
6. **绘制分流**：`annotate()` 根据检测结果是否含 `class_id` 自动选择
   YOLO 样式或传统轮廓样式（修复了 `KeyError: 'class_id'`）。
