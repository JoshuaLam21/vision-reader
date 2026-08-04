# vision-reader 设计文档

- 日期：2026-08-04
- 状态：已批准
- 技术栈：Python（需安装运行时）

## 1. 背景与目标

为**没有图像能力**的大语言模型（LLM）提供一个"看图能力"工具：把图片的像素信息 + 坐标整理成模型可读的**文本**，模型通过迭代式的局部观察（类似 CNN 的感受野 + 池化思路）理解复杂图片（截图、文档、图表、UI 界面等）。

目标不是"一次处理整张大图"，而是**渐进式聚焦**：

1. 模型先看全图的低分辨率概览（chunk 网格摘要），据此判断哪些区域值得关注。
2. 模型按归一化坐标 (0~1) 裁剪出局部区域并放大。
3. 模型选择一种文本编码方式读取该区域的像素信息；可选地对区域做 OCR 提取文字。
4. 模型可反复执行 2~3，逐步聚焦多个感兴趣区域。
5. 最终把所有观察汇总为一份结构化的 Markdown 报告。

**交付形态**：Python 库 + CLI，不做 MCP（后续可按需封装）。

## 2. 核心概念

### 2.1 分块概览（overview）
全图被切成 N×N 的 chunk 网格。每个 chunk 输出一段紧凑文本统计：

- 归一化坐标范围 (x1, y1, x2, y2)
- 主色（RGB + 十六进制）
- 平均亮度
- 边缘密度（衡量该区域纹理/结构复杂度）
- 颜色方差（衡量该区域颜色丰富度）

模型根据这些统计决定细看哪些区域。默认 8×8 网格，可调。

### 2.2 区域裁剪（crop）
模型用归一化坐标指定任意矩形区域，工具裁剪并放大（默认 2 倍，插值用 LANCZOS），随后转成文本编码。裁剪区域会 clamp 到图像边界，越界部分给出告警信息。

### 2.3 可插拔编码器（encoders）
三种编码方式，模型按需指定：

| 名称 | 输出 | 特点 |
|------|------|------|
| `grayscale_grid` | 灰度值网格（量化位数可调，默认 0~9） | 最接近 CNN 感受野的原始读数，信息无损失但 token 消耗大 |
| `ascii_art` | 字符明暗密度 | token 最省、最易读，丢失颜色 |
| `color_stats` | 分块主色/均值/边缘密度统计 | 适合颜色与结构判断 |

编码器通过 registry 注册，模型可以自由切换。灰度网格与 ASCII 的输出都附上"该网格每格对应的像素坐标范围"供模型对齐空间关系。

### 2.4 OCR（可选）
对指定区域提取文字。**默认引擎 EasyOCR（ch_sim + en）**：

- 理由：pip 可安装、纯 Python 生态、中英文效果良好、无需外部服务。
- PaddleOCR 中文效果更优，但依赖 PaddlePaddle（体积大、可能有系统依赖），作为**可选后端**占位，通过接口实现 + 配置一行切换。

OCR 结果输出：识别文本、每个文本框的 bbox（归一化坐标）、置信度。

## 3. 架构与模块

```
Projects/active/vision-reader/
├── pyproject.toml
├── README.md
├── vision_reader/
│   ├── __init__.py
│   ├── image_io.py        # 图片加载：路径/base64 → 统一格式，异常处理
│   ├── coordinates.py     # 归一化坐标 ↔ 像素坐标换算、越界 clamp
│   ├── crop.py            # 裁剪 + LANCZOS 放大 + 灰度化
│   ├── overview.py        # 全图 chunk 网格摘要
│   ├── report.py          # 多次观察汇总为 Markdown
│   ├── cli.py             # 命令行入口
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── base.py        # Encoder 接口 + registry
│   │   ├── grayscale_grid.py
│   │   ├── ascii_art.py
│   │   └── color_stats.py
│   └── ocr/
│       ├── __init__.py
│       ├── base.py        # OCR 引擎接口
│       ├── easyocr_engine.py
│       └── paddleocr_engine.py   # 可选后端占位
├── tests/
│   ├── conftest.py        # 合成图生成工具
│   ├── test_coordinates.py
│   ├── test_crop.py
│   ├── test_encoders.py
│   ├── test_overview.py
│   └── test_ocr.py
└── demo/
    └── demo.py            # 完整调用链路演示
```

## 4. 数据流

```
图片(路径/base64) → image_io 解码 → overview(8×8) → 模型判断关注区
    → crop(归一化坐标) → encode(grayscale|ascii|color) → 模型读取
    → ocr(可选) → 文字/bbox
    → 反复迭代 → report 汇总为 Markdown
```

## 5. 接口设计

### 5.1 库接口（vision_reader 包）

```python
# 概览：返回 chunk 列表（含坐标、统计）
overview.chunks(image, grid=(8, 8)) -> list[Chunk]

# 裁剪：返回裁剪后的图像（放大后）
crop.region(image, x1, y1, x2, y2, scale=2.0, grayscale=True) -> np.ndarray

# 编码：把图像编码成文本
encoders.encode(image, name="ascii_art", **options) -> EncodedText

# OCR
ocr.recognize(image, engine="easyocr", languages=["ch_sim", "en"]) -> OcrResult

# 报告
report.build(observations: list[Observation], title="") -> str  # Markdown
```

### 5.2 CLI 接口

```
vision overview <image> [--grid NxM] [--json]
vision crop <image> --region x1,y1,x2,y2 [--scale 2.0] [--encode ascii_art|grayscale_grid|color_stats] [--size WxH] [--grayscale]
vision ocr <image> [--region x1,y1,x2,y2] [--engine easyocr]
vision report <json-files...> --out report.md
vision demo                   # 生成合成图并跑完整链路
```

`overview` 输出即"模型的第一眼"，`crop` 的编码文本即"模型的细看"，两者都包含归一化坐标索引。

## 6. 异常处理

| 场景 | 行为 |
|------|------|
| 图片文件不存在 / 解码失败 | 结构化错误（非崩溃），CLI 返回非零退出码并输出可读信息 |
| 归一化坐标非法（非数值、越界 0~1） | 拒绝并提示合法范围 |
| 坐标区域超出图像边界 | clamp 到边界 + 输出中附告警 |
| OCR 引擎未安装 / 返回空结果 | 返回空结果对象（not None），报告中标明"未识别到文字" |
| 未注册的编码器名 | 列出可用编码器并报错 |

## 7. 测试策略

- 全部测试使用 **合成图**（Pillow/numpy 生成）：色块、几何图形、绘制的中英文文字，保证可复现、不依赖网络。
- 单测覆盖：坐标换算（含边界）、裁剪尺寸/插值、三种编码器的输出格式与内容、overview 网格统计、OCR 识别合成图中文字。
- EasyOCR 首次运行需下载模型权重，测试中用缓存；若网络不可用则 OCR 测试跳过（pytest.mark.skipif），其余测试不受影响。

## 8. demo 流程

`vision demo` 生成一张合成图（含标题文字、色块区域、几何图形、一段正文文字），然后自动执行：

1. overview 全图概览
2. 模拟"模型选择"：按预设规则挑选 2~3 个高边缘密度/含文字区域
3. 对每个区域 crop + 编码（混合使用不同编码器）
4. 对含文字区域做 OCR
5. report 汇总输出 Markdown

输出到 `demo/output/`，并打印到终端。

## 9. 非目标（本次不做）

- MCP server 封装（接口已按"工具化"设计，后续可包一层）
- PaddleOCR 实际集成（仅占位接口）
- PDF 输入（可先转图片再喂入）
- 目标检测/显著性模型（显著性靠边缘密度、颜色方差等统计近似）
