# vision-reader

给**没有图像能力的 LLM** 的"看图能力"工具：把图片的像素 + 坐标整理成模型可读的**文本**，让模型通过渐进式局部观察（类似 CNN 的感受野 + 池化）理解复杂图片（截图、文档、图表、UI 界面等）。

核心思路：**先看全图 chunk 概览 → 模型决定关注哪些归一化坐标区域 → 裁剪放大并转成文本编码细看 → 可选 OCR 提取文字 → 汇总 Markdown 报告**。

## 安装

需要 Python ≥ 3.11（推荐用 [uv](https://docs.astral.sh/uv/) 管理）：

```bash
uv sync --extra dev   # 安装依赖（含 EasyOCR/torch）
```

## 快速开始

```bash
uv run vision demo                 # 生成合成图并跑完整链路，输出 demo/output/report.md
uv run vision demo --out my_out    # 自定义输出目录
```

也可以直接跑脚本：

```bash
uv run python demo/demo.py
```

## CLI 用法

所有坐标均为**归一化坐标 (0~1)**。

### 1. 全图概览（模型的第一眼）

```bash
uv run vision overview <图片> --grid 8x8
```

输出 N×N chunk 网格，每块含归一化坐标、主色、亮度、边缘密度、颜色方差，模型据此决定细看哪些区域。

### 2. 裁剪 + 编码（模型的细看）

```bash
uv run vision crop <图片> --region 0.1,0.1,0.5,0.5 --encode ascii_art --size 64
```

参数：

| 参数 | 说明 |
|---|---|
| `--region x1,y1,x2,y2` | 归一化坐标（必填） |
| `--scale` | 放大倍数，默认 2.0（LANCZOS） |
| `--encode` | 编码器：`ascii_art`（默认）/ `grayscale_grid` / `color_stats` |
| `--size N` | 编码网格宽（grayscale_grid 的 `grid_width` / ascii_art 的 `width`） |
| `--color` | 保留彩色（默认裁剪后灰度化） |
| `--json` | 以 Observation JSON 输出（供 `report` 汇总） |

### 3. OCR 提取文字

```bash
uv run vision ocr <图片> --region 0.02,0.42,0.6,0.58 --engine easyocr --languages ch_sim,en
```

缺省 `--region` 时识别整图。

### 4. 汇总报告

```bash
uv run vision report obs1.json obs2.json --out report.md --title "图片理解报告"
```

把多个 `--json` 输出（overview / crop / ocr）汇总为结构化 Markdown：整体概览、局部细节（含坐标与编码器）、OCR 结果、坐标索引表。

## 三种编码器（模型按需自选）

| 名称 | 输出 | 特点 |
|---|---|---|
| `grayscale_grid` | 灰度数字网格（量化级数可调） | 最接近原始像素读数，token 消耗大 |
| `ascii_art` | 字符明暗密度图 | token 最省、最易读 |
| `color_stats` | 分块主色/亮度/边缘密度统计 | 适合颜色与结构判断 |

编码器可插拔：实现 `vision_reader/encoders/base.py` 的 `Encoder` 接口并加 `@register` 即可。

## Python 库用法

```python
from vision_reader import crop, image_io, overview
from vision_reader.encoders import encode
from vision_reader.ocr import recognize
from vision_reader.report import Observation, build

img = image_io.load_image("截图.png")          # 支持路径 / base64 / bytes / ndarray
chunks = overview.chunks(img, grid=(8, 8))    # 全图概览
print(overview.overview_text(chunks))

cr = crop.region(img, 0.1, 0.1, 0.5, 0.5, scale=2.0)  # 裁剪放大
print(encode(cr.image, name="grayscale_grid", grid_width=32))  # 编码细看

result = recognize(cr.image, engine="easyocr", languages=("ch_sim", "en"))  # OCR
print(result.text)

report_md = build([obs1, obs2], title="图片理解报告")  # 汇总
```

## OCR 引擎

- **默认 EasyOCR（ch_sim + en）**：pip 可装、纯 Python 生态、无需外部服务；首次使用会自动下载模型权重到 `~/.EasyOCR`。
- **PaddleOCR（可选占位）**：中文效果更优但依赖 PaddlePaddle（体积大）。当前为占位实现，接口已就绪：补全 `vision_reader/ocr/paddleocr_engine.py` 后，一行切换 `--engine paddleocr`。

## MCP Server（可选）

把 vision-reader 封装为 MCP 工具，供 Claude Code / Reasonix 等支持 MCP 的客户端直接调用。

安装并启动：

```bash
uv sync --extra mcp                     # 安装 mcp SDK
uv run python -m vision_reader.mcp_server   # stdio transport
```

### 工具清单

| 工具 | 说明 |
|---|---|
| `vision_load_image` | 注册图片（路径或 base64），返回 `image_id` + 宽高 |
| `vision_overview` | 全图 chunk 概览（默认 8x8，可调 grid） |
| `vision_crop` | 按归一化坐标裁剪 + 编码（`region="x1,y1,x2,y2"`，编码器可换） |
| `vision_ocr` | 区域 OCR（返回文本 + 置信度 + 归一化 bbox） |
| `vision_list_encoders` / `vision_list_ocr_engines` | 查询可用的编码器 / OCR 引擎 |

设计要点：`vision_load_image` 注册图片后，后续工具只传 `image_id`（server 端缓存），避免重复传 base64 浪费 token；OCR 引擎实例在 server 内复用，模型只加载一次。所有坐标均为归一化 (0~1)，看哪些区域由模型自行决定。

### 注册到客户端

Claude Code（项目 `.mcp.json`）：

```json
{
  "mcpServers": {
    "vision-reader": {
      "command": "uv",
      "args": ["--directory", "C:/Users/Lam/Desktop/Workspace/Company/Projects/active/vision-reader", "run", "python", "-m", "vision_reader.mcp_server"]
    }
  }
}
```

Reasonix：同样以 stdio server 方式注册（在 `.mcp.json` 或全局配置中指向上述 command/args），或在 Reasonix 中把 `uv run --directory <项目路径> python -m vision_reader.mcp_server` 配置为 MCP server 启动命令。

### 使用流程（给模型的建议）

1. `vision_load_image` 注册图片 → 拿到 `image_id`
2. `vision_overview(image_id)` 看全图概览，记下值得细看的归一化区域
3. `vision_crop(image_id, region=..., encoder=ascii_art)` 细看；颜色/像素级需求可换 `color_stats` / `grayscale_grid`
4. 需要读文字时 `vision_ocr(image_id, region=...)`
5. 反复 2~4 直到理解足够，再自行汇总语义

## 接入 Reasonix / Claude 类 agent（bash 方式）

不需要 MCP 时，CLI 本身即工具形态，agent 可通过 bash 工具直接调用，例如注册为一个 Reasonix skill：

```markdown
# 看图的工具（vision-reader）
当需要理解一张图片时，按渐进式流程调用：
1. `uv run vision overview <img> --grid 8x8` → 看全图概览，记下值得关注的归一化坐标
2. `uv run vision crop <img> --region x1,y1,x2,y2 --encode ascii_art` → 细看某区域
3. 需要读文字时：`uv run vision ocr <img> --region x1,y1,x2,y2`
4. 反复 2~3 直到理解足够，再自行组织语义
```

要点：**坐标一律用归一化 (0~1)**；编码器可混用（结构用 ascii_art、颜色用 color_stats、像素级用 grayscale_grid）；token 预算有限时优先 `ascii_art`。

## 项目结构

```
vision_reader/
├── image_io.py          # 图片加载（路径/base64/bytes/ndarray）
├── coordinates.py       # 归一化坐标 ↔ 像素换算、越界处理
├── crop.py              # 裁剪 + LANCZOS 放大 + 灰度化
├── overview.py          # 全图 chunk 网格摘要
├── report.py            # 多次观察汇总为 Markdown
├── cli.py               # CLI 入口（vision 命令）
├── demo_runner.py       # demo 完整链路
├── encoders/            # 可插拔编码器（grayscale_grid/ascii_art/color_stats）
├── ocr/                 # 可插拔 OCR（easyocr 默认 / paddleocr 占位）
└── synthetic.py         # 合成测试图生成
tests/                   # pytest 单测（全部基于合成图）
demo/                    # demo 脚本与输出
```

## 测试

```bash
uv run pytest -q
```

全部测试基于合成图，可复现；EasyOCR 用例在模型下载失败时自动跳过。
