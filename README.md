# vision-reader

给**没有图像能力的 LLM** 的"看图能力"工具：把图片的像素 + 坐标整理成模型可读的**文本**，让模型理解复杂图片（截图、文档、图表、UI 界面等）。

**用法一句话：`vision analyze 你的图片.png`，其余全部自动。** 内部自动完成 全图概览 → 自动挑选关注区域 → 编码细看 → OCR → 输出完整 Markdown 报告，不需要任何手动调度。

---

## 安装（约 2 分钟）

### 第 1 步：安装 uv（Python 包管理器，会自动带上 Python，无需手动装 Python）

**Windows**（任选其一）：

```powershell
winget install astral-sh.uv
```

或

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux**：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

装完重开终端，输入 `uv --version` 能输出版本号即成功。

### 第 2 步：获取项目并安装依赖

```bash
git clone https://github.com/JoshuaLam21/vision-reader.git
cd vision-reader
uv sync --extra dev --extra mcp
```

> ⏳ 首次安装需要下载 torch / easyocr 等依赖，**约几百 MB，请耐心等几分钟**，进度条走完即可。

### 第 3 步：验证安装成功

```bash
uv run vision demo
```

看到终端输出一份 Markdown 报告、并在 `demo/output/report.md` 生成文件，说明安装成功。

---

## 快速开始（30 秒上手）

### 看你自己的一张图

```bash
uv run vision analyze 你的图片.png --out report.md
```

打开 `report.md`，就是完整的图片理解报告（整体概览 + 局部细节 + OCR 文字 + 坐标索引）。

> 首次运行会自动下载 OCR 模型权重（约 60 MB，一次性，存到 `~/.EasyOCR`），请耐心等待。

### 没有图片？跑个演示

```bash
uv run vision demo
```

自动生成一张合成图并分析，输出到 `demo/output/`。

### 常用命令速查

| 想做什么 | 命令 |
|---|---|
| 一键分析图片（主用法） | `uv run vision analyze <图片>` |
| 分析并保存报告 | `uv run vision analyze <图片> --out report.md` |
| 只看全图概览 | `uv run vision overview <图片>` |
| 裁剪某区域细看 | `uv run vision crop <图片> --region 0.1,0.1,0.5,0.5` |
| 提取某区域文字 | `uv run vision ocr <图片> --region 0.1,0.1,0.5,0.5` |
| 跑完整演示 | `uv run vision demo` |

所有坐标都是**归一化坐标 (0~1)**，`0,0` 是左上角，`1,1` 是右下角。

---

## 常见问题（FAQ）

**Q：`uv` 命令找不到？**
安装 uv 后需要重开终端（让 PATH 生效）；Windows 用户也可以重启一下终端窗口。

**Q：安装/下载很慢？**
首次 `uv sync` 要下载 torch（几百 MB），首次 `analyze` 要下载 OCR 模型（约 60 MB）。都是**一次性**的，之后秒开。网络慢可考虑配置镜像源。

**Q：`vision analyze` 报错？**
先跑 `uv run vision demo` 确认环境正常；如果 demo 正常而你自己的图报错，可能是图片路径含中文/空格——用引号包起来：`uv run vision analyze "我的 图片.png"`。

**Q：我想让 Claude / 我的 AI 直接"看图"？**
见下方 **MCP Server** 章节，配置一次后，AI 就能调用 `vision_analyze` 工具看你的图。

---

## CLI 用法

### 一键分析（推荐，主用法）

```bash
uv run vision analyze <图片> [--grid 8x8] [--top 3] [--out report.md] [--summary] [--no-ocr]
```

| 参数 | 说明 |
|---|---|
| `--grid NxM` | 全图概览网格，默认 8x8 |
| `--top N` | 自动细看的区域数，默认 3 |
| `--out 路径` | 报告保存路径（默认仅打印） |
| `--summary` | 输出 token 精简摘要（布局要点+区域要点+OCR 汇总，约省 80% token；适合大图/长上下文受限） |
| `--no-ocr` | 跳过 OCR |

内部自动流程：全图概览 → 按边缘密度排序挑选候选区域、膨胀并合并重叠 → 每个区域自动选编码器（颜色丰富用 `color_stats`，否则 `ascii_art`）→ 疑似文字区域自动 OCR → 汇总 Markdown 报告。

**报告开头自带"给模型的导读"**：说明坐标归一化、ASCII 字符含义、边缘密度解读、OCR 可能出错等，帮助模型正确解读像素统计，无需额外提示。

### 分步用法（高级）

```bash
uv run vision overview <图片> --grid 8x8          # 1. 全图 chunk 概览
uv run vision crop <图片> --region 0.1,0.1,0.5,0.5 --encode ascii_art --size 64   # 2. 裁剪+编码
uv run vision ocr <图片> --region 0.02,0.42,0.6,0.58 --engine easyocr --languages ch_sim,en  # 3. OCR
uv run vision report obs1.json obs2.json --out report.md   # 4. 汇总报告
```

`crop` 参数：`--scale` 放大倍数（默认 2.0）、`--encode` 编码器（`ascii_art` / `grayscale_grid` / `color_stats`）、`--color` 保留彩色、`--json` 输出 JSON 供 `report` 汇总。

---

## 三种编码器（模型按需自选）

| 名称 | 输出 | 特点 |
|---|---|---|
| `grayscale_grid` | 灰度数字网格（量化级数可调） | 最接近原始像素读数，token 消耗大 |
| `ascii_art` | 字符明暗密度图 | token 最省、最易读 |
| `color_stats` | 分块主色/亮度/边缘密度统计 | 适合颜色与结构判断 |

编码器可插拔：实现 `vision_reader/encoders/base.py` 的 `Encoder` 接口并加 `@register` 即可。

---

## Python 库用法

```python
from vision_reader import image_io
from vision_reader.analyzer import analyze

img = image_io.load_image("截图.png")          # 支持路径 / base64 / bytes / ndarray
result = analyze(img, top_n=3, ocr=True)       # 一键分析（无需调度）
print(result.to_report())                      # 完整 Markdown 报告

# 需要精细控制时，可用分步 API：
from vision_reader import crop, overview
from vision_reader.encoders import encode
from vision_reader.ocr import recognize

chunks = overview.chunks(img, grid=(8, 8))     # 全图概览
cr = crop.region(img, 0.1, 0.1, 0.5, 0.5, scale=2.0)  # 裁剪放大
print(encode(cr.image, name="grayscale_grid", grid_width=32))  # 编码细看
result = recognize(cr.image, engine="easyocr", languages=("ch_sim", "en"))  # OCR
```

---

## OCR 引擎

- **默认 EasyOCR（ch_sim + en）**：pip 可装、纯 Python 生态、无需外部服务；首次使用会自动下载模型权重到 `~/.EasyOCR`。
- **PaddleOCR（可选占位）**：中文效果更优但依赖 PaddlePaddle（体积大）。当前为占位实现，接口已就绪：补全 `vision_reader/ocr/paddleocr_engine.py` 后，一行切换 `--engine paddleocr`。

---

## MCP Server（让 AI 直接看图）

把 vision-reader 封装为 MCP 工具，配置一次后，Claude Code / Reasonix / Cursor 等支持 MCP 的客户端里的 AI 就能调用工具看图。

### 第 0 步（可选）：全局安装，注册时不用写路径

```bash
uv tool install . --extra mcp   # 全局安装 vision / vision-mcp 命令（含依赖，约几百 MB，一次性）
vision-mcp                     # 测试：启动后不退出、无报错即正常（Ctrl+C 停止）
```

装完后 `vision-mcp` 是全局命令，注册配置只需一行（见第 3 步"方式一"），任何目录都能用。

### 第 1 步：确认已装 mcp 依赖

```bash
uv sync --extra mcp        # 安装过（--extra dev --extra mcp）则跳过
```

### 第 2 步：测试 server 能启动

```bash
uv run python -m vision_reader.mcp_server
```

启动后不退出、无报错即为正常（按 `Ctrl+C` 停止）。

### 第 3 步：注册到客户端

**方式一：全局命令（推荐，做了第 0 步后适用）**——Claude Code 项目根目录 `.mcp.json`：

```json
{
  "mcpServers": {
    "vision-reader": {
      "command": "vision-mcp"
    }
  }
}
```

**方式二：项目路径**（没做第 0 步；把 `<项目路径>` 换成你的实际路径）：

```json
{
  "mcpServers": {
    "vision-reader": {
      "command": "uv",
      "args": ["--directory", "<项目路径>", "run", "python", "-m", "vision_reader.mcp_server"]
    }
  }
}
```

**Reasonix / Cursor**：同样以 stdio server 方式注册，指向上述 command/args。

### 工具清单

| 工具 | 说明 |
|---|---|
| `vision_analyze` | **【一键】自动分析整图**：概览+选区域+编码+OCR，返回完整 Markdown 报告（`summary=True` 返回精简摘要） |
| `vision_load_image` | 注册图片（路径或 base64），返回 `image_id` + 宽高 |
| `vision_overview` | 全图 chunk 概览（默认 8x8，可调 grid） |
| `vision_crop` | 按归一化坐标裁剪 + 编码（`region="x1,y1,x2,y2"`，编码器可换） |
| `vision_ocr` | 区域 OCR（返回文本 + 置信度 + 归一化 bbox） |
| `vision_list_encoders` / `vision_list_ocr_engines` | 查询可用的编码器 / OCR 引擎 |

**给 AI 的使用建议**：用户说"看这张图"，直接调用 `vision_analyze` 一次即可，无需任何调度；需要精细控制时再用 `vision_load_image` + 分步工具（后续工具只传 `image_id`，避免重复传 base64 浪费 token）。OCR 引擎在 server 启动时预热，首次调用不卡顿。

### 验证接入成功

注册后在客户端里对 AI 说：**"分析这张图：<图片路径>"**。AI 应自动调用 `vision_analyze` 并返回一份 Markdown 报告（含整体概览、局部细节、OCR 文字、坐标索引）。如果 AI 说找不到工具，检查 `.mcp.json` 的 command/args 和路径是否正确，重启客户端生效。

---

## 接入你的 AI Agent（不用 MCP，bash 方式）

不想用 MCP 时，CLI 本身即工具形态，任何 agent 都可以通过 bash 直接调用。做法：把下面的说明写进你的 agent 的 skill / 命令配置：

```markdown
# 看图的工具（vision-reader）
当需要理解一张图片时：
1. `uv run vision analyze <img> --out report.md` → 一键自动分析，读 report.md 理解图片
2. 需要更细看某区域时：`uv run vision crop <img> --region x1,y1,x2,y2 --encode ascii_art`
3. 需要读文字时：`uv run vision ocr <img> --region x1,y1,x2,y2`
```

前提：agent 运行环境已安装 vision-reader（`uv sync --extra dev --extra mcp`，或全局 `uv tool install . --extra mcp` 后直接用 `vision analyze`）。要点：坐标一律用归一化 (0~1)；token 预算有限时优先 `ascii_art`。

---

## 项目结构

```
vision_reader/
├── image_io.py          # 图片加载（路径/base64/bytes/ndarray）
├── coordinates.py       # 归一化坐标 ↔ 像素换算、越界处理
├── crop.py              # 裁剪 + LANCZOS 放大 + 灰度化
├── overview.py          # 全图 chunk 网格摘要
├── analyzer.py          # 一键自动分析（概览→选区域→编码→OCR→报告）
├── report.py            # 多次观察汇总为 Markdown
├── cli.py               # CLI 入口（vision 命令）
├── mcp_server.py        # MCP server（FastMCP + stdio）
├── demo_runner.py       # demo 完整链路
├── encoders/            # 可插拔编码器（grayscale_grid/ascii_art/color_stats）
├── ocr/                 # 可插拔 OCR（easyocr 默认 / paddleocr 占位）
└── synthetic.py         # 合成测试图生成
tests/                   # pytest 单测（全部基于合成图）
demo/                    # demo 脚本与输出
```

---

## 测试

```bash
uv run pytest -q
```

全部测试基于合成图，可复现；EasyOCR 用例在模型下载失败时自动跳过。
