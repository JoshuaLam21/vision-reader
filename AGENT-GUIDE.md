# 接入 AI Agent 指南（AGENT-GUIDE）

本文档回答一个问题：**一个 AI agent 要调用 vision-reader，它需要知道什么？** 写给想把 vision-reader 接入自己 agent（Claude Code / Reasonix / Cursor / 自研 agent）的开发者。

配套文件：`skills/vision-reader/SKILL.md`（可直接复制进 agent 的 skill 目录的现成 skill）。

---

## 一、知识卡：agent 需要知道的六层信息

### ① 工具存在性（知道有什么）

MCP 方式（7 个工具）：

| 工具 | 作用 |
|---|---|
| `vision_analyze` | **一键分析**：概览+选区域+编码+OCR → 完整 Markdown 报告（`summary=True` 出精简摘要） |
| `vision_load_image` | 注册图片（路径/base64）→ 返回 `image_id` |
| `vision_overview` | 全图 chunk 概览（主色/亮度/边缘密度/颜色方差 + 归一化坐标） |
| `vision_crop` | 按归一化坐标裁剪 + 编码（ascii_art / grayscale_grid / color_stats） |
| `vision_ocr` | 区域文字识别（文本 + 置信度 + bbox） |
| `vision_list_encoders` / `vision_list_ocr_engines` | 查询可用编码器 / OCR 引擎 |

bash 方式（2 个命令）：`vision analyze` / `vision crop` / `vision ocr`（详见 README）。

### ② 触发条件（知道何时用）

- 用户要求"看/分析/描述/理解这张图（截图、文档、图表、UI 界面）"
- 用户提供了图片路径、base64
- 需要从图中提取文字
- **边界**：只处理图片；PDF 需先转成图片再喂入

### ③ 调用策略（知道怎么用最有效）

1. **首选 `vision_analyze` 一次搞定**——用户只需要结果时
2. **大图或 token 紧张 → `summary=True`**（实测省 ~80% token）
3. **需要精确细节 → 分步**：`vision_load_image` → `vision_overview` 看概览 → `vision_crop(region="x1,y1,x2,y2")` 细看 → `vision_ocr` 提文字
4. 编码器选择：结构/布局用 `ascii_art`（省 token），颜色判断用 `color_stats`，像素级用 `grayscale_grid`

### ④ 输出解读（知道报告是什么）

- 报告是 Markdown，自带 **"给模型的导读"**：ASCII 字符含义、坐标归一化、边缘密度解读、OCR 免责声明
- **ASCII/灰度图只给形状与布局线索，不是图像本身**——模型应结合常识推断，不要当成真实像素
- 颜色以主色 hex/RGB 给出；`color_stats` 的表格可直接读
- OCR 结果可能识别错（中文尤其），要结合上下文判断

### ⑤ 边界与坑

- 失败时工具返回 `错误: ...` 文本，不是异常——agent 应读取并转述原因
- 图片模糊 → OCR 空结果/乱码
- 首次调用慢（模型加载）；MCP server 已做启动预热
- 颜色信息在 `ascii_art` / `grayscale_grid` 中丢失（灰度化）——要颜色用 `color_stats`

### ⑥ 兜底（不够怎么办）

- 报告信息不足 → `vision_crop` 指定坐标放大细看
- 需要更多文字 → `vision_ocr` 指定区域
- 换观察视角 → 换 encoder 重看同一区域

---

## 二、两种接入方式对照

| | MCP 方式（推荐） | bash 方式 |
|---|---|---|
| agent 怎么知道工具 | 工具描述自动注入（list_tools） | 必须把 SKILL.md / prompt 写进 agent 配置 |
| 安装 | `uv sync --extra mcp` | `uv sync --extra dev --extra mcp` |
| 注册 | `.mcp.json` 一行/几行 | 无注册，直接调命令 |
| 适用 | Claude Code / Reasonix / Cursor 等支持 MCP 的客户端 | 任何能跑 shell 的 agent |

MCP 注册与全局安装步骤见 README「MCP Server」章节。

---

## 三、给 agent 的 prompt 模板（bash 方式）

没有 MCP 时，把下面这段写进 agent 的系统提示 / skill 配置（把 `<项目路径>` 换成实际路径）：

```markdown
你有"看图"能力（vision-reader）。
- 当用户给图片路径或要求"看图/分析图片/描述截图/提取图中文字"时，执行：
  uv run --directory <项目路径> vision analyze <图片路径> --out report.md
- 先读 report.md 里的"给模型的导读"，再解读各段（概览表/编码细节/OCR/坐标索引）。
- 报告不够细时，裁剪细看：uv run --directory <项目路径> vision crop <图片> --region x1,y1,x2,y2 --encode ascii_art
- 需要读文字时：uv run --directory <项目路径> vision ocr <图片> --region x1,y1,x2,y2
- 坐标是归一化 (0~1)，(0,0) 左上角，(1,1) 右下角；region 格式 "x1,y1,x2,y2"。
- OCR 可能识别错，请结合上下文判断。
- 大图或上下文紧张时加 --summary（token 省 ~80%）。
- 命令失败时把"错误: ..."转述给用户，并检查图片路径与坐标范围。
```

---

## 四、调用序列示例

**场景 1：用户给一张截图问"这是什么页面？"**

```
vision_analyze(image=截图.png)          # 一键，默认完整报告
→ 报告含：主色布局、区域要点、OCR 文字
→ agent 组织回答："这是 xx 页面，包含标题 xx、按钮 xx……"
```

**场景 2：用户问"这张图表里最大的数字是多少？"**

```
vision_analyze(image=图.png, summary=True)     # 先看摘要找数字区域
vision_crop(image=图.png, region="0.1,0.1,0.5,0.5", encoder=grayscale_grid)  # 放大看细节
vision_ocr(image=图.png, region="0.1,0.1,0.5,0.5")                            # 读数字
```

**场景 3：token 预算有限（长会话）**

```
vision_analyze(image=图.png, summary=True)     # 只拿布局要点 + OCR 汇总
```

---

## 五、常见问题

**Q：agent 说找不到工具？** 检查 `.mcp.json` 的 command/args 是否正确、项目路径是否存在、客户端是否重启。

**Q：报告里 ASCII 图看不懂？** 报告开头有导读；也可以换 `grayscale_grid`（数字）或 `color_stats`（表格）重看同一区域。

**Q：OCR 识别错字？** 中文合成字体识别率低，真实截图会好很多；可对同一区域调高分辨率重试，或换 `--languages`。
