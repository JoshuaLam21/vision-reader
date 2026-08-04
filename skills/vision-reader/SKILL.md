---
name: vision-reader
description: 让无图像能力的模型"看图"：分析截图/文档/图表/UI 界面，输出像素统计报告与 OCR 文字。当用户要求看图、分析图片、描述截图、理解图表、提取图片中的文字时使用。
---

# vision-reader：给模型看图的能力

把图片转成文本统计报告，让没有图像能力的模型理解图片内容。**推荐流程：先一键分析拿报告，报告不够再分步细看。**

## 何时使用

- 用户要求"看/分析/描述/理解这张图（截图、文档、图表、UI 界面）"
- 用户提供了图片路径或 base64
- 需要提取图中的文字
- 注意：只处理图片，PDF 需先转成图片

## 怎么调用（按优先顺序）

### 1. 一键分析（首选）

```
vision_analyze(image=<图片路径或 image_id>, summary=false)
```

- 返回完整 Markdown 报告：整体概览 + 局部细节 + OCR 文字 + 坐标索引
- **大图或上下文紧张时**：`summary=true`（省 ~80% token，只保留布局要点+区域要点+OCR 汇总）

### 2. 分步细看（报告不够时）

```
vision_load_image(image_path=<路径>)          # 注册图片，拿 image_id（后续复用）
vision_overview(image=<image_id>)             # 看全图 chunk 概览，找值得细看的区域
vision_crop(image=<image_id>, region="x1,y1,x2,y2", encoder="ascii_art")  # 裁剪放大细看
vision_ocr(image=<image_id>, region="x1,y1,x2,y2")                        # 提取文字
```

## 关键规则

- **坐标都是归一化 (0~1)**：`(0,0)`=左上角，`(1,1)`=右下角
- `region` 参数是字符串，格式 `"x1,y1,x2,y2"`
- 编码器选择：结构/布局 → `ascii_art`（省 token）；颜色 → `color_stats`；像素级 → `grayscale_grid`
- **OCR 可能识别错（中文尤其）**，要结合上下文判断，不要盲信
- **ASCII/灰度图只表示形状与布局线索，不是真实图像**；颜色看报告中的主色 hex

## 解读报告

- 报告开头有"给模型的导读"，先读它再解读各段
- 整体概览表：每块含主色/亮度/边缘密度/颜色方差——边缘密度高（>0.15）≈ 文字/线条/纹理
- 局部细节：ASCII/灰度/色块编码文本
- OCR 段：识别出的文字

## 失败处理

- 返回 `错误: ...` 文本时：检查图片路径是否存在、坐标是否在 0~1 范围
- OCR 空结果：可能是图片模糊或区域无文字；可提示用户提供更清晰的图，或对区域调高 scale 重试
- 报告信息不足：用 `vision_crop` 对具体坐标放大细看，或换编码器重看

## bash 方式（无 MCP 时）

用 CLI 命令替代工具调用（把 `<项目路径>` 换成实际路径）：

```
uv run --directory <项目路径> vision analyze <图片> --out report.md
uv run --directory <项目路径> vision crop <图片> --region x1,y1,x2,y2 --encode ascii_art
uv run --directory <项目路径> vision ocr <图片> --region x1,y1,x2,y2
```
