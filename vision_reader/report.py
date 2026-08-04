"""报告汇总：把多次观察（overview / crop 编码 / OCR 结果）汇总为结构化 Markdown 报告。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """一次观察记录，由调用方（模型调度逻辑 / CLI / demo）收集。"""

    kind: str  # "overview" | "crop" | "ocr"
    label: str  # 显示名，如 "整体概览"、"区域A"
    content: str  # 文本内容（编码输出 / OCR 文本 / 概览表格）
    region: tuple[float, float, float, float] | None = None  # 归一化坐标 (x1,y1,x2,y2)
    meta: dict = field(default_factory=dict)  # 附加信息，如编码器名、引擎名、警告


def _format_region(region: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = region
    return f"({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})"


def build(observations: list[Observation], title: str = "图片理解报告") -> str:
    """把 observations 汇总为一份 Markdown 报告。

    结构：
    1. 整体概览（所有 kind=overview）
    2. 局部细节（所有 kind=crop，附坐标与编码器名）
    3. OCR 结果（所有 kind=ocr，附坐标与引擎名）
    4. 坐标索引表（全部带 region 的观察）
    """
    lines = [f"# {title}", "", "> 由 vision-reader 渐进式视觉观察生成", ""]

    overviews = [o for o in observations if o.kind == "overview"]
    crops = [o for o in observations if o.kind == "crop"]
    ocrs = [o for o in observations if o.kind == "ocr"]

    # 1. 整体概览
    lines.append("## 1. 整体概览")
    lines.append("")
    if overviews:
        for o in overviews:
            lines.append(o.content)
            lines.append("")
    else:
        lines.append("_（未采集整体概览）_")
        lines.append("")

    # 2. 局部细节
    lines.append("## 2. 局部细节")
    lines.append("")
    if crops:
        for i, o in enumerate(crops, start=1):
            lines.append(f"### 2.{i} {o.label}")
            lines.append("")
            if o.region:
                lines.append(f"- 归一化区域: `{_format_region(o.region)}`")
            if o.meta.get("encoder"):
                lines.append(f"- 编码器: `{o.meta['encoder']}`")
            for w in o.meta.get("warnings", []):
                lines.append(f"- ⚠️ {w}")
            lines.append("")
            lines.append("```")
            lines.append(o.content.rstrip("\n"))
            lines.append("```")
            lines.append("")
    else:
        lines.append("_（未采集局部细节）_")
        lines.append("")

    # 3. OCR 结果
    lines.append("## 3. OCR 结果")
    lines.append("")
    if ocrs:
        for i, o in enumerate(ocrs, start=1):
            lines.append(f"### 3.{i} {o.label}（引擎 `{o.meta.get('engine', '?')}`）")
            lines.append("")
            if o.region:
                lines.append(f"- 归一化区域: `{_format_region(o.region)}`")
            lines.append("")
            content = o.content.strip()
            if not content:
                lines.append("_（未识别到文字）_")
            else:
                lines.append("```")
                lines.append(content)
                lines.append("```")
            lines.append("")
    else:
        lines.append("_（未做 OCR）_")
        lines.append("")

    # 4. 坐标索引
    lines.append("## 4. 坐标索引")
    lines.append("")
    indexed = [o for o in observations if o.region]
    if indexed:
        lines.append("| 标签 | 归一化坐标 (x1, y1, x2, y2) | 类型 |")
        lines.append("|---|---|---|")
        for o in indexed:
            lines.append(f"| {o.label} | `{_format_region(o.region)}` | {o.kind} |")
    else:
        lines.append("_（无带坐标的观察）_")

    return "\n".join(lines) + "\n"
