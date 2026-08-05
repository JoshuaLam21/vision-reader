"""一键自动分析：用户只需给一张图，内部自动完成全流程并输出完整报告。

流程（无需用户/模型调度）：
1. 全图 overview 概览
2. 按边缘密度/颜色方差自动挑选值得细看的区域（合并重叠）
3. 对每个区域自动选择编码器并编码细看
4. 对疑似含文字的区域自动 OCR
5. 汇总为完整 Markdown 报告
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import crop as crop_mod
from . import overview
from .encoders import encode
from .ocr import recognize
from .report import Observation, build

# 报告开头的"模型导读"：帮助模型正确解读像素统计输出
GUIDE = (
    "> **给模型的导读**：本报告由像素统计生成，不是图片本身，请据此结合常识推断。\n"
    "> - 坐标均为归一化 (0~1)，(0,0)=左上角，(1,1)=右下角\n"
    "> - ASCII 明暗图字符表（暗→亮）: ` .:-=+*#%@`；`%%`≈深色实体，`.`≈浅色空白\n"
    "> - 灰度网格：数字越大越亮（0=最暗）\n"
    "> - 边缘密度高（>0.15）≈ 该区域有文字/线条/纹理，值得注意\n"
    "> - OCR 文字可能含识别错误，请结合上下文判断\n"
)


@dataclass
class RegionDetail:
    """一个被自动选中并细看的区域。"""

    label: str
    region: tuple[float, float, float, float]  # 归一化 (x1, y1, x2, y2)
    encoder: str
    encoded: str
    edge_density: float
    color_hex: str = ""
    ocr_text: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """一次自动分析的完整结果。"""

    overview_text: str
    regions: list[RegionDetail]
    title: str = "图片理解报告"
    layout_points: list[str] = field(default_factory=list)  # 布局要点（摘要模式用）

    def to_report(self) -> str:
        """组装为完整 Markdown 报告（导读 + 四段式）。"""
        observations = [
            Observation(kind="overview", label="整体概览", content=self.overview_text),
        ]
        for i, r in enumerate(self.regions, start=1):
            observations.append(
                Observation(
                    kind="crop",
                    label=f"{r.label}（区域{i}）",
                    content=r.encoded,
                    region=r.region,
                    meta={"encoder": r.encoder, "warnings": r.warnings},
                )
            )
            if r.ocr_text:
                observations.append(
                    Observation(
                        kind="ocr",
                        label=f"{r.label} 文字",
                        content=r.ocr_text,
                        region=r.region,
                        meta={"engine": "auto"},
                    )
                )
        return GUIDE + "\n" + build(observations, title=self.title)

    def to_summary(self) -> str:
        """组装为 token 精简的摘要报告（导读 + 布局要点 + 区域要点表 + OCR 汇总）。"""
        lines = [
            f"# {self.title}（摘要）",
            "",
            GUIDE,
            "",
            "## 布局要点",
            "",
        ]
        lines.extend(f"- {p}" for p in self.layout_points)
        if not self.layout_points:
            lines.append("_（无显著布局特征）_")
        lines += ["", "## 区域要点", "", "| 区域 | 位置 | 主色 | 边缘密度 | 特征 |", "|---|---|---|---|---|"]
        for r in self.regions:
            box = tuple(round(v, 3) for v in r.region)
            feature = "疑似文字/纹理" if r.edge_density >= 0.15 else "普通区域"
            lines.append(f"| {r.label} | {box} | {r.color_hex or '-'} | {r.edge_density:.2f} | {feature} |")

        ocr_lines = [r for r in self.regions if r.ocr_text]
        lines += ["", "## OCR 汇总", ""]
        if ocr_lines:
            for r in ocr_lines:
                lines.append(f"- {r.label}（{tuple(round(v, 3) for v in r.region)}）: {r.ocr_text}")
        else:
            lines.append("_（未识别到文字）_")
        return "\n".join(lines) + "\n"


def _build_layout_points(chunks: list[overview.Chunk]) -> list[str]:
    """从 chunk 统计提炼布局要点：显著主色区域 + 高信息区域。"""
    points: list[str] = []
    total = len(chunks) or 1

    # 主色分布（排除占比过大的单一背景色也列出，供判断）
    color_count = Counter(c.color_hex for c in chunks)
    for hexv, n in color_count.most_common(3):
        if n / total < 0.05:
            break
        rep = next((c for c in chunks if c.color_hex == hexv), None)
        if rep is None:
            continue
        rgb = f"({rep.color_rgb[0]},{rep.color_rgb[1]},{rep.color_rgb[2]})"
        box = (round(rep.x1, 2), round(rep.y1, 2), round(rep.x2, 2), round(rep.y2, 2))
        points.append(f"主色 {hexv} {rgb} 约占 {n}/{total} 块，示例区域 {box}")

    # 高信息区域（文字/纹理密集）
    hot = sorted(chunks, key=lambda c: c.edge_density, reverse=True)[:3]
    for c in hot:
        if c.edge_density >= 0.15:
            box = (round(c.x1, 2), round(c.y1, 2), round(c.x2, 2), round(c.y2, 2))
            points.append(f"高信息区域 {box}：边缘密度 {c.edge_density:.2f}（疑似文字/线条/纹理）")
    return points


# ---- 自动选区域 ----

def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


def _bbox_union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def select_regions(
    chunks: list[overview.Chunk],
    grid: tuple[int, int] = (8, 8),
    top_n: int = 3,
    expand: float = 0.5,
    iou_threshold: float = 0.3,
) -> list[tuple[float, float, float, float]]:
    """按边缘密度排序挑选 top 区域，膨胀为区域并贪心合并重叠，返回归一化 bbox 列表。

    - 候选取 top_n * 2 个（合并后可能减少）
    - expand: 每个候选块向四周扩展的格子比例（默认 0.5 格）
    - iou_threshold: 重叠超过该 IoU 的候选合并为一个区域
    """
    rows, cols = grid
    cell_w, cell_h = 1.0 / cols, 1.0 / rows
    ranked = sorted(chunks, key=lambda c: c.edge_density, reverse=True)

    candidates: list[tuple[float, float, float, float]] = []
    for c in ranked[: max(1, top_n * 2)]:
        candidates.append(
            (
                max(0.0, c.x1 - cell_w * expand),
                max(0.0, c.y1 - cell_h * expand),
                min(1.0, c.x2 + cell_w * expand),
                min(1.0, c.y2 + cell_h * expand),
            )
        )

    regions: list[tuple[float, float, float, float]] = []
    for cand in candidates:
        merged = False
        for i, reg in enumerate(regions):
            if _iou(cand, reg) > iou_threshold:
                regions[i] = _bbox_union(cand, reg)
                merged = True
                break
        if not merged:
            regions.append(cand)
    return regions[:top_n]


# ---- 自动编码器 ----

def _pick_encoder(color_variance: float, edge_density: float) -> str:
    """启发式选编码器：颜色丰富 → color_stats；否则 ascii_art（token 省、易读）。"""
    if color_variance > 2500:
        return "color_stats"
    return "ascii_art"


# ---- 主入口 ----

def _text_regions(items, expand: float = 0.6) -> list[tuple[float, float, float, float]]:
    """把 OCR 结果的 bbox 扩展成观察区域，并按"行"合并。

    文字是"哪里值得看"的最强信号（真实 UI/文档/图表都含文字），
    优先于边缘密度猜测。expand 为相对文字大小的扩展比例。
    同一行（y 方向重叠）的文字合并为一个区域，避免导航栏/标题被切碎。
    """
    boxes: list[tuple[float, float, float, float]] = []
    for it in items:
        x1, y1, x2, y2 = it.bbox
        w, h = x2 - x1, y2 - y1
        boxes.append(
            (
                max(0.0, x1 - w * expand * 0.5),
                max(0.0, y1 - h * expand),
                min(1.0, x2 + w * expand * 0.5),
                min(1.0, y2 + h * expand),
            )
        )
    # 按 y 排序，y 方向重叠（同一行）则横向合并
    merged: list[tuple[float, float, float, float]] = []
    for b in sorted(boxes, key=lambda r: (r[1], r[0])):
        if merged and b[1] < merged[-1][3] and b[3] > merged[-1][1]:
            merged[-1] = _bbox_union(merged[-1], b)
        else:
            merged.append(b)
    return merged


def _ocr_text_in(items, region: tuple[float, float, float, float]) -> str:
    """取区域内的 OCR 文本（按 bbox 交集）。"""
    x1, y1, x2, y2 = region
    hits = [
        it.text
        for it in items
        if it.bbox[0] < x2 and it.bbox[2] > x1 and it.bbox[1] < y2 and it.bbox[3] > y1
    ]
    return "\n".join(hits)


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def analyze(
    img: np.ndarray,
    grid: tuple[int, int] = (8, 8),
    top_n: int = 3,
    ocr: bool = True,
    languages: tuple[str, ...] = ("ch_sim", "en"),
    title: str = "图片理解报告",
) -> AnalysisResult:
    """对整图做一次自动分析，返回完整结果（含 Markdown 报告）。

    区域选择是 **OCR 驱动**的：先对整图 OCR 一次，文字 bbox 直接作为观察区域
    （真实截图/文档/图表都含文字，比边缘密度猜测可靠得多），
    不足的坑位再用边缘密度补充非文字高信息区域。
    """
    chunks = overview.chunks(img, grid=grid)
    overview_text = overview.overview_text(chunks, grid=grid)
    layout_points = _build_layout_points(chunks)

    # 整图 OCR（只跑一次，结果按坐标复用于各区域）
    full_items = []
    if ocr:
        try:
            full_items = recognize(img, engine="easyocr", languages=languages).items
        except Exception:  # noqa: BLE001  OCR 失败不影响整体分析
            full_items = []

    # 区域来源 1：OCR 文字 bbox（最可靠）
    text_regions = _text_regions(full_items)
    # 区域来源 2：边缘密度高信息区（补充非文字区域）
    # 注意：文字区最多占 top_n-1 个，保证至少留一个坑位给"视觉内容区"
    # （3D 视图/图表/照片主体），否则真实截图的控件文字会把视觉区挤掉。
    edge_regions = select_regions(chunks, grid=grid, top_n=max(1, top_n))
    regions = list(text_regions[: max(0, top_n - 1)])
    for r in edge_regions:
        if len(regions) >= top_n:
            break
        if not any(_overlaps(r, t) for t in regions):
            regions.append(r)
    regions = regions[:top_n]

    region_details: list[RegionDetail] = []
    for bbox in regions:
        cr = crop_mod.region(img, *bbox, scale=2.0, grayscale=False)  # 保留彩色供统计
        x1, y1, x2, y2 = bbox

        # 区域统计：颜色方差 + 边缘密度 + 主色（取覆盖块）
        covered = [c for c in chunks if c.x2 > x1 and c.x1 < x2 and c.y2 > y1 and c.y1 < y2]
        block_vars = [c.color_variance for c in covered]
        color_variance = float(np.mean(block_vars)) if block_vars else 0.0
        edge_density = max((c.edge_density for c in covered), default=0.0)
        if covered:
            color_hex = Counter(c.color_hex for c in covered).most_common(1)[0][0]
        else:
            color_hex = ""

        encoder = _pick_encoder(color_variance, edge_density)
        kwargs = {"name": encoder}
        if encoder == "ascii_art":
            kwargs["width"] = 56
        elif encoder == "grayscale_grid":
            kwargs["grid_width"] = 28
        encoded = encode(cr.image, **kwargs)

        ocr_text = _ocr_text_in(full_items, bbox) if ocr else ""

        region_details.append(
            RegionDetail(
                label=f"区域{len(region_details) + 1}",
                region=bbox,
                encoder=encoder,
                encoded=encoded,
                edge_density=round(edge_density, 3),
                color_hex=color_hex,
                ocr_text=ocr_text,
                warnings=cr.warnings,
            )
        )

    return AnalysisResult(
        overview_text=overview_text,
        regions=region_details,
        title=title,
        layout_points=layout_points,
    )
