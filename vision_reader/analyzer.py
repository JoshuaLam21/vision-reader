"""一键自动分析：用户只需给一张图，内部自动完成全流程并输出完整报告。

流程（无需用户/模型调度）：
1. 全图 overview 概览
2. 按边缘密度/颜色方差自动挑选值得细看的区域（合并重叠）
3. 对每个区域自动选择编码器并编码细看
4. 对疑似含文字的区域自动 OCR
5. 汇总为完整 Markdown 报告
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import crop as crop_mod
from . import overview
from .encoders import encode
from .ocr import recognize
from .report import Observation, build


@dataclass
class RegionDetail:
    """一个被自动选中并细看的区域。"""

    label: str
    region: tuple[float, float, float, float]  # 归一化 (x1, y1, x2, y2)
    encoder: str
    encoded: str
    edge_density: float
    ocr_text: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """一次自动分析的完整结果。"""

    overview_text: str
    regions: list[RegionDetail]
    title: str = "图片理解报告"

    def to_report(self) -> str:
        """组装为完整 Markdown 报告。"""
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
        return build(observations, title=self.title)


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

def analyze(
    img: np.ndarray,
    grid: tuple[int, int] = (8, 8),
    top_n: int = 3,
    ocr: bool = True,
    languages: tuple[str, ...] = ("ch_sim", "en"),
    title: str = "图片理解报告",
    edge_threshold: float = 0.12,
) -> AnalysisResult:
    """对整图做一次自动分析，返回完整结果（含 Markdown 报告）。"""
    chunks = overview.chunks(img, grid=grid)
    overview_text = overview.overview_text(chunks, grid=grid)

    regions: list[RegionDetail] = []
    for bbox in select_regions(chunks, grid=grid, top_n=top_n):
        cr = crop_mod.region(img, *bbox, scale=2.0, grayscale=False)  # 保留彩色供统计/OCR
        x1, y1, x2, y2 = bbox

        # 区域统计：颜色方差 + 边缘密度（取覆盖块的平均）
        block_vars = [
            c.color_variance
            for c in chunks
            if c.x2 > x1 and c.x1 < x2 and c.y2 > y1 and c.y1 < y2
        ]
        color_variance = float(np.mean(block_vars)) if block_vars else 0.0
        edge_density = max((c.edge_density for c in chunks if c.x2 > x1 and c.x1 < x2 and c.y2 > y1 and c.y1 < y2), default=0.0)

        encoder = _pick_encoder(color_variance, edge_density)
        kwargs = {"name": encoder}
        if encoder == "ascii_art":
            kwargs["width"] = 56
        elif encoder == "grayscale_grid":
            kwargs["grid_width"] = 28
        encoded = encode(cr.image, **kwargs)

        ocr_text = ""
        if ocr and edge_density >= edge_threshold:
            try:
                result = recognize(cr.image, engine="easyocr", languages=languages)
                ocr_text = result.text
            except Exception:  # noqa: BLE001  OCR 失败不影响整体分析
                ocr_text = ""

        regions.append(
            RegionDetail(
                label=f"区域{len(regions) + 1}",
                region=bbox,
                encoder=encoder,
                encoded=encoded,
                edge_density=round(edge_density, 3),
                ocr_text=ocr_text,
                warnings=cr.warnings,
            )
        )

    return AnalysisResult(overview_text=overview_text, regions=regions, title=title)
