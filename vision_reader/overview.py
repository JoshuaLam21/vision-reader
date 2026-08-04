"""全图分块概览：把整图切成 N×N chunk，每块输出文本统计，模型据此决定细看哪些区域。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .encoders.color_stats import _dominant_color, _edge_density
from .encoders.base import _to_gray


@dataclass
class Chunk:
    """一个 chunk 的统计摘要。"""

    x1: float
    y1: float
    x2: float
    y2: float
    color_hex: str
    color_rgb: tuple[int, int, int]
    brightness: float  # 0~255
    edge_density: float  # 0~1
    color_variance: float  # 每通道方差均值


def chunks(image: np.ndarray, grid: tuple[int, int] = (8, 8)) -> list[Chunk]:
    """把 ``image`` 切成 grid=(rows, cols) 个 chunk，返回统计列表（行优先）。"""
    rows, cols = grid
    if rows < 1 or cols < 1:
        raise ValueError("grid 必须 >= 1x1")

    h, w = image.shape[:2]
    gray = _to_gray(image)
    cell_h, cell_w = h / rows, w / cols
    result: list[Chunk] = []

    for gy in range(rows):
        y0, y1 = int(round(gy * cell_h)), int(round((gy + 1) * cell_h))
        for gx in range(cols):
            x0, x1 = int(round(gx * cell_w)), int(round((gx + 1) * cell_w))
            block_rgb = image[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
            block_gray = gray[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]

            dom = _dominant_color(block_rgb)
            brightness = float(block_gray.mean())
            edge = _edge_density(block_gray)
            variance = float(np.mean([block_rgb[..., c].var() for c in range(3)]))

            result.append(
                Chunk(
                    x1=round(x0 / w, 4),
                    y1=round(y0 / h, 4),
                    x2=round(x1 / w, 4),
                    y2=round(y1 / h, 4),
                    color_hex="#{:02x}{:02x}{:02x}".format(dom[0], dom[1], dom[2]),
                    color_rgb=(int(dom[0]), int(dom[1]), int(dom[2])),
                    brightness=round(brightness, 1),
                    edge_density=round(edge, 3),
                    color_variance=round(variance, 1),
                )
            )
    return result


def overview_text(chunks_list: list[Chunk], grid: tuple[int, int] = (8, 8)) -> str:
    """把 chunk 列表格式化为 Markdown 表格文本。"""
    rows, cols = grid
    lines = [
        f"全图概览（{rows}x{cols} chunk 网格，共 {len(chunks_list)} 块）",
        "每块含归一化坐标范围 (x1,y1,x2,y2)、主色、平均亮度、边缘密度（结构复杂度）、颜色方差（颜色丰富度）。",
        "",
        "| 块 | 归一化范围 (x1,y1,x2,y2) | 主色 | 亮度 | 边缘密度 | 颜色方差 |",
        "|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(chunks_list):
        rgb = f"({c.color_rgb[0]},{c.color_rgb[1]},{c.color_rgb[2]})"
        lines.append(
            f"| {i} | ({c.x1},{c.y1},{c.x2},{c.y2}) | {rgb} {c.color_hex} "
            f"| {c.brightness:.0f} | {c.edge_density:.2f} | {c.color_variance:.0f} |"
        )
    return "\n".join(lines)


def overview_json(chunks_list: list[Chunk]) -> dict:
    """把 chunk 列表转为 JSON 友好的 dict。"""
    return {"chunks": [asdict(c) for c in chunks_list]}


def render_chunk_grid(chunks_list: list[Chunk], grid: tuple[int, int] = (8, 8)) -> str:
    """渲染一个二维"亮度/边缘"速览网格，帮助模型快速定位高信息量区域。

    每个格子显示两个字符：亮度（0-9）与边缘密度（. 低 / E 高）。
    """
    rows, cols = grid
    lines = ["亮度/边缘速览（每格: 亮度0-9 + E=边缘密集）:", ""]
    for gy in range(rows):
        line = []
        for gx in range(cols):
            c = chunks_list[gy * cols + gx]
            bright = min(9, int(c.brightness * 9 / 255))
            edge = "E" if c.edge_density > 0.15 else "."
            line.append(f"{bright}{edge}")
        lines.append(" ".join(line))
    return "\n".join(lines)
