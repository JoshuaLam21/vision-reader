"""色块统计编码器：把区域分成若干块，每块输出主色 / 亮度 / 边缘密度。"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Encoder, _to_gray, register


def _dominant_color(block_rgb: np.ndarray) -> np.ndarray:
    """块内主色：RGB 量化到 32 级/通道后取众数。"""
    q = (block_rgb // 32) * 32
    pixels = q.reshape(-1, 3)
    _, counts = np.unique(pixels, axis=0, return_counts=True)
    return pixels[np.argmax(counts)]


def _edge_density(block_gray: np.ndarray, threshold: int = 40) -> float:
    """边缘密度：Sobel 梯度幅值超过阈值的像素占比 (0~1)。"""
    if block_gray.size == 0:
        return 0.0
    gx = cv2.Sobel(block_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(block_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag > threshold))


@register
class ColorStatsEncoder(Encoder):
    name = "color_stats"
    description = "色块统计：分块输出主色/亮度/边缘密度，适合颜色与结构判断"

    def encode(self, image: np.ndarray, blocks: tuple[int, int] = (4, 4)) -> str:
        """编码为分块统计文本。

        - ``blocks``: (rows, cols) 分块数，默认 4x4。
        """
        rows, cols = blocks
        if rows < 1 or cols < 1:
            raise ValueError("blocks 必须 >= 1x1")

        h, w = image.shape[:2]
        gray = _to_gray(image)
        cell_h, cell_w = h / rows, w / cols

        lines = [f"色块统计 {rows}x{cols}，图 {w}x{h}px，每块约 {cell_w:.1f}x{cell_h:.1f}px"]
        lines.append("| 块 | 归一化范围 (x1,y1,x2,y2) | 主色 | 亮度 | 边缘密度 |")
        lines.append("|---|---|---|---|---|")

        for gy in range(rows):
            y0, y1 = int(round(gy * cell_h)), int(round((gy + 1) * cell_h))
            for gx in range(cols):
                x0, x1 = int(round(gx * cell_w)), int(round((gx + 1) * cell_w))
                block_rgb = image[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
                block_gray = gray[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]

                dom = _dominant_color(block_rgb)
                brightness = float(block_gray.mean())
                edge = _edge_density(block_gray)

                norm = (x0 / w, y0 / h, x1 / w, y1 / h)
                rgb_str = f"({dom[0]},{dom[1]},{dom[2]})"
                hex_str = "#{:02x}{:02x}{:02x}".format(dom[0], dom[1], dom[2])
                lines.append(
                    f"| r{gy}c{gx} | ({norm[0]:.3f},{norm[1]:.3f},{norm[2]:.3f},{norm[3]:.3f}) "
                    f"| {rgb_str} {hex_str} | {brightness:.0f} | {edge:.2f} |"
                )
        return "\n".join(lines)
