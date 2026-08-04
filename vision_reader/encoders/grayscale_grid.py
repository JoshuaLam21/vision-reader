"""灰度值网格编码器：把区域量化成数字网格，最接近 CNN 感受野的"原始像素"读数。"""

from __future__ import annotations

import numpy as np

from .base import Encoder, _block_mean, _to_gray, register


@register
class GrayscaleGridEncoder(Encoder):
    name = "grayscale_grid"
    description = "灰度值网格：把区域量化成 0~N 的灰度数字网格，信息无损失但 token 消耗大"

    def encode(
        self,
        image: np.ndarray,
        grid_width: int = 32,
        grid_height: int | None = None,
        levels: int = 10,
    ) -> str:
        """编码为数字网格文本。

        - ``grid_width`` / ``grid_height``：网格分辨率（每格是块内平均亮度）。
        - ``levels``：量化级数（默认 10，即 0~9；0=最暗，levels-1=最亮）。
        """
        if grid_width < 1 or grid_height is not None and grid_height < 1:
            raise ValueError("grid_width / grid_height 必须 >= 1")
        if levels < 2:
            raise ValueError("levels 必须 >= 2")

        gray = _to_gray(image)
        h, w = gray.shape
        if grid_height is None:
            grid_height = max(1, int(round(grid_width * h / w)))

        cell_h, cell_w = h / grid_height, w / grid_width
        pad = len(str(levels - 1))

        rows = []
        for gy in range(grid_height):
            y0, y1 = int(round(gy * cell_h)), int(round((gy + 1) * cell_h))
            cells = []
            for gx in range(grid_width):
                x0, x1 = int(round(gx * cell_w)), int(round((gx + 1) * cell_w))
                mean = _block_mean(gray, y0, y1, x0, x1)
                cells.append(int(round(mean * (levels - 1) / 255)))
            rows.append(" ".join(f"{v:0{pad}d}" for v in cells))

        header = (
            f"灰度网格 {grid_width}x{grid_height}，量化 {levels} 级"
            f"（0=最暗，{levels - 1}=最亮），图 {w}x{h}px，每格约 {cell_w:.1f}x{cell_h:.1f}px\n"
        )
        col_ruler = "   " + " ".join(f"{gx % 10}" for gx in range(grid_width)) + "\n"
        body = "\n".join(f"{gy:3d} " + row for gy, row in enumerate(rows))
        return header + col_ruler + body
