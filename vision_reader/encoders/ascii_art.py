"""ASCII 明暗密度编码器：字符从暗到亮，token 最省、最易读。"""

from __future__ import annotations

import numpy as np

from .base import Encoder, _block_mean, _to_gray, register


@register
class AsciiArtEncoder(Encoder):
    name = "ascii_art"
    description = "ASCII 明暗密度图：字符从暗到亮，token 最省、最易读"

    RAMP = " .:-=+*#%@"

    def encode(self, image: np.ndarray, width: int = 64, ramp: str | None = None) -> str:
        """编码为 ASCII 明暗图。

        - ``width``：输出字符宽度；高度按纵横比自动校正（字符近似宽:高 = 1:2）。
        - ``ramp``：明暗字符表，从左（暗）到右（亮），默认 `` .:-=+*#%@``。
        """
        if width < 1:
            raise ValueError("width 必须 >= 1")
        ramp = ramp or self.RAMP
        if not ramp:
            raise ValueError("ramp 不能为空")

        gray = _to_gray(image)
        h, w = gray.shape
        aspect = 0.5  # 字符高度约为宽度的 2 倍
        height = max(1, int(round((h / w) * width * aspect)))

        cell_h, cell_w = h / height, w / width
        rows = []
        for gy in range(height):
            y0, y1 = int(round(gy * cell_h)), int(round((gy + 1) * cell_h))
            line = []
            for gx in range(width):
                x0, x1 = int(round(gx * cell_w)), int(round((gx + 1) * cell_w))
                mean = _block_mean(gray, y0, y1, x0, x1)
                idx = min(int(mean * (len(ramp) - 1) / 255), len(ramp) - 1)
                line.append(ramp[idx])
            rows.append("".join(line))

        header = (
            f"ASCII 明暗图 {width}x{height}（字符 = 块内平均亮度），图 {w}x{h}px，"
            f"每格约 {cell_w:.1f}x{cell_h:.1f}px\n字符表（暗→亮）: {ramp}\n"
        )
        return header + "\n".join(rows)
