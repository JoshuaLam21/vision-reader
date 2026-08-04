"""合成测试图生成：不依赖真实图片，保证测试与 demo 可复现。"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(size: int = 28) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """找一个可用的中文字体（找不到则回退默认字体，中文会显示为方块）。"""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_image(width: int = 400, height: int = 300, title: str = "Vision Reader") -> np.ndarray:
    """生成一张确定性的合成图（RGB uint8 ndarray）：

    - 浅灰背景
    - 左上：红色标题条
    - 右上：蓝色圆
    - 底部：绿色条带
    - 中间：黑色文字 ``title``
    """
    img = Image.new("RGB", (width, height), (240, 240, 240))
    d = ImageDraw.Draw(img)

    # 左上红色标题条
    d.rectangle([20, 20, max(21, int(width * 0.5)), max(21, 70)], fill=(200, 40, 40))
    # 右上蓝色圆（自适应小尺寸，保证 y0 < y1）
    cx0, cy0 = int(width * 0.7), 30
    cx1 = max(cx0 + 1, int(width * 0.95))
    cy1 = max(cy0 + 1, int(height * 0.4))
    d.ellipse([cx0, cy0, cx1, cy1], fill=(40, 90, 200))
    # 底部绿色条带
    gy0 = int(height * 0.8)
    gy1 = max(gy0 + 1, int(height * 0.93))
    d.rectangle([20, gy0, int(width * 0.95), gy1], fill=(40, 160, 80))
    # 中间黑色文字
    font = find_font(max(20, height // 10))
    d.text((40, int(height * 0.45)), title, fill=(20, 20, 20), font=font)

    return np.asarray(img)
