"""区域裁剪：按归一化坐标裁剪 + LANCZOS 放大 + 可选灰度化。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import coordinates
from .errors import CropError


@dataclass
class CroppedRegion:
    """裁剪结果：放大后的图像 + 实际裁剪范围（像素与归一化）+ 告警。"""

    image: np.ndarray  # (H, W, 3) RGB 或 (H, W) 灰度
    pixel_box: tuple[int, int, int, int]  # 原图中的 (px1, py1, px2, py2)
    norm_box: tuple[float, float, float, float]  # clamp 后的归一化 (x1, y1, x2, y2)
    scale: float
    grayscale: bool
    warnings: list[str] = field(default_factory=list)


def region(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float = 2.0,
    grayscale: bool = True,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> CroppedRegion:
    """按归一化坐标 (0~1) 裁剪 ``image``（RGB uint8），可选放大并灰度化。

    - 坐标越界会被 clamp 到图像边界，并在返回结果的 ``warnings`` 中说明。
    - ``scale`` 必须为正数；放大插值默认 LANCZOS4。
    """
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise CropError(f"scale 必须是数值，得到 {type(scale).__name__}")
    if scale <= 0:
        raise CropError(f"scale 必须为正数，得到 {scale}")

    height, width = image.shape[:2]
    px1, py1, px2, py2, clamped = coordinates.clamp_region(x1, y1, x2, y2, width, height)

    sub = image[py1:py2, px1:px2]
    if grayscale:
        sub = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)

    if scale != 1.0:
        new_w = max(1, int(round(sub.shape[1] * scale)))
        new_h = max(1, int(round(sub.shape[0] * scale)))
        sub = cv2.resize(sub, (new_w, new_h), interpolation=interpolation)

    warnings = []
    if clamped:
        warnings.append("裁剪区域部分超出图像边界，已 clamp 到边界内")

    return CroppedRegion(
        image=sub,
        pixel_box=(px1, py1, px2, py2),
        norm_box=coordinates.norm_box(px1, py1, px2, py2, width, height),
        scale=scale,
        grayscale=grayscale,
        warnings=warnings,
    )
