"""归一化坐标 (0~1) ↔ 像素坐标换算，含越界 clamp 与合法性校验。"""

from __future__ import annotations

import numbers

from .errors import CoordinateError

Num = numbers.Real


def _check_norm(value: Num, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Num):
        raise CoordinateError(f"{name} 必须是数值，得到 {type(value).__name__}")
    if value < 0.0 or value > 1.0:
        raise CoordinateError(f"{name} 必须在 [0, 1] 范围内，得到 {value}")


def clamp_region(
    x1: Num, y1: Num, x2: Num, y2: Num, width: int, height: int
) -> tuple[int, int, int, int, bool]:
    """归一化坐标 → 像素 bbox，越界 clamp。

    返回 ``(px1, py1, px2, py2, clamped)``，其中 clamped=True 表示发生了越界修正
    （例如 x1==x2 之类退化区域被拒绝时会抛 CoordinateError）。
    归一化坐标必须在 [0,1]；像素结果保证落在 [0, width]×[0, height] 且宽高至少 1px。
    """
    _check_norm(x1, "x1")
    _check_norm(y1, "y1")
    _check_norm(x2, "x2")
    _check_norm(y2, "y2")

    raw = (
        x1 * width,
        y1 * height,
        x2 * width,
        y2 * height,
    )
    px1, py1, px2, py2 = (int(round(v)) for v in raw)

    # 反向区域（x1>x2 或 y1>y2）归一化为正向
    if px1 > px2:
        px1, px2 = px2, px1
    if py1 > py2:
        py1, py2 = py2, py1

    clamped = px1 < 0 or py1 < 0 or px2 > width or py2 > height
    px1, py1 = max(0, min(px1, width)), max(0, min(py1, height))
    px2, py2 = max(0, min(px2, width)), max(0, min(py2, height))

    if px2 - px1 < 1 or py2 - py1 < 1:
        raise CoordinateError("裁剪区域无效：换算后宽或高为 0（区域过小或全在图像外）")
    return px1, py1, px2, py2, clamped


def norm_box(px1: int, py1: int, px2: int, py2: int, width: int, height: int) -> tuple[float, float, float, float]:
    """像素 bbox → 归一化 bbox（用于回显实际裁剪区域）。"""
    return (px1 / width, py1 / height, px2 / width, py2 / height)
