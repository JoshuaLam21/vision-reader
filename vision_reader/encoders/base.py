"""编码器接口与注册表：把图像区域编码成模型可读的文本。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..errors import EncoderError

_REGISTRY: dict[str, type["Encoder"]] = {}


class Encoder(ABC):
    """编码器基类。子类设置 ``name`` / ``description`` 并实现 ``encode``。"""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def encode(self, image: np.ndarray, **options: Any) -> str:
        """把图像编码为模型可读的文本。"""


def register(cls: type[Encoder]) -> type[Encoder]:
    """类装饰器：把编码器注册进 registry。"""
    if not cls.name or cls.name == "base":
        raise EncoderError(f"编码器 {cls.__name__} 必须设置非空 name")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[Encoder]:
    if name not in _REGISTRY:
        raise EncoderError(f"未知编码器 '{name}'，可用: {', '.join(names())}")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


def encode(image: np.ndarray, name: str = "ascii_art", **options: Any) -> str:
    """便捷入口：按名称编码。"""
    return get(name)().encode(image, **options)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) → 灰度 (H, W) uint8。"""
    if image.ndim == 2:
        return image
    return np.round(
        0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    ).astype(np.uint8)


def _block_mean(gray: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    """灰度图某块的均值（保证非空块）。"""
    block = gray[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
    return float(block.mean())
