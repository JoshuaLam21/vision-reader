"""OCR 引擎接口、结果结构与注册表（可插拔）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..errors import OCRError

_ENGINES: dict[str, type["OcrEngine"]] = {}


@dataclass
class TextItem:
    """一条识别出的文本。bbox 为归一化坐标 (x1, y1, x2, y2)。"""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class OcrResult:
    """一次识别的结果。"""

    engine: str
    items: list[TextItem]

    @property
    def text(self) -> str:
        """按识别顺序拼接全部文本（每行一条）。"""
        return "\n".join(item.text for item in self.items)

    def is_empty(self) -> bool:
        return not self.items


class OcrEngine(ABC):
    """OCR 引擎基类。子类设置 ``name`` 并实现 ``recognize``。"""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OcrResult:
        """识别 ``image``（RGB uint8）中的文字。"""


def register(cls: type[OcrEngine]) -> type[OcrEngine]:
    if not cls.name or cls.name == "base":
        raise OCRError(f"OCR 引擎 {cls.__name__} 必须设置非空 name")
    _ENGINES[cls.name] = cls
    return cls


def get(name: str, **init_options) -> OcrEngine:
    """按名称构造引擎实例；init_options 传给构造函数（如 languages）。"""
    if name not in _ENGINES:
        raise OCRError(f"未知 OCR 引擎 '{name}'，可用: {', '.join(names())}")
    return _ENGINES[name](**init_options)


def names() -> list[str]:
    return sorted(_ENGINES)


def recognize(image: np.ndarray, engine: str = "easyocr", **init_options) -> OcrResult:
    """便捷入口：按名称构造引擎并识别。"""
    return get(engine, **init_options).recognize(image)
