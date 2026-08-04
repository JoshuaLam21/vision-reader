"""PaddleOCR 引擎（可选后端，占位实现）。

中文效果优于 EasyOCR，但依赖 PaddlePaddle（体积大、可能有系统依赖）。
当前仅注册占位：调用时明确提示安装步骤。接口与 EasyOcrEngine 完全一致，
补全实现后即可一行切换：``ocr.recognize(img, engine="paddleocr")``。
"""

from __future__ import annotations

import numpy as np

from ..errors import OCRError
from .base import OcrEngine, OcrResult, register


@register
class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"
    description = "PaddleOCR 可选后端（占位）：中文效果更优，需额外安装 PaddlePaddle"

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def recognize(self, image: np.ndarray) -> OcrResult:
        raise OCRError(
            "PaddleOCR 后端为占位实现。启用步骤："
            "1) pip install paddlepaddle paddleocr；"
            "2) 在 vision_reader/ocr/paddleocr_engine.py 中实现 recognize()（接口见 easyocr_engine.py）。"
            "切换方式：ocr.recognize(img, engine='paddleocr')。"
        )
