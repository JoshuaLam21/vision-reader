"""OCR 适配层。导入本包即注册全部引擎。"""

from . import easyocr_engine  # noqa: F401  注册副作用
from . import paddleocr_engine  # noqa: F401
from .base import OcrEngine, OcrResult, TextItem, get, names, recognize

__all__ = ["OcrEngine", "OcrResult", "TextItem", "get", "names", "recognize"]
