"""OCR 引擎测试。EasyOCR 首次运行需下载模型权重，不可用时自动跳过。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vision_reader.errors import OCRError
from vision_reader.ocr.base import get, names
from vision_reader.synthetic import find_font


@pytest.fixture(scope="session")
def easyocr_en():
    """初始化 EasyOCR（仅 en，加速下载）；失败则跳过整个模块相关用例。"""
    try:
        from vision_reader.ocr.easyocr_engine import EasyOcrEngine

        engine = EasyOcrEngine(languages=("en",))
        # 触发模型加载（下载权重）
        engine.recognize(np.zeros((40, 120, 3), np.uint8))
        return engine
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"EasyOCR 不可用（无网络下载模型或初始化失败）: {exc}")


def _text_image(text: str, width: int = 320, height: int = 90, font_size: int = 32) -> np.ndarray:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((12, (height - font_size) // 2), text, fill=(0, 0, 0), font=find_font(font_size))
    return np.asarray(img)


def test_registry_contains_easyocr():
    assert "easyocr" in names()
    assert "paddleocr" in names()


def test_unknown_engine():
    with pytest.raises(OCRError):
        get("no_such_engine")


def test_recognize_english(easyocr_en):
    arr = _text_image("Vision Reader")
    result = easyocr_en.recognize(arr)
    assert not result.is_empty()
    joined = " ".join(i.text for i in result.items).lower()
    assert "vision" in joined or "reader" in joined


def test_bbox_normalized(easyocr_en):
    arr = _text_image("Hello OCR")
    result = easyocr_en.recognize(arr)
    assert not result.is_empty()
    for item in result.items:
        x1, y1, x2, y2 = item.bbox
        assert 0.0 <= x1 <= x2 <= 1.0
        assert 0.0 <= y1 <= y2 <= 1.0


def test_empty_on_blank_image(easyocr_en):
    blank = np.full((80, 200, 3), 255, np.uint8)
    result = easyocr_en.recognize(blank)
    assert result.is_empty()


def test_paddle_placeholder_raises():
    from vision_reader.ocr.paddleocr_engine import PaddleOcrEngine

    with pytest.raises(OCRError):
        PaddleOcrEngine().recognize(np.zeros((10, 10, 3), np.uint8))
