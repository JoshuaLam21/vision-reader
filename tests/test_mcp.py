"""MCP server 测试：核心逻辑函数、image_id 缓存、工具注册。"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from vision_reader import mcp_server
from vision_reader.errors import CoordinateError, ImageLoadError


def test_tools_registered():
    async def _list():
        return await mcp_server.mcp.list_tools()

    tools = asyncio.run(_list())
    names = {t.name for t in tools}
    assert {
        "vision_analyze",
        "vision_load_image",
        "vision_overview",
        "vision_crop",
        "vision_ocr",
        "vision_list_encoders",
        "vision_list_ocr_engines",
    } <= names


def test_resolve_image_cache(test_image):
    mcp_server._IMAGES.clear()
    mcp_server._IMAGES["imgX"] = test_image
    assert mcp_server._resolve_image("imgX") is test_image


def test_resolve_image_missing():
    mcp_server._IMAGES.clear()
    with pytest.raises(ImageLoadError):
        mcp_server._resolve_image("no_such_id_or_path")


def test_parse_region():
    assert mcp_server._parse_region("0.1,0.2,0.3,0.4") == (0.1, 0.2, 0.3, 0.4)
    with pytest.raises(ValueError):
        mcp_server._parse_region("0.1,0.2")
    with pytest.raises(ValueError):
        mcp_server._parse_region("0.1,0.2,0.3")


def test_do_overview(test_image):
    text = mcp_server._do_overview(test_image, (4, 4))
    assert "全图概览（4x4 chunk" in text
    assert "亮度/边缘速览" in text


def test_do_crop(test_image):
    text = mcp_server._do_crop(test_image, (0.0, 0.0, 0.5, 0.5), 2.0, "ascii_art", 0, False)
    assert "区域（归一化坐标）" in text
    assert "ASCII 明暗图" in text


def test_do_crop_grayscale_grid_with_size(test_image):
    text = mcp_server._do_crop(test_image, (0.0, 0.0, 0.5, 0.5), 1.0, "grayscale_grid", 16, False)
    assert "灰度网格 16x" in text


def test_do_crop_degenerate_region_rejected(test_image):
    with pytest.raises(CoordinateError):
        mcp_server._do_crop(test_image, (0.5, 0.5, 0.5, 0.5), 1.0, "ascii_art", 0, False)


@pytest.fixture(scope="module")
def ocr_engine_en():
    """EasyOCR en 引擎（复用模块级缓存）；不可用时跳过。"""
    try:
        engine = mcp_server._get_engine("easyocr", ("en",))
        # 触发模型加载
        engine.recognize(np.zeros((40, 120, 3), np.uint8))
        return engine
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"EasyOCR 不可用: {exc}")


def test_do_ocr(ocr_engine_en, test_image):
    from PIL import Image, ImageDraw

    from vision_reader.synthetic import find_font

    img = Image.new("RGB", (320, 90), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((12, 30), "Hello MCP", fill=(0, 0, 0), font=find_font(32))
    arr = np.asarray(img)

    result = mcp_server._do_ocr(arr, None, "easyocr", ("en",))
    assert result["count"] >= 1
    assert "hello" in result["text"].lower() or "mcp" in result["text"].lower()
    # bbox 归一化
    for item in result["items"]:
        x1, y1, x2, y2 = item["bbox"]
        assert 0.0 <= x1 <= x2 <= 1.0
        assert 0.0 <= y1 <= y2 <= 1.0


def test_do_ocr_with_region(ocr_engine_en, test_image):
    result = mcp_server._do_ocr(test_image, (0.0, 0.0, 0.5, 0.5), "easyocr", ("en",))
    assert "region" in result
    x1, y1, x2, y2 = result["region"]
    assert (x1, y1, x2, y2) == (0.0, 0.0, 0.5, 0.5)
