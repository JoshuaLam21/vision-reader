"""analyzer 一键自动分析测试。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vision_reader import overview
from vision_reader.analyzer import AnalysisResult, RegionDetail, _pick_encoder, analyze, select_regions
from vision_reader.synthetic import find_font, make_image


def test_select_regions_bounds(test_image):
    chunks = overview.chunks(test_image, grid=(8, 8))
    regions = select_regions(chunks, grid=(8, 8), top_n=3)
    assert 1 <= len(regions) <= 3
    for x1, y1, x2, y2 in regions:
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0


def test_select_regions_prefers_text_area(test_image):
    """合成图中间有文字，自动选择应覆盖文字所在区域。"""
    chunks = overview.chunks(test_image, grid=(8, 8))
    regions = select_regions(chunks, grid=(8, 8), top_n=2)
    # 文字位于 y≈0.45-0.55，x 从 0.1 起；至少一个区域与文字区域重叠
    text_box = (0.05, 0.42, 0.6, 0.58)
    assert any(
        r[0] < text_box[2] and r[2] > text_box[0] and r[1] < text_box[3] and r[3] > text_box[1]
        for r in regions
    )


def test_analyze_structure(test_image):
    result = analyze(test_image, grid=(8, 8), top_n=3, ocr=False)
    assert isinstance(result, AnalysisResult)
    assert "全图概览" in result.overview_text
    assert len(result.regions) >= 1
    for r in result.regions:
        assert isinstance(r, RegionDetail)
        assert r.encoder in {"ascii_art", "color_stats", "grayscale_grid"}
        assert "ASCII" in r.encoded or "色块" in r.encoded


def test_analyze_report(test_image):
    result = analyze(test_image, grid=(8, 8), top_n=2, ocr=False)
    md = result.to_report()
    assert "# 图片理解报告" in md
    assert "## 1. 整体概览" in md
    assert "## 2. 局部细节" in md
    assert "## 4. 坐标索引" in md
    for r in result.regions:
        assert r.label in md


def test_report_contains_guide(test_image):
    """完整报告开头应包含模型导读。"""
    result = analyze(test_image, grid=(8, 8), top_n=2, ocr=False)
    md = result.to_report()
    assert "给模型的导读" in md
    assert "坐标均为归一化" in md
    assert "ASCII 明暗图字符表" in md


def test_summary_structure(test_image):
    """摘要模式：布局要点 + 区域要点表 + OCR 汇总。"""
    result = analyze(test_image, grid=(8, 8), top_n=2, ocr=False)
    md = result.to_summary()
    assert "（摘要）" in md
    assert "## 布局要点" in md
    assert "## 区域要点" in md
    assert "## OCR 汇总" in md
    assert "给模型的导读" in md
    # 区域要点表含主色
    assert "| 区域1 |" in md
    assert result.regions[0].color_hex  # 主色非空


def test_summary_is_smaller_than_report(test_image):
    """摘要 token 应明显小于完整报告。"""
    result = analyze(test_image, grid=(8, 8), top_n=2, ocr=False)
    full = result.to_report()
    summary = result.to_summary()
    assert len(summary) < len(full) * 0.6  # 至少砍掉 40%


def test_layout_points_detected(test_image):
    """布局要点应包含主色分布与高信息区域。"""
    result = analyze(test_image, grid=(8, 8), top_n=2, ocr=False)
    assert result.layout_points
    assert any("主色" in p for p in result.layout_points)


def test_pick_encoder_heuristic():
    assert _pick_encoder(color_variance=5000, edge_density=0.1) == "color_stats"
    assert _pick_encoder(color_variance=100, edge_density=0.1) == "ascii_art"


@pytest.fixture(scope="module")
def analyzer_ocr():
    """验证 analyze 的 OCR 链路可用（EasyOCR en）。"""
    try:
        from vision_reader.ocr.base import get

        engine = get("easyocr", languages=("en",))
        engine.recognize(np.zeros((40, 120, 3), np.uint8))
        return engine
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"EasyOCR 不可用: {exc}")


def test_analyze_with_ocr(analyzer_ocr, test_image):
    # 白底上叠加清晰大号英文，确保被自动选中并被 OCR 识别
    canvas = Image.fromarray(make_image(600, 400))
    d = ImageDraw.Draw(canvas)
    d.rectangle([30, 150, 570, 250], fill=(255, 255, 255))
    font = find_font(48)
    d.text((45, 160), "Vision Reader", fill=(0, 0, 0), font=font)
    arr = np.asarray(canvas)

    result = analyze(arr, grid=(8, 8), top_n=3, ocr=True, languages=("en",))
    joined = " ".join(r.ocr_text for r in result.regions).lower()
    assert "vision" in joined or "reader" in joined
