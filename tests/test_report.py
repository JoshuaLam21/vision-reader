"""report 模块测试：多次观察汇总为 Markdown。"""

from vision_reader.report import Observation, build


def test_build_full():
    obs = [
        Observation(kind="overview", label="整体概览", content="| 块 | ... |\n| 0 | ... |"),
        Observation(
            kind="crop",
            label="区域A",
            content="灰度网格 8x8\n000 111",
            region=(0.0, 0.0, 0.5, 0.5),
            meta={"encoder": "grayscale_grid", "warnings": ["已 clamp"]},
        ),
        Observation(
            kind="ocr",
            label="区域A文字",
            content="Vision Reader",
            region=(0.0, 0.0, 0.5, 0.5),
            meta={"engine": "easyocr"},
        ),
    ]
    md = build(obs, title="测试报告")
    assert md.startswith("# 测试报告")
    assert "## 1. 整体概览" in md
    assert "## 2. 局部细节" in md
    assert "### 2.1 区域A" in md
    assert "编码器: `grayscale_grid`" in md
    assert "⚠️ 已 clamp" in md
    assert "## 3. OCR 结果" in md
    assert "引擎 `easyocr`" in md
    assert "## 4. 坐标索引" in md
    assert "| 区域A | `(0.000, 0.000, 0.500, 0.500)` | crop |" in md


def test_build_empty():
    md = build([], title="空报告")
    assert "（未采集整体概览）" in md
    assert "（未采集局部细节）" in md
    assert "（未做 OCR）" in md
    assert "（无带坐标的观察）" in md


def test_build_ocr_empty_text():
    obs = [
        Observation(
            kind="ocr",
            label="无文字区域",
            content="",
            region=(0.0, 0.0, 0.1, 0.1),
            meta={"engine": "easyocr"},
        )
    ]
    md = build(obs)
    assert "（未识别到文字）" in md
