"""encoders 模块测试：registry、三种编码器输出。"""

import pytest

from vision_reader.encoders.base import encode, get, names
from vision_reader.errors import EncoderError
from vision_reader.synthetic import make_image


def test_registry_contains_all():
    assert {"grayscale_grid", "ascii_art", "color_stats"} <= set(names())


def test_unknown_encoder():
    with pytest.raises(EncoderError):
        encode(make_image(64, 48), name="no_such_encoder")


def test_get_unknown():
    with pytest.raises(EncoderError):
        get("nope")


def test_grayscale_grid_output(test_image):
    out = encode(test_image, name="grayscale_grid", grid_width=16, levels=10)
    assert "灰度网格 16x" in out
    # 400x300 → grid_height = round(16 * 300/400) = 12
    digit_rows = [ln for ln in out.splitlines() if ln and ln[:3].strip().lstrip("-").isdigit()]
    assert len(digit_rows) == 12
    # 量化值都落在 0~9
    for row in digit_rows:
        cells = row.split()[1:]
        assert all(0 <= int(c) <= 9 for c in cells)


def test_ascii_art_output(test_image):
    out = encode(test_image, name="ascii_art", width=32)
    assert "ASCII 明暗图 32x" in out
    # height = round(32 * (300/400) * 0.5) = 12
    body = [ln for ln in out.splitlines() if ln and not ln.startswith("ASCII") and not ln.startswith("字符表")]
    assert len(body) >= 12


def test_color_stats_output(test_image):
    out = encode(test_image, name="color_stats", blocks=(2, 2))
    assert out.count("| r") == 4  # 2x2 = 4 行数据
    assert "边缘密度" in out


def test_encoders_deterministic(test_image):
    a = encode(test_image, name="ascii_art", width=16)
    b = encode(test_image, name="ascii_art", width=16)
    assert a == b


def test_grayscale_grid_levels_validation(test_image):
    with pytest.raises(ValueError):
        encode(test_image, name="grayscale_grid", levels=1)
    with pytest.raises(ValueError):
        encode(test_image, name="grayscale_grid", grid_width=0)
