"""coordinates 模块测试。"""

import pytest

from vision_reader import coordinates
from vision_reader.errors import CoordinateError


def test_basic_conversion():
    assert coordinates.clamp_region(0.25, 0.25, 0.75, 0.75, 400, 300) == (100, 75, 300, 225, False)


def test_full_image():
    assert coordinates.clamp_region(0, 0, 1, 1, 400, 300) == (0, 0, 400, 300, False)


def test_reversed_region_normalized():
    # x1 > x2 / y1 > y2 会被归一化为正向
    assert coordinates.clamp_region(0.75, 0.75, 0.25, 0.25, 400, 300) == (100, 75, 300, 225, False)


def test_out_of_range_rejected():
    with pytest.raises(CoordinateError):
        coordinates.clamp_region(-0.1, 0, 0.5, 0.5, 400, 300)
    with pytest.raises(CoordinateError):
        coordinates.clamp_region(0, 0, 1.1, 1, 400, 300)
    with pytest.raises(CoordinateError):
        coordinates.clamp_region(0, 2.0, 1, 1, 400, 300)


def test_non_numeric_rejected():
    with pytest.raises(CoordinateError):
        coordinates.clamp_region("a", 0, 1, 1, 400, 300)
    with pytest.raises(CoordinateError):
        coordinates.clamp_region(True, 0, 1, 1, 400, 300)


def test_degenerate_region_rejected():
    # 换算后宽或高为 0
    with pytest.raises(CoordinateError):
        coordinates.clamp_region(0.5, 0.5, 0.5, 0.75, 400, 300)


def test_tiny_region():
    # 极小但非零区域
    assert coordinates.clamp_region(0.49, 0.49, 0.51, 0.51, 400, 300) == (196, 147, 204, 153, False)


def test_norm_box_roundtrip():
    box = coordinates.norm_box(100, 75, 300, 225, 400, 300)
    assert box == (0.25, 0.25, 0.75, 0.75)
