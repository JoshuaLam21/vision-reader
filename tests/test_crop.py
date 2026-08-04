"""crop 模块测试。"""

import pytest

from vision_reader import crop
from vision_reader.errors import CoordinateError, CropError
from vision_reader.synthetic import make_image


def test_crop_scaled_grayscale(test_image):
    r = crop.region(test_image, 0, 0, 0.5, 0.5, scale=2.0)
    # 裁剪 200x150，放大 2x → 400x300 灰度
    assert r.image.shape == (300, 400)
    assert r.image.ndim == 2
    assert r.pixel_box == (0, 0, 200, 150)
    assert r.norm_box == (0.0, 0.0, 0.5, 0.5)
    assert r.scale == 2.0
    assert r.grayscale is True


def test_crop_color_no_scale(test_image):
    r = crop.region(test_image, 0, 0, 0.5, 0.5, scale=1.0, grayscale=False)
    assert r.image.shape == (150, 200, 3)


def test_crop_content(test_image):
    # 左上区域应包含红色标题条：裁剪 0,0,0.2,0.2 后均值偏红
    r = crop.region(test_image, 0, 0, 0.2, 0.2, scale=1.0, grayscale=False)
    mean_r = r.image[..., 0].mean()
    mean_b = r.image[..., 2].mean()
    assert mean_r > mean_b + 20


def test_crop_invalid_scale(test_image):
    with pytest.raises(CropError):
        crop.region(test_image, 0, 0, 1, 1, scale=0)
    with pytest.raises(CropError):
        crop.region(test_image, 0, 0, 1, 1, scale=-1)
    with pytest.raises(CropError):
        crop.region(test_image, 0, 0, 1, 1, scale="x")


def test_crop_invalid_coords(test_image):
    with pytest.raises(CoordinateError):
        crop.region(test_image, -0.5, 0, 1, 1)
