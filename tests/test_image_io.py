"""image_io 模块测试：路径 / bytes / base64 / data URI / ndarray。"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from vision_reader import image_io
from vision_reader.errors import ImageLoadError
from vision_reader.synthetic import make_image


def _png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_load_path(tmp_path):
    arr = make_image(64, 48)
    p = tmp_path / "t.png"
    Image.fromarray(arr).save(p)
    loaded = image_io.load_image(str(p))
    assert loaded.shape == (48, 64, 3)
    np.testing.assert_array_equal(loaded, arr)


def test_load_bytes():
    b = _png_bytes(make_image(64, 48))
    loaded = image_io.load_image(b)
    assert loaded.shape == (48, 64, 3)


def test_load_raw_base64():
    b = _png_bytes(make_image(64, 48))
    loaded = image_io.load_image(base64.b64encode(b).decode())
    assert loaded.shape == (48, 64, 3)


def test_load_data_uri():
    b = _png_bytes(make_image(64, 48))
    uri = "data:image/png;base64," + base64.b64encode(b).decode()
    loaded = image_io.load_image(uri)
    assert loaded.shape == (48, 64, 3)


def test_load_rgba_ndarray():
    arr = make_image(64, 48)
    rgba = np.dstack([arr, np.full((48, 64), 255, np.uint8)])
    loaded = image_io.load_image(rgba)
    assert loaded.shape == (48, 64, 3)


def test_load_gray_ndarray():
    gray = np.full((30, 40), 128, np.uint8)
    loaded = image_io.load_image(gray)
    assert loaded.shape == (30, 40, 3)


def test_load_missing_file():
    with pytest.raises(ImageLoadError):
        image_io.load_image("no_such_file_xyz_123.png")


def test_load_bad_bytes():
    with pytest.raises(ImageLoadError):
        image_io.load_image(b"this is definitely not an image payload")


def test_load_bad_type():
    with pytest.raises(ImageLoadError):
        image_io.load_image(12345)  # type: ignore[arg-type]
