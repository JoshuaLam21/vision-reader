"""图片加载：路径 / data-URI / 裸 base64 / bytes / ndarray → 统一 RGB uint8 数组。"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import ImageLoadError

ImageSource = Union[str, bytes, np.ndarray]

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def load_image(source: ImageSource) -> np.ndarray:
    """把输入加载为 RGB uint8 numpy 数组 (H, W, 3)。

    支持的输入：
    - 本地文件路径（str）
    - data URI，如 ``data:image/png;base64,....``
    - 裸 base64 字符串（长度 >= 32 且只含 base64 字符集时才会被当作 base64 解析）
    - bytes（原始图片字节）
    - 已有 ndarray（RGB / RGBA / 灰度，自动归一化到 RGB uint8）
    """
    if isinstance(source, np.ndarray):
        return _normalize_ndarray(source)

    if isinstance(source, bytes):
        raw = source
    elif isinstance(source, str):
        raw = _decode_str(source)
    else:
        raise ImageLoadError(f"不支持的图片输入类型: {type(source).__name__}")

    try:
        with Image.open(io.BytesIO(raw)) as img:
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"图片解码失败: {exc}") from exc


def _normalize_ndarray(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:  # 灰度
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA → RGB
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageLoadError(f"不支持的 ndarray 形状: {arr.shape}")
    return np.ascontiguousarray(arr)


def _decode_str(source: str) -> bytes:
    if source.startswith("data:"):
        comma = source.find(",")
        if comma == -1:
            raise ImageLoadError("非法的 data URI（缺少逗号分隔的 payload）")
        source = source[comma + 1 :]
        try:
            return base64.b64decode(source, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise ImageLoadError(f"data URI 的 base64 解码失败: {exc}") from exc

    # 优先按文件路径处理（路径中可能恰巧只含 base64 字符集）
    if os.path.exists(source):
        with open(source, "rb") as f:
            return f.read()

    if len(source.strip()) >= 32 and _BASE64_RE.fullmatch(source.strip()):
        try:
            return base64.b64decode(source, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise ImageLoadError(f"base64 解码失败: {exc}") from exc

    raise ImageLoadError(f"输入既不是存在的文件路径，也不是合法 base64 数据: {source[:64]}...")
