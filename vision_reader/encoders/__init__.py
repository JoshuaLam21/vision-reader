"""可插拔编码器。导入本包即完成注册（子模块的 @register 副作用）。"""

from . import ascii_art  # noqa: F401  注册副作用
from . import color_stats  # noqa: F401
from . import grayscale_grid  # noqa: F401
from .base import Encoder, encode, get, names, register

__all__ = ["Encoder", "encode", "get", "names", "register"]
