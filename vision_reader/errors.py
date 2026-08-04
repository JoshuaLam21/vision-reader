"""vision-reader 异常层次：所有错误都是 VisionReaderError 的子类，便于调用方统一捕获。"""


class VisionReaderError(Exception):
    """vision-reader 基础异常。"""


class ImageLoadError(VisionReaderError):
    """图片加载或解码失败。"""


class CoordinateError(VisionReaderError):
    """归一化坐标非法。"""


class CropError(VisionReaderError):
    """裁剪失败。"""


class EncoderError(VisionReaderError):
    """编码器相关错误（未知编码器、参数非法等）。"""


class OCRError(VisionReaderError):
    """OCR 相关错误。"""
