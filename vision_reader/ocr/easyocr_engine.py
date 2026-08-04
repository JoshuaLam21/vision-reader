"""EasyOCR 引擎（默认）：中英文 ch_sim+en，纯 pip 依赖，无需外部服务。

注意：首次使用会下载检测/识别模型权重到用户目录（~/.EasyOCR）。
"""

from __future__ import annotations

import numpy as np

from ..errors import OCRError
from .base import OcrEngine, OcrResult, TextItem, register


@register
class EasyOcrEngine(OcrEngine):
    name = "easyocr"
    description = "EasyOCR 默认引擎：ch_sim+en，轻量、纯 Python 生态"

    def __init__(self, languages: tuple[str, ...] = ("ch_sim", "en"), gpu: bool = False):
        self._languages = list(languages)
        self._gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise OCRError("easyocr 未安装：pip install easyocr") from exc
            try:
                # 抑制 easyocr 的 "Using CPU" 与模型下载进度等噪声日志
                import logging

                logging.getLogger("easyocr").setLevel(logging.ERROR)
                self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
            except Exception as exc:  # noqa: BLE001
                raise OCRError(f"EasyOCR 初始化失败（可能需要网络下载模型）: {exc}") from exc
        return self._reader

    def recognize(self, image: np.ndarray) -> OcrResult:
        reader = self._get_reader()
        try:
            results = reader.readtext(image, detail=1, paragraph=False)
        except Exception as exc:  # noqa: BLE001
            raise OCRError(f"EasyOCR 识别失败: {exc}") from exc

        h, w = image.shape[:2]
        items = []
        for bbox, text, conf in results:
            pts = np.asarray(bbox, dtype=float)
            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            items.append(
                TextItem(
                    text=str(text),
                    confidence=float(conf),
                    bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
                )
            )
        return OcrResult(engine=self.name, items=items)
