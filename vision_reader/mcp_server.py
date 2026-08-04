"""MCP server：把 vision-reader 封装为 MCP 工具（FastMCP + stdio transport）。

运行：``uv run python -m vision_reader.mcp_server``

设计：
- ``image_id`` 缓存：模型先 ``vision_load_image`` 注册图片，后续工具只传 id，避免重复传 base64。
- OCR 引擎实例缓存：模型只加载一次，复用 reader。
- 所有坐标为归一化 (0~1)；看哪些区域由模型自行决定（渐进式观察）。
"""

from __future__ import annotations

import numpy as np
from mcp.server.fastmcp import FastMCP

from . import crop as crop_mod
from . import image_io, overview
from .encoders import encode, names as encoder_names
from .errors import VisionReaderError
from .ocr.base import get as ocr_get_engine, names as ocr_engine_names

# ---- 图像缓存：image_id → RGB ndarray ----
_IMAGES: dict[str, np.ndarray] = {}
_IMAGE_COUNTER = 0

# ---- OCR 引擎实例缓存：(engine, languages) → OcrEngine ----
_ENGINES: dict[tuple[str, tuple[str, ...]], object] = {}


def _resolve_image(image: str) -> np.ndarray:
    """image 参数：优先按 image_id 查缓存，否则按路径/base64 加载。"""
    if image in _IMAGES:
        return _IMAGES[image]
    return image_io.load_image(image)


def _get_engine(engine: str, languages: tuple[str, ...]):
    key = (engine, languages)
    if key not in _ENGINES:
        _ENGINES[key] = ocr_get_engine(engine, languages=languages)
    return _ENGINES[key]


def _parse_region(text: str) -> tuple[float, float, float, float]:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("region 需要 4 个逗号分隔的归一化坐标 x1,y1,x2,y2")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _do_overview(img: np.ndarray, grid: tuple[int, int]) -> str:
    chunks = overview.chunks(img, grid=grid)
    return overview.overview_text(chunks, grid=grid) + "\n\n" + overview.render_chunk_grid(chunks, grid=grid)


def _do_crop(
    img: np.ndarray,
    region: tuple[float, float, float, float],
    scale: float,
    encoder: str,
    size: int,
    color: bool,
) -> str:
    cr = crop_mod.region(img, *region, scale=scale, grayscale=not color)
    kwargs: dict = {"name": encoder}
    if size > 0:
        key = "grid_width" if encoder == "grayscale_grid" else "width"
        kwargs[key] = size
    elif encoder == "grayscale_grid":
        kwargs["grid_width"] = 32
    elif encoder == "ascii_art":
        kwargs["width"] = 64
    text = encode(cr.image, **kwargs)
    box = tuple(round(v, 4) for v in cr.norm_box)
    prefix = f"区域（归一化坐标）: {box}，像素 bbox: {cr.pixel_box}\n"
    for w in cr.warnings:
        prefix += f"警告: {w}\n"
    return prefix + text


def _do_ocr(
    img: np.ndarray,
    region: tuple[float, float, float, float] | None,
    engine: str,
    languages: tuple[str, ...],
) -> dict:
    target, norm = img, (0.0, 0.0, 1.0, 1.0)
    if region is not None:
        cr = crop_mod.region(img, *region, scale=1.0, grayscale=False)
        target, norm = cr.image, cr.norm_box
    result = _get_engine(engine, languages).recognize(target)
    return {
        "text": result.text,
        "count": len(result.items),
        "region": norm,
        "items": [
            {
                "text": item.text,
                "confidence": round(item.confidence, 4),
                "bbox": tuple(round(v, 4) for v in item.bbox),
            }
            for item in result.items
        ],
    }


mcp = FastMCP(
    "vision-reader",
    instructions=(
        "给无图像能力的 LLM 的看图工具。推荐流程："
        "1) vision_load_image 注册图片，拿到 image_id；"
        "2) vision_overview 看全图 chunk 概览（每块含归一化坐标与统计），自行决定值得细看的区域；"
        "3) vision_crop 按归一化坐标 (0~1) 裁剪并编码成文本细看（编码器可选 ascii_art/grayscale_grid/color_stats）；"
        "4) 需要读文字时 vision_ocr。可反复 3~4 渐进聚焦，最后自行汇总语义。"
    ),
)


@mcp.tool()
def vision_load_image(image_path: str = "", image_base64: str = "") -> dict:
    """注册图片并返回 image_id（后续工具用 image 参数引用，避免重复传图）。

    参数二选一：image_path（本地路径）或 image_base64（data URI / 裸 base64）。
    """
    global _IMAGE_COUNTER
    try:
        source = image_base64 or image_path
        if not source:
            raise ValueError("必须提供 image_path 或 image_base64 之一")
        img = image_io.load_image(source)
        _IMAGE_COUNTER += 1
        image_id = f"img{_IMAGE_COUNTER}"
        _IMAGES[image_id] = img
        h, w = img.shape[:2]
        return {"image_id": image_id, "width": int(w), "height": int(h)}
    except VisionReaderError as exc:
        return {"error": str(exc)}


@mcp.tool()
def vision_overview(image: str, grid: str = "8x8") -> str:
    """看全图 chunk 概览：每块输出归一化坐标/主色/亮度/边缘密度/颜色方差，据此决定细看哪些区域。

    image 为 image_id（推荐）或图片路径/base64；grid 如 8x8。
    """
    try:
        parts = [int(v) for v in grid.lower().split("x")]
        if len(parts) != 2 or any(p < 1 for p in parts):
            raise ValueError("grid 格式应为 NxM，如 8x8")
        return _do_overview(_resolve_image(image), tuple(parts))  # type: ignore[arg-type]
    except (VisionReaderError, ValueError) as exc:
        return f"错误: {exc}"


@mcp.tool()
def vision_crop(
    image: str,
    region: str,
    scale: float = 2.0,
    encoder: str = "ascii_art",
    size: int = 0,
    color: bool = False,
) -> str:
    """按归一化坐标裁剪区域并编码成模型可读文本。

    region 格式 'x1,y1,x2,y2'（0~1）；scale 放大倍数（默认 2.0）；
    encoder: ascii_art（默认）/ grayscale_grid / color_stats；
    size: 编码网格宽（grayscale_grid 的 grid_width / ascii_art 的 width，0=默认）；
    color: True 时保留彩色（默认灰度化）。
    """
    try:
        return _do_crop(_resolve_image(image), _parse_region(region), scale, encoder, size, color)
    except (VisionReaderError, ValueError) as exc:
        return f"错误: {exc}"


@mcp.tool()
def vision_ocr(
    image: str,
    region: str = "",
    engine: str = "easyocr",
    languages: str = "ch_sim,en",
) -> dict:
    """识别区域文字。region 缺省识别整图；返回 text 与逐条 items（文本/置信度/归一化 bbox）。"""
    try:
        img = _resolve_image(image)
        region_parsed = _parse_region(region) if region else None
        langs = tuple(l.strip() for l in languages.split(",") if l.strip())
        return _do_ocr(img, region_parsed, engine, langs)
    except (VisionReaderError, ValueError) as exc:
        return {"error": str(exc)}


@mcp.tool()
def vision_list_encoders() -> str:
    """列出可用的像素编码器。"""
    return "可用编码器: " + ", ".join(encoder_names())


@mcp.tool()
def vision_list_ocr_engines() -> str:
    """列出可用的 OCR 引擎。"""
    return "可用 OCR 引擎: " + ", ".join(ocr_engine_names())


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
