"""命令行入口：vision overview / crop / ocr / report / demo。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import crop as crop_mod
from . import image_io, overview
from .analyzer import analyze
from .demo_runner import run as run_demo
from .encoders import encode, names as encoder_names
from .errors import VisionReaderError
from .ocr import recognize as ocr_recognize
from .ocr.base import names as ocr_engine_names
from .report import Observation, build


def _parse_region(text: str) -> tuple[float, float, float, float]:
    try:
        parts = [float(p) for p in text.split(",")]
    except ValueError as exc:
        raise SystemExit("--region 需要 4 个逗号分隔的数值: x1,y1,x2,y2") from exc
    if len(parts) != 4:
        raise SystemExit("--region 需要 4 个逗号分隔的数值: x1,y1,x2,y2")
    return tuple(parts)


def _parse_grid(text: str) -> tuple[int, int]:
    try:
        parts = [int(v) for v in text.lower().split("x")]
    except ValueError as exc:
        raise SystemExit("--grid 格式应为 NxM，如 8x8") from exc
    if len(parts) != 2 or any(p < 1 for p in parts):
        raise SystemExit("--grid 格式应为 NxM，如 8x8")
    return tuple(parts)  # type: ignore[return-value]


def cmd_overview(args) -> int:
    img = image_io.load_image(args.image)
    grid = _parse_grid(args.grid)
    chunks = overview.chunks(img, grid=grid)
    if args.as_json:
        obs = Observation(
            kind="overview",
            label=args.label,
            content=overview.overview_text(chunks, grid=grid),
            meta={"grid": args.grid},
        )
        print(json.dumps(asdict(obs), ensure_ascii=False, indent=2))
    else:
        print(overview.overview_text(chunks, grid=grid))
        print()
        print(overview.render_chunk_grid(chunks, grid=grid))
    return 0


def cmd_crop(args) -> int:
    img = image_io.load_image(args.image)
    x1, y1, x2, y2 = _parse_region(args.region)
    cropped = crop_mod.region(img, x1, y1, x2, y2, scale=args.scale, grayscale=not args.color)

    kwargs: dict = {"name": args.encode}
    if args.size:
        if args.encode == "color_stats":
            print("提示: --size 不适用于 color_stats，已忽略", file=sys.stderr)
        else:
            key = "grid_width" if args.encode == "grayscale_grid" else "width"
            kwargs[key] = args.size
    elif args.encode == "grayscale_grid":
        kwargs["grid_width"] = 32
    elif args.encode == "ascii_art":
        kwargs["width"] = 64
    encoded = encode(cropped.image, **kwargs)

    norm = tuple(round(v, 4) for v in cropped.norm_box)
    if args.as_json:
        obs = Observation(
            kind="crop",
            label=args.label,
            content=encoded,
            region=cropped.norm_box,
            meta={"encoder": args.encode, "scale": args.scale, "warnings": cropped.warnings},
        )
        print(json.dumps(asdict(obs), ensure_ascii=False, indent=2))
    else:
        print(f"裁剪区域（归一化）: {norm}，像素 bbox: {cropped.pixel_box}，放大 {args.scale}x")
        for w in cropped.warnings:
            print(f"⚠️ {w}")
        print()
        print(encoded)
    return 0


def cmd_ocr(args) -> int:
    img = image_io.load_image(args.image)
    if args.region:
        x1, y1, x2, y2 = _parse_region(args.region)
        cropped = crop_mod.region(img, x1, y1, x2, y2, scale=1.0, grayscale=False)
        target, norm = cropped.image, cropped.norm_box
    else:
        target, norm = img, (0.0, 0.0, 1.0, 1.0)

    languages = tuple(lang.strip() for lang in args.languages.split(",") if lang.strip())
    result = ocr_recognize(target, engine=args.engine, languages=languages)

    if args.as_json:
        obs = Observation(
            kind="ocr",
            label=args.label,
            content=result.text,
            region=norm,
            meta={"engine": args.engine, "count": len(result.items)},
        )
        print(json.dumps(asdict(obs), ensure_ascii=False, indent=2))
    else:
        print(f"OCR（引擎 {args.engine}，识别 {len(result.items)} 条，区域 {tuple(round(v, 4) for v in norm)}）")
        if result.is_empty():
            print("（未识别到文字）")
        else:
            for item in result.items:
                box = tuple(round(v, 3) for v in item.bbox)
                print(f"  {item.text!r}  conf={item.confidence:.2f}  bbox={box}")
    return 0


def cmd_report(args) -> int:
    observations: list[Observation] = []
    for path in args.observations:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            observations.extend(Observation(**o) for o in data)
        else:
            observations.append(Observation(**data))
    md = build(observations, title=args.title)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"报告已写入: {args.out}")
    print(md)
    return 0


def cmd_analyze(args) -> int:
    """一键分析：自动概览 + 选区域 + 编码 + OCR，输出完整报告。"""
    img = image_io.load_image(args.image)
    grid = _parse_grid(args.grid)
    result = analyze(
        img,
        grid=grid,
        top_n=args.top,
        ocr=not args.no_ocr,
        title=args.title,
    )
    report_md = result.to_summary() if args.summary else result.to_report()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"报告已写入: {args.out}")
    print(report_md)
    return 0


def cmd_demo(args) -> int:
    report_path = run_demo(out_dir=args.out, grid=_parse_grid(args.grid))
    print(f"\nDemo 完成，报告: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 cp950，中文输出会崩溃；统一切到 UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        prog="vision",
        description="为无图像能力的 LLM 提供渐进式视觉细节理解：像素+坐标 → 可读文本",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="【一键】自动分析：概览+选区域+编码+OCR，输出完整报告")
    p.add_argument("image", help="图片路径或 base64")
    p.add_argument("--grid", default="8x8", help="chunk 网格 NxM（默认 8x8）")
    p.add_argument("--top", type=int, default=3, help="自动细看的区域数（默认 3）")
    p.add_argument("--out", default="", help="报告输出路径（默认仅打印）")
    p.add_argument("--summary", action="store_true", help="输出 token 精简摘要（默认完整报告）")
    p.add_argument("--no-ocr", action="store_true", help="跳过 OCR")
    p.add_argument("--title", default="图片理解报告")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("overview", help="全图分块概览（模型的第一眼）")
    p.add_argument("image", help="图片路径或 base64")
    p.add_argument("--grid", default="8x8", help="chunk 网格 NxM（默认 8x8）")
    p.add_argument("--json", action="store_true", dest="as_json", help="以 Observation JSON 输出")
    p.add_argument("--label", default="整体概览")
    p.set_defaults(func=cmd_overview)

    p = sub.add_parser("crop", help="按归一化坐标裁剪并编码")
    p.add_argument("image")
    p.add_argument("--region", required=True, help="归一化坐标 x1,y1,x2,y2（0~1）")
    p.add_argument("--scale", type=float, default=2.0, help="放大倍数（默认 2.0）")
    p.add_argument("--encode", default="ascii_art", choices=encoder_names(), help="编码器")
    p.add_argument("--size", type=int, default=None, help="编码网格宽（grayscale_grid 的 grid_width / ascii_art 的 width）")
    p.add_argument("--color", action="store_true", help="保留彩色（默认裁剪后灰度化）")
    p.add_argument("--json", action="store_true", dest="as_json", help="以 Observation JSON 输出")
    p.add_argument("--label", default="区域")
    p.set_defaults(func=cmd_crop)

    p = sub.add_parser("ocr", help="OCR 识别区域文字")
    p.add_argument("image")
    p.add_argument("--region", default=None, help="归一化坐标 x1,y1,x2,y2；缺省为整图")
    p.add_argument("--engine", default="easyocr", choices=ocr_engine_names(), help="OCR 引擎")
    p.add_argument("--languages", default="ch_sim,en", help="语言列表，逗号分隔（默认 ch_sim,en）")
    p.add_argument("--json", action="store_true", dest="as_json", help="以 Observation JSON 输出")
    p.add_argument("--label", default="OCR")
    p.set_defaults(func=cmd_ocr)

    p = sub.add_parser("report", help="把多个 Observation JSON 汇总为 Markdown 报告")
    p.add_argument("observations", nargs="+", help="观察 JSON 文件（可多个）")
    p.add_argument("--out", required=True, help="输出报告路径")
    p.add_argument("--title", default="图片理解报告")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo", help="生成合成图并跑完整调用链路")
    p.add_argument("--out", default="demo/output", help="输出目录（默认 demo/output）")
    p.add_argument("--grid", default="8x8")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except VisionReaderError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
