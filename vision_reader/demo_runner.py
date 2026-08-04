"""完整调用链路演示：生成合成图 → overview → 选区域 → crop+encode → OCR → report。

模拟"无图像能力的模型"的渐进式观察流程：先看全图概览，再按归一化坐标
细看多个感兴趣区域（不同编码器），对文字区域做 OCR，最后汇总 Markdown 报告。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vision_reader import crop as crop_mod
from vision_reader import overview
from vision_reader.encoders import encode
from vision_reader.ocr import recognize
from vision_reader.report import Observation, build
from vision_reader.synthetic import make_image


def _select_regions() -> list[dict]:
    """模拟"模型根据 overview 选出的关注区域"（800x600 合成图的已知布局）。"""
    return [
        {"label": "红色标题条", "region": (0.02, 0.03, 0.5, 0.12), "encoder": "ascii_art", "size": 48},
        {"label": "蓝色圆", "region": (0.7, 0.05, 0.95, 0.4), "encoder": "color_stats", "size": None},
        {"label": "正文文字区", "region": (0.02, 0.42, 0.6, 0.58), "encoder": "grayscale_grid", "size": 32},
        {"label": "底部绿色条带", "region": (0.02, 0.8, 0.95, 0.93), "encoder": "ascii_art", "size": 40},
    ]


def run(out_dir: str = "demo/output", grid: tuple[int, int] = (8, 8), save_png: bool = True) -> Path:
    """执行完整链路，返回生成的报告路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 生成合成图
    img = make_image(800, 600, title="Vision Reader 演示")
    if save_png:
        Image.fromarray(img).save(out / "sample.png")
    print(f"[1] 合成图已生成: {out / 'sample.png'}（800x600）\n")

    observations: list[Observation] = []

    # 2. 全图概览（模型的第一眼）
    chunks = overview.chunks(img, grid=grid)
    obs_overview = Observation(
        kind="overview",
        label="整体概览",
        content=overview.overview_text(chunks, grid=grid),
        meta={"grid": f"{grid[0]}x{grid[1]}"},
    )
    observations.append(obs_overview)
    print("[2] 全图概览（8x8 chunk）：")
    print(obs_overview.content, "\n")

    # 3. 逐个区域细看（模拟模型迭代调度 crop + encode）
    for i, sel in enumerate(_select_regions(), start=1):
        region = sel["region"]
        cropped = crop_mod.region(img, *region, scale=2.0)
        kwargs = {"name": sel["encoder"]}
        if sel["size"]:
            # grayscale_grid 用 grid_width，ascii_art 用 width
            key = "grid_width" if sel["encoder"] == "grayscale_grid" else "width"
            kwargs[key] = sel["size"]
        encoded = encode(cropped.image, **kwargs)

        obs = Observation(
            kind="crop",
            label=sel["label"],
            content=encoded,
            region=cropped.norm_box,
            meta={"encoder": sel["encoder"], "warnings": cropped.warnings},
        )
        observations.append(obs)
        print(f"[3.{i}] {sel['label']}（编码器 {sel['encoder']}，区域 {tuple(round(v, 3) for v in cropped.norm_box)}）")
        print(encoded.splitlines()[0], "...")
        print()

    # 4. 对文字区域做 OCR
    text_region = (0.02, 0.42, 0.6, 0.58)
    cropped = crop_mod.region(img, *text_region, scale=2.0, grayscale=False)
    result = recognize(cropped.image, engine="easyocr", languages=("ch_sim", "en"))
    obs_ocr = Observation(
        kind="ocr",
        label="正文文字区 OCR",
        content=result.text if not result.is_empty() else "",
        region=cropped.norm_box,
        meta={"engine": "easyocr", "count": len(result.items)},
    )
    observations.append(obs_ocr)
    print(f"[4] OCR（easyocr ch_sim+en，识别 {len(result.items)} 条）：")
    print(result.text if result.text else "（未识别到文字）", "\n")

    # 5. 汇总报告
    report_md = build(observations, title="Vision Reader 演示图理解报告")
    report_path = out / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[5] 报告已生成: {report_path}")

    # 6. 同时把观察 JSON 落盘，便于 vision report 子命令复用
    obs_path = out / "observations.json"
    obs_path.write_text(_dump_observations(observations), encoding="utf-8")
    print(f"    观察记录: {obs_path}")
    return report_path


def _dump_observations(observations: list[Observation]) -> str:
    import json

    from dataclasses import asdict

    return json.dumps([asdict(o) for o in observations], ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()
