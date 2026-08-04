"""一键 demo：生成合成图 → vision analyze 自动分析（概览/选区域/编码/OCR）→ 输出完整 Markdown 报告。

演示"用户只给一张图，其余全部自动"的用法。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vision_reader.analyzer import analyze
from vision_reader.synthetic import make_image


def run(out_dir: str = "demo/output", grid: tuple[int, int] = (8, 8), save_png: bool = True) -> Path:
    """生成合成图并自动分析，返回报告路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 生成合成图
    img = make_image(800, 600, title="Vision Reader 演示")
    if save_png:
        Image.fromarray(img).save(out / "sample.png")
    print(f"[1] 合成图已生成: {out / 'sample.png'}（800x600）\n")

    # 2. 一键自动分析（用户只需给图，其余自动）
    print("[2] 一键分析中（自动概览 → 选区域 → 编码细看 → OCR）...\n")
    result = analyze(img, grid=grid, top_n=3, ocr=True, title="Vision Reader 演示图理解报告")

    # 3. 输出完整报告
    report_md = result.to_report()
    report_path = out / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"[3] 报告已生成: {report_path}")
    return report_path


if __name__ == "__main__":
    run()
