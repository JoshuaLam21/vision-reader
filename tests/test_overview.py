"""overview 模块测试。"""

from vision_reader import overview


def test_chunks_count_and_ranges(test_image):
    cs = overview.chunks(test_image, grid=(8, 8))
    assert len(cs) == 64
    for c in cs:
        assert 0 <= c.x1 < c.x2 <= 1
        assert 0 <= c.y1 < c.y2 <= 1
        assert 0.0 <= c.edge_density <= 1.0
        assert 0 <= c.brightness <= 255
        assert c.color_hex.startswith("#") and len(c.color_hex) == 7
        assert len(c.color_rgb) == 3


def test_overview_text(test_image):
    cs = overview.chunks(test_image, grid=(4, 4))
    text = overview.overview_text(cs, grid=(4, 4))
    assert "全图概览（4x4 chunk 网格" in text
    assert text.count("\n| ") >= 4


def test_overview_json(test_image):
    cs = overview.chunks(test_image, grid=(2, 2))
    data = overview.overview_json(cs)
    assert len(data["chunks"]) == 4
    assert set(data["chunks"][0]) >= {"x1", "y1", "x2", "y2", "color_hex", "brightness", "edge_density"}


def test_render_chunk_grid(test_image):
    cs = overview.chunks(test_image, grid=(4, 4))
    out = overview.render_chunk_grid(cs, grid=(4, 4))
    lines = out.splitlines()
    # 头部 2 行 + 4 行网格
    assert len(lines) == 6
    assert all(len(ln.split()) == 4 for ln in lines[2:])


def test_chunks_edge_density_detects_text(test_image):
    # 中间含文字区域边缘密度应高于纯色角落
    cs = overview.chunks(test_image, grid=(4, 4))
    text_chunk = next(c for c in cs if c.y1 >= 0.25 and c.y1 < 0.5 and c.x1 >= 0.0 and c.x1 < 0.25)
    corner_chunk = next(c for c in cs if c.y1 >= 0.5 and c.y1 < 0.75 and c.x1 >= 0.75)
    assert text_chunk.edge_density > corner_chunk.edge_density
