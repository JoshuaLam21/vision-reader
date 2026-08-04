"""pytest 共享 fixture：合成图。"""

from __future__ import annotations

import pytest

from vision_reader.synthetic import find_font, make_image


@pytest.fixture(scope="session")
def font28():
    return find_font(28)


@pytest.fixture(scope="session")
def test_image():
    return make_image(400, 300)
