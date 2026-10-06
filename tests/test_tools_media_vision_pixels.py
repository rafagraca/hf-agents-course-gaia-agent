"""``describe_image`` converts GIF, BMP and TIFF to PNG: the conversion is bounded by pixels, not by file size.

A small file can describe a huge picture (and decoding it costs memory), and a big uncompressed BMP can give a
small PNG, so the check before the conversion counts pixels and the check of the 4 MB limit looks at the PNG.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from gaia_agent.tools import _groq_vision
from gaia_agent.tools._groq_vision import MAX_PIXELS, image_data_url
from gaia_agent.tools._local_files import ToolError


def test_the_pixel_limit_is_a_realistic_number() -> None:
    assert MAX_PIXELS == 25_000_000


@pytest.mark.parametrize("name", ["big.gif", "big.bmp", "big.tiff"])
def test_an_image_with_too_many_pixels_is_refused_before_it_is_converted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    Image.new("RGB", (10, 10), "red").save(tmp_path / name)
    monkeypatch.setattr(_groq_vision, "MAX_PIXELS", 99)

    def must_not_convert(self: Image.Image, mode: str) -> Image.Image:
        raise AssertionError("the image was converted")

    monkeypatch.setattr(Image.Image, "convert", must_not_convert)

    with pytest.raises(ToolError, match="pixels"):
        image_data_url(tmp_path / name)


def test_an_image_exactly_at_the_pixel_limit_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Image.new("RGB", (10, 10), "red").save(tmp_path / "ok.bmp")
    monkeypatch.setattr(_groq_vision, "MAX_PIXELS", 100)

    assert image_data_url(tmp_path / "ok.bmp").startswith("data:image/png;base64,")


def test_a_big_uncompressed_bmp_that_gives_a_small_png_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "flat.bmp"
    Image.new("RGB", (1500, 1500), "white").save(path)  # about 6.75 MB on disk, 9 MB as base64
    assert path.stat().st_size * 4 // 3 > _groq_vision.MAX_BASE64_BYTES

    url = image_data_url(path)

    assert len(base64.b64decode(url.split(",", 1)[1])) < _groq_vision.MAX_BASE64_BYTES


def test_a_converted_image_whose_png_is_too_big_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image.new("RGB", (40, 40), "red").save(tmp_path / "x.bmp")
    monkeypatch.setattr(_groq_vision, "MAX_BASE64_BYTES", 100)

    with pytest.raises(ToolError, match="MB as base64"):
        image_data_url(tmp_path / "x.bmp")


def test_an_empty_converted_file_is_still_refused(tmp_path: Path) -> None:
    (tmp_path / "empty.gif").write_bytes(b"")

    with pytest.raises(ToolError, match="empty"):
        image_data_url(tmp_path / "empty.gif")
