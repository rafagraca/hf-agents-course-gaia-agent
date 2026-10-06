"""Tests for ``describe_image``: Groq vision (``qwen/qwen3.8-27b``), HTTP faked with ``responses``, no network."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
import responses
from PIL import Image
from vision_fakes import FAKE_KEY, PNG_BYTES, completion, make_image, sent_json

from gaia_agent.config import Settings
from gaia_agent.tools import _groq_vision
from gaia_agent.tools._groq_vision import (
    CHAT_URL,
    MAX_BASE64_BYTES,
    VISION_MODEL,
    build_payload,
    image_data_url,
)
from gaia_agent.tools._local_files import ToolError
from gaia_agent.tools.media import DescribeImageTool

FAKE_KEY = "fake-groq-key-for-tests"
PNG_BYTES = b"\x89PNG\r\n\x1a\n-invented-image-bytes-"
REAL_PAUSE = _groq_vision._pause  # captured before the autouse fixture replaces it


@pytest.fixture
def http():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocked:
        yield mocked


@pytest.fixture
def attachments(settings: Settings) -> Path:
    return settings.files_dir


@pytest.fixture(autouse=True)
def groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)


@pytest.fixture(autouse=True)
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the waits instead of sleeping."""
    waited: list[float] = []
    monkeypatch.setattr(_groq_vision, "_pause", waited.append)
    return waited


@pytest.fixture
def describe(settings: Settings) -> DescribeImageTool:
    return DescribeImageTool(settings)


# --------------------------------------------------------------------------- the payload (pure)


def test_build_payload_has_one_user_message_with_text_then_image() -> None:
    payload = build_payload("data:image/png;base64,AAAA", "What shape?")

    assert payload["model"] == "qwen/qwen3.8-27b" == VISION_MODEL
    assert payload["max_tokens"] == 1500
    assert "reasoning_effort" not in payload
    (message,) = payload["messages"]
    assert message["role"] == "user"
    text_part, image_part = message["content"]
    assert text_part["type"] == "text"
    assert "What shape?" in text_part["text"]
    assert "instead of guessing" in text_part["text"]
    assert image_part == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_build_payload_rejects_an_empty_question() -> None:
    with pytest.raises(ToolError, match="question"):
        build_payload("data:image/png;base64,AAAA", "   ")


def test_image_data_url_encodes_the_file_with_its_mime_type(tmp_path: Path) -> None:
    path = tmp_path / "x.JPG"
    path.write_bytes(b"jpeg-bytes")

    assert image_data_url(path) == "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode()


# --------------------------------------------------------------------------- describe_image


def test_describe_image_posts_the_image_to_groq_and_returns_the_answer(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, json=completion("  A red square  "))

    result = describe(name, "What shape is shown?")

    assert result == "A red square"
    request = http.calls[0].request
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in request.url
    body = sent_json(http)
    assert body["model"] == "qwen/qwen3.8-27b"
    url = body["messages"][0]["content"][1]["image_url"]["url"]
    assert url == "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


@pytest.mark.parametrize(
    ("name", "mime"),
    [("a.jpg", "image/jpeg"), ("a.JPEG", "image/jpeg"), ("a.webp", "image/webp"), ("a.png", "image/png")],
)
def test_describe_image_sets_the_mime_type_from_the_extension(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, name: str, mime: str
) -> None:
    make_image(attachments, name)
    http.add(responses.POST, CHAT_URL, json=completion("ok"))

    describe(name, "what?")

    assert sent_json(http)["messages"][0]["content"][1]["image_url"]["url"].startswith(f"data:{mime};base64,")


@pytest.mark.parametrize("name", ["anim.gif", "scan.bmp", "photo.tiff", "photo.TIF"])
def test_describe_image_converts_other_formats_to_png(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, name: str
) -> None:
    Image.new("RGB", (6, 4), "red").save(attachments / name)
    http.add(responses.POST, CHAT_URL, json=completion("converted"))

    assert describe(name, "what?") == "converted"

    url = sent_json(http)["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert (sent.format, sent.size) == ("PNG", (6, 4))


def test_describe_image_sends_the_first_frame_of_an_animated_gif(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    frames = [Image.new("RGB", (5, 5), colour) for colour in ("red", "blue")]
    frames[0].save(attachments / "two.gif", save_all=True, append_images=frames[1:])
    http.add(responses.POST, CHAT_URL, json=completion("first frame"))

    describe("two.gif", "what?")

    url = sent_json(http)["messages"][0]["content"][1]["image_url"]["url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    assert sent.getpixel((0, 0))[0] > 200  # red, not blue


def test_describe_image_reports_a_damaged_image_it_cannot_convert(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    make_image(attachments, "broken.gif", b"this is not an image at all")

    result = describe("broken.gif", "what?")

    assert result.startswith("Error:")
    assert "could not convert" in result
    assert not http.calls


@pytest.mark.parametrize("name", ["notes.txt", "noextension", "drawing.svg", "layers.psd"])
def test_describe_image_rejects_unsupported_files(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, name: str
) -> None:
    make_image(attachments, name, b"some bytes")

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "not a supported image" in result
    assert not http.calls


def test_describe_image_rejects_an_empty_file(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_image(attachments, content=b"")

    assert "empty" in describe(name, "what?")
    assert not http.calls


def test_describe_image_reports_a_file_that_cannot_be_read(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_image(attachments)

    def locked(self: Path) -> bytes:
        raise PermissionError("locked by another process")

    monkeypatch.setattr(Path, "read_bytes", locked)

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "could not read the image" in result
    assert not http.calls


def test_describe_image_refuses_an_image_over_4_mb_in_base64(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    # 3 MB of bytes is 4 MB of base64: just over the limit once the last byte is added.
    size = MAX_BASE64_BYTES // 4 * 3 + 1
    (attachments / "big.png").write_bytes(b"\x00" * size)

    result = describe("big.png", "what?")

    assert result.startswith("Error:")
    assert "4 MB" in result
    assert not http.calls


def test_describe_image_accepts_an_image_just_under_the_limit(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    (attachments / "fits.png").write_bytes(b"\x00" * (MAX_BASE64_BYTES // 4 * 3))
    http.add(responses.POST, CHAT_URL, json=completion("fits"))

    assert describe("fits.png", "what?") == "fits"


def test_describe_image_refuses_a_file_outside_the_attachments_folder(
    describe: DescribeImageTool, tmp_path: Path, http: responses.RequestsMock
) -> None:
    outside = tmp_path / "private.png"
    outside.write_bytes(PNG_BYTES)

    result = describe(str(outside), "what?")

    assert result.startswith("Error:")
    assert "outside" in result
    assert not http.calls


def test_describe_image_reports_a_missing_file(describe: DescribeImageTool) -> None:
    assert describe("absent.png", "what?").startswith("Error: file not found")


def test_describe_image_needs_a_question(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_image(attachments)

    result = describe(name, "   ")

    assert result.startswith("Error:")
    assert "question" in result
    assert not http.calls


def test_describe_image_without_a_key_says_so_and_sends_nothing(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_image(attachments)
    monkeypatch.delenv("GROQ_API_KEY")

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "GROQ_API_KEY" in result
    assert not http.calls


def test_describe_image_works_without_explicit_settings(attachments: Path, http: responses.RequestsMock) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, json=completion("default settings"))

    assert DescribeImageTool()(name, "what?") == "default settings"


def test_describe_image_truncates_a_very_long_answer(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, json=completion("word " * 3000))

    result = describe(name, "what?")

    assert "[truncated:" in result
    assert len(result) < 4200
