"""Image questions answered by Groq's vision model (``qwen/qwen3.8-27b``) through chat completions.

One image goes in per call, as a base64 ``data:`` URL inside an OpenAI-style user message
(https://console.groq.com/docs/vision). Groq refuses requests whose image is over 4 MB in base64 and
counts every image as about 2048 input tokens against the model's per-minute budget, so a busy minute
answers HTTP 429: that is waited for once, up to a minute, and then reported. Every failure becomes a short
:class:`ToolError` that never contains the API key.
"""

from __future__ import annotations

import base64
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

from gaia_agent.tools._api_errors import error_message
from gaia_agent.tools._gemini import question_prompt
from gaia_agent.tools._local_files import ToolError

CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = "qwen/qwen3.8-27b"  # no reasoning_effort: on this model it would switch the reasoning on
MAX_TOKENS = 1500
MEGABYTE = 1024 * 1024
MAX_BASE64_BYTES = 4 * MEGABYTE
MAX_PIXELS = 25_000_000  # a GIF, BMP or TIFF with more is not decoded: it would take hundreds of MB as RGBA
REQUEST_TIMEOUT_S = (10, 120)  # connect, read
MAX_RETRY_WAIT_S = 60.0
DEFAULT_RETRY_WAIT_S = 15.0  # when a 429 comes without a usable retry-after

IMAGE_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
CONVERTED_SUFFIXES = frozenset({".gif", ".bmp", ".tif", ".tiff"})  # re-encoded as PNG before sending

_SECONDS = re.compile(r"\d+(\.\d+)?")


def _pause(seconds: float) -> None:
    time.sleep(seconds)


# --------------------------------------------------------------------------- request


def _png_bytes(path: Path) -> bytes:
    """Re-encode a GIF, BMP or TIFF as PNG (the first frame of an animation); refuse one with too many pixels."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width * height > MAX_PIXELS:
                raise ToolError(f"the image has {width * height:,} pixels; the limit is {MAX_PIXELS:,} pixels")
            image.seek(0)
            buffer = io.BytesIO()
            image.convert("RGBA").save(buffer, format="PNG")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:  # a damaged file is an OSError
        raise ToolError(f"could not convert {path.name} to PNG ({type(exc).__name__})") from exc
    return buffer.getvalue()


def _base64_size(byte_count: int) -> int:
    return -(-byte_count // 3) * 4


def _check_size(byte_count: int, *, limit: bool = True) -> None:
    if byte_count == 0:
        raise ToolError("the image file is empty")
    if limit and _base64_size(byte_count) > MAX_BASE64_BYTES:
        raise ToolError(
            f"the image is {_base64_size(byte_count) / MEGABYTE:.1f} MB as base64; "
            f"the limit is {MAX_BASE64_BYTES // MEGABYTE} MB"
        )


def image_data_url(path: Path) -> str:
    """The ``data:<mime>;base64,...`` URL of the image at ``path`` (GIF, BMP and TIFF are sent as PNG)."""
    suffix = path.suffix.lower()
    converted = suffix in CONVERTED_SUFFIXES
    if suffix not in IMAGE_MIME_TYPES and not converted:
        raise ToolError(f"{path.name} is not a supported image (use png, jpg, webp, gif, bmp or tiff)")
    try:
        # A converted file is judged by its pixels before and by its PNG after: its own size says little.
        _check_size(path.stat().st_size, limit=not converted)
        data = _png_bytes(path) if converted else path.read_bytes()
    except OSError as exc:
        raise ToolError(f"could not read the image file ({type(exc).__name__})") from exc
    _check_size(len(data))
    mime_type = "image/png" if converted else IMAGE_MIME_TYPES[suffix]
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def build_payload(data_url: str, question: object) -> dict:
    """The chat-completions body: one user message with the question, then the image."""
    content = [
        {"type": "text", "text": question_prompt("image", question)},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    return {"model": VISION_MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": MAX_TOKENS}


# --------------------------------------------------------------------------- the call


def _post(api_key: str, payload: dict) -> requests.Response:
    try:
        return requests.post(
            CHAT_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=REQUEST_TIMEOUT_S
        )
    except requests.Timeout as exc:
        raise ToolError("the Groq request timed out") from exc
    except requests.RequestException as exc:
        raise ToolError("could not reach the Groq API") from exc


def _retry_after(response: requests.Response) -> float | None:
    """The ``retry-after`` seconds of a reply, or None when absent or not a plain number."""
    text = response.headers.get("retry-after", "").strip()
    return float(text) if _SECONDS.fullmatch(text) else None


def _send(api_key: str, payload: dict) -> requests.Response:
    """Post once; on a 429 wait for the advertised time (one minute at most) and post once more."""
    response = _post(api_key, payload)
    if response.status_code != 429:
        return response
    wait = _retry_after(response)
    wait = DEFAULT_RETRY_WAIT_S if wait is None else wait
    if wait <= MAX_RETRY_WAIT_S:
        _pause(wait)
        response = _post(api_key, payload)
    return response


def _failure(response: requests.Response, api_key: str) -> ToolError:
    status = response.status_code
    if status in (401, 403):
        return ToolError(f"Groq rejected the API key (HTTP {status}); check GROQ_API_KEY")
    if status == 413:
        return ToolError("the request is too large for Groq (HTTP 413); try a smaller image")
    if status == 429:
        wait = _retry_after(response)
        hint = "" if wait is None else f", retry after {wait:g} s"
        return ToolError(f"Groq rate limit or daily quota reached (HTTP 429{hint}); try again later")
    return ToolError(f"Groq API error (HTTP {status}): {error_message(response, api_key)}")


def _answer_text(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError as exc:
        raise ToolError("Groq returned an unreadable reply") from exc
    if not isinstance(data, dict):
        raise ToolError("Groq returned an unreadable reply")
    choices = data.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    text = content.strip() if isinstance(content, str) else ""
    if not text:
        raise ToolError("Groq returned no text for this image")
    return text


def ask_groq_about_image(api_key: str, path: Path, question: object) -> str:
    """Send the image at ``path`` and ``question`` to the Groq vision model and return its answer."""
    payload = build_payload(image_data_url(path), question)
    response = _send(api_key, payload)
    if response.status_code != 200:
        raise _failure(response, api_key)
    return _answer_text(response)
