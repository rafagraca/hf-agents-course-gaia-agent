"""Shared helpers of the ``describe_image`` tests: invented replies and attachments."""

from __future__ import annotations

import json
from pathlib import Path

import responses

FAKE_KEY = "fake-groq-key-for-tests"
PNG_BYTES = b"\x89PNG\r\n\x1a\n-invented-image-bytes-"


def completion(text: str) -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def make_image(attachments: Path, name: str = "picture.png", content: bytes = PNG_BYTES) -> str:
    (attachments / name).write_bytes(content)
    return name


def sent_json(http: responses.RequestsMock, index: int = 0) -> dict:
    return json.loads(http.calls[index].request.body)
