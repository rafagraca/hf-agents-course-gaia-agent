"""A rough token count for chat messages, shared by the token budget and the memory trimming.

Measured with cl100k: about 4.3-5.8 characters per token for English, ~3 for code, ~2.2 for JSON and URLs
(and ~1.5 for random alphanumerics). The estimate of 3.5 characters per token therefore over-counts prose and
under-counts structured data; the real count of each reply is what the budget books. On top of the text, gpt-oss
adds ~70 tokens of chat template to every request, which the estimate does not include.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["CHARS_PER_TOKEN", "estimate_tokens", "text_length"]

CHARS_PER_TOKEN = 3.5


def text_length(message: Any) -> int:
    """Count the text characters of a chat message: a ``ChatMessage`` or a ``{"content": ...}`` dict."""
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if isinstance(content, str):
        return len(content)
    if isinstance(content, Sequence):  # a list of parts such as {"type": "text", "text": "..."}
        return sum(
            len(part["text"]) for part in content if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return 0


def estimate_tokens(messages: Sequence[Any]) -> int:
    """Estimate the size of ``messages`` in tokens (characters / 3.5, rounded up). Only text is counted."""
    return math.ceil(sum(text_length(message) for message in messages) / CHARS_PER_TOKEN)
