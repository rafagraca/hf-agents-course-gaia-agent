"""One web-search request to Groq's browser-search model (``openai/gpt-oss-20b`` with the ``browser_search`` tool).

The model runs several searches on its own and answers in one reply (see
https://console.groq.com/docs/browser-search). This module only builds the request, sends it and turns the reply
into a short text with its sources; counting, waiting and logging belong to :mod:`gaia_agent.tools.deep_search`.
Failures become a :class:`SearchFailure` with a short reason that never contains the API key.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests

from gaia_agent.provider_errors import FailureKind, classify_failure, retry_after_seconds
from gaia_agent.tools._local_files import redact
from gaia_agent.tools._sources import blocked_source, looks_like_answer_key

__all__ = [
    "BLOCKED_REASON",
    "MAX_COMPLETION_TOKENS",
    "MAX_REPLY_CHARS",
    "SEARCH_MODEL",
    "SearchFailure",
    "SearchReply",
    "build_payload",
    "format_reply",
    "search",
]

SEARCH_URL = "https://api.groq.com/openai/v1/chat/completions"
SEARCH_MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 2048
TEMPERATURE = 0.2
REQUEST_TIMEOUT_S = 120
MAX_REPLY_CHARS = 1500
MAX_SOURCES = 5
MAX_URL_CHARS = 200  # a longer link is skipped: cutting it would give a link that leads nowhere
REASON_LIMIT = 200
BLOCKED_REASON = "the search used a blocked source"
SYSTEM_PROMPT = (
    "You answer factual questions using web search. Search as many times as you need, then reply with a concise "
    "factual answer (a few sentences at most) and list the source URLs you relied on. Never use a benchmark's "
    "dataset, nor Spaces, repositories or forum threads that publish answers to exercises or exams. If you cannot "
    "find the answer, say that you don't know; never guess."
)

_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,;:!?"


class SearchFailure(Exception):
    """The search did not give an answer. The message is a short reason that is safe to show an agent."""

    def __init__(self, reason: str, *, kind: FailureKind = FailureKind.OTHER, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.kind = kind
        self.retry_after = retry_after


@dataclass(frozen=True)
class SearchReply:
    """What a successful search returned: the answer, its source links and the tokens it cost (None if unreported).

    ``blocked`` is True when the reply cannot be given to the agent: some page the search touched is a place
    that publishes benchmark answers, or what came back looks like an answer key (see ``_sources``).
    """

    text: str
    sources: tuple[str, ...]
    input_tokens: int | None
    output_tokens: int | None
    seconds: float
    blocked: bool = False


class _GroqHttpError(Exception):
    """An HTTP error reply in the shape :mod:`gaia_agent.provider_errors` reads (status, headers, text)."""

    def __init__(self, response: requests.Response, message: str) -> None:
        super().__init__(message)
        self.status_code = response.status_code
        self.response = response


def build_payload(query: str) -> dict[str, Any]:
    """The JSON body of the request for ``query``."""
    return {
        "model": SEARCH_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
        "tools": [{"type": "browser_search"}],
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperature": TEMPERATURE,
    }


def _error_text(response: requests.Response, api_key: str) -> str:
    """The error text of a reply on one line, whole (a retry hint sits at its end) and without the key."""
    try:
        text = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        text = response.content.decode("utf-8", errors="replace")
    return redact(" ".join(str(text).split()), api_key)


def _failure_for(response: requests.Response, api_key: str) -> SearchFailure:
    text = _error_text(response, api_key)
    exc = _GroqHttpError(response, text)
    kind = classify_failure(exc)
    status = response.status_code
    if kind is FailureKind.RATE_LIMITED:
        return SearchFailure("rate limit reached (per minute)", kind=kind, retry_after=retry_after_seconds(exc))
    if kind is FailureKind.DAILY_QUOTA:
        return SearchFailure("the daily quota of the Groq search model is used up", kind=kind)
    if kind is FailureKind.TOO_LARGE:
        return SearchFailure("the request is too large for the Groq API", kind=kind)
    if kind is FailureKind.AUTH:
        return SearchFailure(f"Groq rejected the API key (HTTP {status}); check GROQ_API_KEY", kind=kind)
    return SearchFailure(f"Groq API error (HTTP {status}): {text[:REASON_LIMIT]}", kind=kind)


def _walk_urls(node: object) -> Iterator[str]:
    """Every string found under a ``url`` key anywhere in ``node`` (dicts and lists, in document order)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "url" and isinstance(value, str):
                yield value
            else:
                yield from _walk_urls(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_urls(item)


def _walk_strings(node: object) -> Iterator[str]:
    """Every string anywhere in ``node`` (dict values, list items), in document order."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_strings(item)


def _urls_in_text(text: str) -> Iterator[str]:
    for match in _URL_IN_TEXT.finditer(text):
        yield match.group(0).rstrip(_TRAILING_PUNCTUATION)


def _collect_sources(message: dict[str, Any], text: str) -> tuple[str, ...]:
    """Up to :data:`MAX_SOURCES` distinct http(s) links: citations first, then the answer's, then the searched."""
    candidates = [
        *_walk_urls(message.get("annotations")),
        *_urls_in_text(text),
        *_walk_urls(message.get("executed_tools")),
    ]
    sources: list[str] = []
    for url in candidates:
        usable = url.lower().startswith(("http://", "https://")) and len(url) <= MAX_URL_CHARS
        if usable and url not in sources:
            sources.append(url)
    return tuple(sources[:MAX_SOURCES])


def _touched_links(message: dict[str, Any], text: str) -> Iterator[str]:
    """Every link the reply carries or the browser touched (all of them: the shown sources are only five)."""
    tool_parts = (message.get("annotations"), message.get("executed_tools"))
    yield from _urls_in_text(text)
    for part in tool_parts:
        yield from _walk_urls(part)
        for string in _walk_strings(part):
            yield from _urls_in_text(string)


def _is_blocked(message: dict[str, Any], text: str) -> bool:
    """True when any link is a blocked source, or the answer or a fetched page looks like an answer key."""
    if any(blocked_source(url) is not None for url in _touched_links(message, text)):
        return True
    fetched = _walk_strings(message.get("executed_tools"))
    return looks_like_answer_key(text) or any(looks_like_answer_key(string) for string in fetched)


def _token_count(body: dict[str, Any], key: str) -> int | None:
    usage = body.get("usage")
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse(response: requests.Response, seconds: float) -> SearchReply:
    try:
        body = response.json()
        message = body["choices"][0]["message"]
        content = message["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise SearchFailure("the reply of the Groq API could not be read") from exc
    text = content.strip() if isinstance(content, str) else ""
    if not text:
        raise SearchFailure("the search returned no answer")
    return SearchReply(
        text=text,
        sources=_collect_sources(message, text),
        input_tokens=_token_count(body, "prompt_tokens"),
        output_tokens=_token_count(body, "completion_tokens"),
        seconds=seconds,
        blocked=_is_blocked(message, text),
    )


def search(api_key: str, query: str) -> SearchReply:
    """Run ``query`` through the browser-search model; raise :class:`SearchFailure` with a safe reason on failure."""
    started = time.monotonic()
    try:
        response = requests.post(
            SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=build_payload(query),
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.Timeout as exc:
        raise SearchFailure("the Groq request timed out") from exc
    except requests.RequestException as exc:
        raise SearchFailure("could not reach the Groq API") from exc
    if response.status_code != 200:
        raise _failure_for(response, api_key)
    return _parse(response, time.monotonic() - started)


def _fit(text: str, room: int) -> str:
    """``text`` cut to ``room`` characters, the visible truncation marker included."""
    if len(text) <= room:
        return text
    marker = f"\n[truncated: {len(text)} chars total]"
    return text[: max(room - len(marker), 0)].rstrip() + marker


def format_reply(reply: SearchReply, limit: int = MAX_REPLY_CHARS) -> str:
    """The answer followed by its source links, at most ``limit`` characters (the answer gives way, not the links)."""
    block = "\nSources:" + "".join(f"\n- {url}" for url in reply.sources) if reply.sources else ""
    return _fit(reply.text, max(limit - len(block), 0)) + block
