"""Tell the failures of a language model provider apart: rate limits, oversized requests, daily quotas,
server hiccups and refused generations.

The checks read the HTTP status and the text of the error, not litellm's exception
classes, so they hold for other SDKs too. One quirk matters: litellm reports a Groq
``413 Request too large`` as a ``RateLimitError`` whose status is 429, so the text has to
be checked before the status.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any


class FailureKind(Enum):
    """How a failed model call should be handled."""

    OTHER = "other"  # not ours to handle: let it propagate
    TOO_LARGE = "too_large"  # the request itself is too big: retrying cannot help
    DAILY_QUOTA = "daily_quota"  # the quota of the day is gone: use another model
    RATE_LIMITED = "rate_limited"  # a per-minute limit: wait, then retry
    TRANSIENT = "transient"  # a server error, timeout or dropped connection: retrying may work
    # The provider refused the model's own reply, e.g. Groq's 400 tool_use_failed when gpt-oss answers with a
    # tool call nobody offered. Sampling is random, so retrying the same request usually works.
    BAD_GENERATION = "bad_generation"
    MODEL_UNAVAILABLE = "model_unavailable"  # the model does not exist (any more) for this provider: use another
    AUTH = "auth"  # the API key is rejected (401) or lacks the permission (403): this model is out of reach


_BAD_GENERATION_TEXT = re.compile(r"tool_use_failed|failed_generation", re.IGNORECASE)
_MODEL_GONE_TEXT = re.compile(
    r"model_not_found|model_decommissioned|has been decommissioned|model\b[^.\n]{0,80}\bdoes not exist|no such model",
    re.IGNORECASE,
)
_AUTH_CLASSES = frozenset({"AuthenticationError", "PermissionDeniedError"})
_MODEL_GONE_STATUSES = frozenset({None, 400, 404, 410, 422})  # a status that cannot be a rate limit or a server error
_TOO_LARGE_TEXT = re.compile(r"request too large", re.IGNORECASE)
_RATE_LIMIT_TEXT = re.compile(r"rate[\s_-]*limit|too many requests", re.IGNORECASE)
_BARE_429 = re.compile(r"\b429\b")
_DAILY_TEXT = re.compile(r"per[\s_-]*day|\b(?:tpd|rpd)\b|\bdaily\b|insufficient[\s_-]*quota", re.IGNORECASE)
# "Please try again in 6m30s." / "in 2.5s" / "in 850ms": one or more number+unit pairs.
_RETRY_IN_TEXT = re.compile(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.IGNORECASE)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.IGNORECASE)
_LIMIT_AND_REQUESTED = re.compile(r"Limit (\d+),(?: Used \d+,)? Requested (\d+)")
_SECONDS_PER_UNIT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_TRANSIENT_STATUSES = frozenset({408, 500, 502, 503, 504})
_TRANSIENT_CLASSES = frozenset(
    {
        "Timeout",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "BadGatewayError",
        "ServiceUnavailableError",
        "TimeoutError",
        "ConnectionError",
    }
)


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _class_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _is_too_large(exc: BaseException, text: str) -> bool:
    return (
        _status_code(exc) == 413
        or "ContextWindowExceededError" in _class_names(exc)
        or _TOO_LARGE_TEXT.search(text) is not None
    )


def _is_rate_limited(exc: BaseException, text: str) -> bool:
    status = _status_code(exc)
    if status == 429 or "RateLimitError" in _class_names(exc):
        return True
    if status is not None:
        return False  # another status is not a rate limit, whatever the text says
    return _RATE_LIMIT_TEXT.search(text) is not None or _BARE_429.search(text) is not None


def _is_transient(exc: BaseException) -> bool:
    return _status_code(exc) in _TRANSIENT_STATUSES or not _TRANSIENT_CLASSES.isdisjoint(_class_names(exc))


def _is_auth_failure(exc: BaseException) -> bool:
    return _status_code(exc) in (401, 403) or not _AUTH_CLASSES.isdisjoint(_class_names(exc))


def _is_model_gone(exc: BaseException, text: str) -> bool:
    status = _status_code(exc)
    if status == 404 or "NotFoundError" in _class_names(exc):
        return True
    return status in _MODEL_GONE_STATUSES and _MODEL_GONE_TEXT.search(text) is not None


def classify_failure(exc: BaseException) -> FailureKind:
    """Decide how a failed model call should be handled."""
    text = str(exc)
    if _BAD_GENERATION_TEXT.search(text):  # first: the refused reply is quoted in the error and could say anything
        return FailureKind.BAD_GENERATION
    if _is_too_large(exc, text):
        return FailureKind.TOO_LARGE
    if _is_auth_failure(exc):
        return FailureKind.AUTH
    if _is_model_gone(exc, text):
        return FailureKind.MODEL_UNAVAILABLE
    if _is_rate_limited(exc, text):
        return FailureKind.DAILY_QUOTA if _DAILY_TEXT.search(text) else FailureKind.RATE_LIMITED
    return FailureKind.TRANSIENT if _is_transient(exc) else FailureKind.OTHER


def _seconds_or_none(raw: Any) -> float | None:
    try:
        seconds = float(raw)
    except (TypeError, ValueError):  # a missing header (None) or text that is not a number
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _retry_after_header(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return _seconds_or_none(headers.get("retry-after"))
    except AttributeError:
        return None  # no response, no headers, or headers that are not a mapping


def _retry_after_text(text: str) -> float | None:
    match = _RETRY_IN_TEXT.search(text)
    if match is None:
        return None
    return sum(
        float(amount) * _SECONDS_PER_UNIT[unit.lower()] for amount, unit in _DURATION_PART.findall(match.group(1))
    )


def retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds the provider asks us to wait: the ``retry-after`` header or "try again in ..." text.

    When both are present the longer one wins (the header is rounded to whole seconds).
    Returns None when the provider gave no hint.
    """
    hints = [h for h in (_retry_after_header(exc), _retry_after_text(str(exc))) if h is not None]
    return max(hints) if hints else None


def limit_and_requested(exc: BaseException) -> tuple[int, int] | None:
    """The ``(limit, requested)`` token figures quoted in a Groq rate-limit message, if any."""
    match = _LIMIT_AND_REQUESTED.search(str(exc))
    return (int(match.group(1)), int(match.group(2))) if match else None
