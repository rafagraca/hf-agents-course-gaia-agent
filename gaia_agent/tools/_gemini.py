"""Video questions answered by Gemini through the REST ``generateContent`` method.

The API key travels in the ``x-goog-api-key`` header and never in the URL, so it
cannot leak through an exception message or a log line; error replies are
additionally scrubbed of it. A busy service (HTTP 429 or 5xx) is retried twice
with a short back-off. Every failure becomes a short :class:`ToolError`.
"""

from __future__ import annotations

import os
import re
import time

import requests

from gaia_agent.config import get_secret
from gaia_agent.tools._api_errors import error_message
from gaia_agent.tools._local_files import ToolError
from gaia_agent.tools._youtube import canonical_url, parse_video_id

ENV_MODEL = "GAIA_GEMINI_MODEL"
DEFAULT_MODEL = "gemini-2.5-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REQUEST_TIMEOUT_S = (10, 120)  # connect, read
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_DELAYS_S = (2.0, 5.0)

_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def gemini_model() -> str:
    """The model id: ``GAIA_GEMINI_MODEL`` (a ``models/`` prefix is allowed) or the default."""
    configured = (os.environ.get(ENV_MODEL) or "").strip().removeprefix("models/")
    if not configured:
        return DEFAULT_MODEL
    if not _MODEL_NAME.fullmatch(configured):
        raise ToolError(f"{ENV_MODEL} is not a valid Gemini model id")
    return configured


def gemini_key() -> str:
    """The API key, or a :class:`ToolError` saying it is missing."""
    key = get_secret("GEMINI_API_KEY")
    if key is None:
        raise ToolError("GEMINI_API_KEY is not set, so Gemini cannot be used")
    return key


# --------------------------------------------------------------------------- request parts


def video_part(url: str) -> dict:
    """The ``file_data`` part pointing Gemini at a public YouTube video."""
    video_id = parse_video_id(url)
    if video_id is None:
        raise ToolError("only YouTube video URLs are supported")
    return {"file_data": {"file_uri": canonical_url(video_id)}}


def question_prompt(subject: str, question: object) -> str:
    """The text part: the question, framed so the model admits what it cannot see or hear."""
    text = question.strip() if isinstance(question, str) else ""
    if not text:
        raise ToolError("question must not be empty")
    return (
        f"Answer the question about this {subject} precisely and concisely. "
        "If something cannot be seen or heard clearly, say so instead of guessing.\n\n"
        f"Question: {text}"
    )


# --------------------------------------------------------------------------- the call


def _post(url: str, api_key: str, payload: dict) -> requests.Response:
    try:
        return requests.post(url, headers={"x-goog-api-key": api_key}, json=payload, timeout=REQUEST_TIMEOUT_S)
    except requests.Timeout as exc:
        raise ToolError("the Gemini request timed out") from exc
    except requests.RequestException as exc:
        raise ToolError("could not reach the Gemini API") from exc


def _send_with_retries(url: str, api_key: str, payload: dict) -> requests.Response:
    response = _post(url, api_key, payload)
    for delay in RETRY_DELAYS_S:
        if response.status_code not in RETRY_STATUSES:
            break
        _pause(delay)
        response = _post(url, api_key, payload)
    return response


def _reply_json(response: requests.Response, model: str, api_key: str) -> dict:
    status = response.status_code
    if status in (401, 403):
        raise ToolError(
            f"Gemini rejected the request (HTTP {status}): {error_message(response, api_key)}; check GEMINI_API_KEY"
        )
    if status == 404:
        raise ToolError(f"Gemini model {model!r} was not found; set {ENV_MODEL} to a current model id")
    if status == 429:
        raise ToolError("Gemini rate limit or daily quota reached (HTTP 429); try again later")
    if status != 200:
        raise ToolError(f"Gemini API error (HTTP {status}): {error_message(response, api_key)}")
    try:
        data = response.json()
    except ValueError as exc:
        raise ToolError("Gemini returned an unreadable reply") from exc
    if not isinstance(data, dict):
        raise ToolError("Gemini returned an unreadable reply")
    return data


def _answer_text(data: dict) -> str:
    candidates = data.get("candidates")
    first = candidates[0] if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) else {}
    parts = (first.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")).strip()
    if text:
        return text
    block = (data.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise ToolError(f"Gemini blocked the request ({block})")
    reason = first.get("finishReason") or ("unknown" if first else "no candidates")
    raise ToolError(f"Gemini returned no text (finish reason: {reason})")


def ask_gemini(api_key: str, media: dict, prompt: str) -> str:
    """Send one media part and a text prompt to Gemini and return the answer text."""
    model = gemini_model()
    payload = {"contents": [{"role": "user", "parts": [media, {"text": prompt}]}]}
    response = _send_with_retries(ENDPOINT.format(model=model), api_key, payload)
    return _answer_text(_reply_json(response, model, api_key))
