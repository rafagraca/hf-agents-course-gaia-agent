"""``deep_search``: one Groq browser-search call that does several web searches for the agent.

The free Groq quota of the search model is small (200 000 tokens a day, and 8 000 per
minute), so the tool counts its own uses: at most :data:`MAX_PER_QUESTION` per question (the run calls
:meth:`DeepSearchTool.reset_question` before each question) and ``GAIA_DEEP_SEARCH_MAX`` per run (default 5:
one real call read about 31 000 input tokens, the pages the browser search fetched, so 5 calls stay under the
daily quota).
The browser reads pages on Groq's servers, so the sources guard of the web tools (``_sources``) is applied to the
reply: a link to a place that publishes benchmark answers, anywhere in it, or text that looks like an answer key,
makes the whole reply unavailable.
Whatever goes wrong (limits, rate limits, the daily quota, errors) comes back as a short
``deep_search unavailable: ...`` text that points to the other tools; the tool never raises and never shows the key.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from smolagents import Tool

from gaia_agent.config import get_secret
from gaia_agent.provider_errors import FailureKind
from gaia_agent.tools._groq_search import (
    BLOCKED_REASON,
    MAX_COMPLETION_TOKENS,
    SEARCH_MODEL,
    SearchFailure,
    SearchReply,
    format_reply,
    search,
)
from gaia_agent.usage_log import UsageLog

ENV_DEEP_SEARCH_MAX = "GAIA_DEEP_SEARCH_MAX"
DEFAULT_MAX_PER_RUN = 5
MAX_PER_QUESTION = 2
MAX_WAIT_SECONDS = 90.0  # a longer rate-limit wait is not taken: the retry would not come in time to matter
DEFAULT_WAIT_SECONDS = 20.0  # when the provider gives no retry hint
USAGE_LABEL = f"{SEARCH_MODEL} (deep_search)"
FALLBACK_ADVICE = "Use web_search and read_webpage instead."


def _unavailable(reason: str) -> str:
    return f"deep_search unavailable: {reason}. {FALLBACK_ADVICE}"


def _run_limit() -> int:
    """``GAIA_DEEP_SEARCH_MAX`` when it is a whole number of 0 or more, else the default."""
    raw = (os.environ.get(ENV_DEEP_SEARCH_MAX) or "").strip()
    return int(raw) if raw.isdecimal() else DEFAULT_MAX_PER_RUN


class DeepSearchTool(Tool):
    """Multi-step web research in one call (Groq ``openai/gpt-oss-20b`` with browser search; needs ``GROQ_API_KEY``)."""

    name = "deep_search"
    description = (
        "For hard factual questions that need multi-step web research: runs several searches and returns a short "
        "answer with source URLs. It can be wrong; confirm important numbers with read_webpage or wikipedia_page "
        "when you have steps left. Limited uses."
    )
    inputs = {"query": {"type": "string", "description": "The question to research, in full."}}
    output_type = "string"

    def __init__(
        self,
        *,
        max_per_run: int | None = None,
        max_per_question: int = MAX_PER_QUESTION,
        usage_log: Path | str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__()
        self._max_per_run = _run_limit() if max_per_run is None else max_per_run
        self._max_per_question = max_per_question
        self._usage = UsageLog(usage_log)
        self._sleep = sleep
        self._run_uses = 0
        self._question_uses = 0
        self._quota_gone = False

    def reset_question(self) -> None:
        """Start a new question: its own two uses are available again (the run's count is kept)."""
        self._question_uses = 0

    def set_usage_log(self, path: Path | str | None) -> None:
        """Send the consumption of the next calls to ``path`` (the ``usage.jsonl`` of the run); None stops logging."""
        self._usage = UsageLog(path)

    def forward(self, query: str) -> str:
        text = (query or "").strip()
        if not text:
            return "Error: the query is empty."
        api_key = get_secret("GROQ_API_KEY")
        if api_key is None:
            return _unavailable("GROQ_API_KEY is not set")
        refusal = self._refusal()
        if refusal is not None:
            return _unavailable(refusal)
        self._run_uses += 1
        self._question_uses += 1
        try:
            reply, waited, attempts = self._search_with_one_retry(api_key, text)
        except SearchFailure as failure:
            self._quota_gone = self._quota_gone or failure.kind is FailureKind.DAILY_QUOTA
            return _unavailable(str(failure))
        self._log(reply, attempts, waited)
        if reply.blocked:  # the tokens were spent, but nothing that came from such a place reaches the agent
            return _unavailable(BLOCKED_REASON)
        return format_reply(reply)

    def _refusal(self) -> str | None:
        """Why this call may not go out, or None when it may."""
        if self._quota_gone:
            return "the daily quota of the Groq search model is used up"
        if self._run_uses >= self._max_per_run:
            return f"the limit of {self._max_per_run} uses per run is reached"
        if self._question_uses >= self._max_per_question:
            return f"already used {self._max_per_question} times for this question"
        return None

    def _search_with_one_retry(self, api_key: str, query: str) -> tuple[SearchReply, float, int]:
        """Search; after a per-minute rate limit wait the hinted time (at most 90 s) and try once more."""
        try:
            return search(api_key, query), 0.0, 1
        except SearchFailure as failure:
            if failure.kind is not FailureKind.RATE_LIMITED:
                raise
            wait = DEFAULT_WAIT_SECONDS if failure.retry_after is None else failure.retry_after
            if wait > MAX_WAIT_SECONDS:
                too_long = f"rate limit reached; the wait of {wait:.0f} s is too long"
                raise SearchFailure(too_long, kind=failure.kind) from failure
        self._sleep(wait)
        return search(api_key, query), wait, 2

    def _log(self, reply: SearchReply, attempts: int, waited: float) -> None:
        self._usage.append(
            USAGE_LABEL,
            tokens=(reply.input_tokens, reply.output_tokens),
            seconds=reply.seconds,
            estimate=0,
            max_tokens=MAX_COMPLETION_TOKENS,
            attempts=attempts,
            waited=waited,
        )
