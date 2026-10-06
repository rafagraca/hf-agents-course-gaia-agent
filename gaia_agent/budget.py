"""Token budgeting and rate-limit handling for language model calls.

``TokenBudget`` tracks tokens per minute for one model; ``BudgetedModel`` wraps
one ``LiteLLMModel`` per model id (primary first, then the fallbacks) behind the
smolagents ``Model`` interface.

A ``BudgetedModel`` call estimates the prompt and waits until it fits the minute's budget;
calls the model with ``max_tokens`` capped by the settings; on a rate limit or a server hiccup
waits for the provider's ``retry-after`` (or backs off 2, 4, 8, 16 s) and retries, up to five
attempts (a refused generation is retried at once); moves on to the next model when the daily
quota is gone; raises ``PromptTooLargeError`` on "request too large"; and books the real usage
in the usage log.

Keep ONE ``BudgetedModel`` for a whole run: the minute window and the memory of which models
ran out of quota live in the instance. Note that smolagents ends a run when ``generate``
raises: ``CodeAgent.run`` re-raises the error as ``AgentGenerationError`` (original as ``__cause__``).

Typical wiring::

    model = BudgetedModel(settings, run_dir / "usage.jsonl")
    agent = CodeAgent(tools, model, step_callbacks=[make_trim_callback()])
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolagents import ChatMessage, LiteLLMModel, Model, Tool

from gaia_agent.config import Settings, get_secret
from gaia_agent.provider_errors import FailureKind, classify_failure, limit_and_requested, retry_after_seconds
from gaia_agent.tokens import estimate_tokens
from gaia_agent.usage_log import UsageLog

__all__ = ["BudgetedModel", "NoModelAvailableError", "PromptTooLargeError", "TokenBudget"]

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0
SAFETY_MARGIN_PERCENT = 5
MAX_ATTEMPTS = 5  # per model and call: the first try plus four retries
BACKOFF_SECONDS = (2.0, 4.0, 8.0, 16.0)  # used when the provider gives no retry-after hint
MAX_WAIT_SECONDS = 120.0  # a longer wait is no per-minute window: the model counts as exhausted
REQUEST_TIMEOUT_SECONDS = 120.0  # litellm's own default is 6000 s, long enough to wedge a whole run
SECRET_NAME_BY_PROVIDER = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
RETRYABLE = frozenset({FailureKind.RATE_LIMITED, FailureKind.TRANSIENT, FailureKind.BAD_GENERATION})

ModelFactory = Callable[[str, str | None], Model]


class PromptTooLargeError(Exception):
    """A prompt cannot fit the per-request token budget, even after trimming."""


class NoModelAvailableError(RuntimeError):
    """Every model in the chain is out of quota, lacks its API key or is otherwise unusable."""


class _UnavailableError(Exception):
    """Internal: the model can serve no more requests in this run; the message says why."""


class TokenBudget:
    """Tokens-per-minute budget for one model; ``clock`` and ``sleep`` are injectable for tests.

    The window slides over the last 60 seconds and ``tpm_limit`` is used with a 5% safety
    margin. Typical use: ``reserve(estimate)``, make the request, ``record(real_tokens)``.
    """

    def __init__(
        self,
        tpm_limit: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tpm_limit <= 0:
            raise ValueError(f"tpm_limit must be positive, got {tpm_limit}")
        self._tpm_limit = tpm_limit
        self._capacity = tpm_limit * (100 - SAFETY_MARGIN_PERCENT) // 100
        self._clock = clock
        self._sleep = sleep
        self._records: deque[tuple[float, int]] = deque()  # (when, tokens), oldest first
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """Tokens usable per minute: the limit minus the safety margin."""
        return self._capacity

    @property
    def used_tokens(self) -> int:
        """Tokens recorded during the last 60 seconds."""
        with self._lock:
            self._forget_old(self._clock())
            return sum(tokens for _, tokens in self._records)

    def estimate_tokens(self, messages: Sequence[Any]) -> int:
        """Estimate the size of ``messages`` in tokens (characters / 3.5, rounded up; see ``gaia_agent.tokens``).

        Only text is counted; the models used here are text-only.
        """
        return estimate_tokens(messages)

    def reserve(self, tokens: int) -> None:
        """Wait (using the injected ``sleep``) until ``tokens`` fit under the limit for the current minute.

        Nothing is booked: the real usage is booked later with :meth:`record`. A request
        that could never fit, not even in an empty window, raises ``PromptTooLargeError``
        at once instead of waiting forever. A clock that does not move while we sleep (a
        frozen or fake one) ends the wait too: the sleep is taken at face value.
        """
        if tokens < 0:
            raise ValueError(f"tokens must not be negative, got {tokens}")
        if tokens > self._capacity:
            raise PromptTooLargeError(
                f"{tokens} tokens can never fit: only {self._capacity} tokens per minute are usable "
                f"({self._tpm_limit} minus a {SAFETY_MARGIN_PERCENT}% safety margin)"
            )
        while (wait := self._seconds_until_room(tokens)) > 0:
            before = self._clock()
            self._sleep(wait)
            if self._clock() <= before:
                break

    def record(self, tokens: int) -> None:
        """Record ``tokens`` actually consumed by a call."""
        if tokens < 0:
            raise ValueError(f"tokens must not be negative, got {tokens}")
        if tokens == 0:
            return
        with self._lock:
            now = self._clock()
            self._forget_old(now)
            self._records.append((now, tokens))

    def _forget_old(self, now: float) -> None:
        """Drop the records that left the window. The caller holds the lock."""
        while self._records and self._records[0][0] + WINDOW_SECONDS <= now:
            self._records.popleft()

    def _seconds_until_room(self, tokens: int) -> float:
        """0 when ``tokens`` fit now, else how long until enough old records have left the window."""
        with self._lock:
            now = self._clock()
            self._forget_old(now)
            excess = sum(used for _, used in self._records) + tokens - self._capacity
            if excess <= 0:
                return 0.0
            wake_up = now
            for recorded_at, used in self._records:
                wake_up = recorded_at + WINDOW_SECONDS
                excess -= used
                if excess <= 0:
                    break
            return wake_up - now


def _monotonic() -> float:
    """``time.monotonic``, looked up at call time so that tests can patch it."""
    return time.monotonic()


def _sleep(seconds: float) -> None:
    """``time.sleep``, looked up at call time so that tests can patch it."""
    time.sleep(seconds)


def _litellm_model(model_id: str, api_key: str | None) -> Model:
    """The real model. Retrying and waiting are ours to do, so the inner model must not retry by itself."""
    import litellm  # slow to import, and tests replace this factory

    litellm.suppress_debug_info = True  # otherwise litellm prints a "Give Feedback / Get Help" banner per error
    return LiteLLMModel(model_id=model_id, api_key=api_key, retry=False, timeout=REQUEST_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class _Request:
    """The arguments of one ``generate`` call."""

    messages: list[ChatMessage]
    stop_sequences: list[str] | None
    response_format: dict[str, str] | None
    tools: list[Tool] | None
    kwargs: Mapping[str, Any]


class BudgetedModel(Model):
    """Model with a token budget per model id, rate-limit retries and fallback to the next model.

    Applies one ``TokenBudget`` per model, caps ``max_tokens`` at ``settings.max_output_tokens``,
    handles 429/413/``RateLimitError`` (retry-after and backoff), moves on to the next model
    when the daily quota is exhausted, and appends one JSON line per call (model, tokens in/out,
    seconds) to ``usage_log`` when a path is given.

    How Groq counts, checked against the real API on 2026-10-06 (8000 tokens per minute on the
    three default models): a request is admitted when its prompt fits what is left of the minute,
    the minute fills with the real prompt plus reply tokens, and the declared ``max_tokens`` is not
    charged (``max_tokens=9000`` was accepted). So a call reserves just the estimated prompt, and
    books the real usage, reply included, when it is done.
    ``model_id`` is the id of the model that served the last call. Calls are serialised, so the
    budgets stay right when several threads share one instance.
    """

    def __init__(
        self,
        settings: Settings,
        usage_log: Path | None = None,
        model_ids: Sequence[str] | None = None,
        *,
        model_factory: ModelFactory | None = None,
        clock: Callable[[], float] = _monotonic,
        sleep: Callable[[float], None] = _sleep,
    ) -> None:
        """Wrap the models ``model_ids`` (default: ``settings.model_id`` then the fallbacks).

        ``model_factory(model_id, api_key)`` builds the model behind an id, lazily at its first
        use; ``clock`` and ``sleep`` are injectable for tests.
        """
        wanted = model_ids if model_ids is not None else (settings.model_id, *settings.fallback_model_ids)
        ids = tuple(dict.fromkeys(wanted))  # no duplicates, same order
        if not ids:
            raise ValueError("at least one model id is required")
        super().__init__(model_id=ids[0])
        self._settings = settings
        self._usage = UsageLog(usage_log)
        self._model_ids = ids
        self._budgets = {model_id: TokenBudget(settings.tpm_limit, clock=clock, sleep=sleep) for model_id in ids}
        self._factory = model_factory or _litellm_model
        self._clock = clock
        self._sleep = sleep
        self._models: dict[str, Model] = {}
        self._unavailable: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def model_ids(self) -> tuple[str, ...]:
        """The models in the order they are tried."""
        return self._model_ids

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Answer from the first model that can, as described in the module docstring.

        Raises ``PromptTooLargeError`` when the prompt cannot fit, ``NoModelAvailableError`` when
        no model is left, and lets any other provider error through unchanged.
        """
        request = _Request(messages, stop_sequences, response_format, tools_to_call_from, kwargs)
        reasons: dict[str, str] = {}
        with self._lock:
            for model_id in self._model_ids:
                reason = self._unavailable.get(model_id)
                if reason is None:
                    try:
                        return self._generate_with(model_id, request)
                    except _UnavailableError as exc:
                        reason = self._unavailable[model_id] = str(exc)
                        logger.warning("%s is out for the rest of this run: %s", model_id, reason)
                reasons[model_id] = reason
        raise NoModelAvailableError(
            "No model can answer: " + "; ".join(f"{model_id}: {reason}" for model_id, reason in reasons.items())
        )

    def _generate_with(self, model_id: str, request: _Request) -> ChatMessage:
        """Call one model, waiting and retrying on rate limits; ``_UnavailableError`` when it is out of quota."""
        model = self._inner_model(model_id)
        budget = self._budgets[model_id]
        estimate = budget.estimate_tokens(request.messages)
        kwargs = self._completion_kwargs(model_id, request.kwargs)
        attempt, waited = 0, 0.0
        while True:
            attempt += 1
            waited += self._reserve(budget, model_id, estimate)
            started = self._clock()
            try:
                message = model.generate(
                    request.messages,
                    stop_sequences=request.stop_sequences,
                    response_format=request.response_format,
                    tools_to_call_from=request.tools,
                    **kwargs,
                )
            except Exception as exc:
                kind = classify_failure(exc)
                if kind is FailureKind.OTHER or (kind in RETRYABLE and attempt >= MAX_ATTEMPTS):
                    raise
                waited += self._wait_or_raise(model_id, kind, exc, attempt)
                continue
            seconds = self._clock() - started
            usage = message.token_usage
            tokens = (None, None) if usage is None else (usage.input_tokens, usage.output_tokens)
            budget.record((estimate + kwargs["max_tokens"]) if usage is None else usage.total_tokens)
            self.model_id = model_id
            self._usage.append(
                model_id,
                tokens=tokens,
                seconds=seconds,
                estimate=estimate,
                max_tokens=kwargs["max_tokens"],
                attempts=attempt,
                waited=waited,
            )
            return message

    def _inner_model(self, model_id: str) -> Model:
        """The model behind ``model_id``, built on first use with the API key of its provider."""
        model = self._models.get(model_id)
        if model is None:
            model = self._models[model_id] = self._factory(model_id, self._api_key(model_id))
        return model

    @staticmethod
    def _api_key(model_id: str) -> str | None:
        """The secret of the provider named in ``model_id`` (``None`` for providers that need none here)."""
        secret_name = SECRET_NAME_BY_PROVIDER.get(model_id.split("/", 1)[0])
        if secret_name is None:
            return None
        key = get_secret(secret_name)
        if key is None:
            raise _UnavailableError(f"{secret_name} is not set")
        return key

    def _completion_kwargs(self, model_id: str, extra: Mapping[str, Any]) -> dict[str, Any]:
        """The keyword arguments for the model: the caller's, with ``max_tokens`` capped by the settings."""
        cap = self._settings.max_output_tokens
        asked = extra.get("max_tokens")
        kwargs = {**extra, "max_tokens": min(asked, cap) if isinstance(asked, int) and asked > 0 else cap}
        if "gpt-oss" in model_id.lower():
            kwargs.setdefault("reasoning_effort", self._settings.reasoning_effort)
        else:  # on qwen it switches the reasoning on and the answer can come back empty
            kwargs.pop("reasoning_effort", None)
        return kwargs

    def _reserve(self, budget: TokenBudget, model_id: str, estimate: int) -> float:
        """Wait until the budget has room for the prompt; return the seconds waited."""
        started = self._clock()
        try:
            budget.reserve(estimate)
        except PromptTooLargeError as exc:
            raise PromptTooLargeError(
                f"{model_id}: a prompt of about {estimate} tokens cannot fit in {budget.capacity} tokens per minute"
            ) from exc
        return self._clock() - started

    def _wait_or_raise(self, model_id: str, kind: FailureKind, exc: BaseException, attempt: int) -> float:
        """Turn a failed call into a pause (returns the seconds slept) or into the error to raise."""
        if kind is FailureKind.TOO_LARGE:
            figures = limit_and_requested(exc)
            detail = f" (limit {figures[0]}, requested {figures[1]})" if figures else ""
            raise PromptTooLargeError(f"{model_id} rejected the request as too large{detail}") from exc
        if kind is FailureKind.DAILY_QUOTA:
            raise _UnavailableError("daily quota exhausted") from exc
        if kind is FailureKind.AUTH:  # said in our own words: the provider's text could echo the key
            raise _UnavailableError("the provider rejected the API key (check it and its permissions)") from exc
        if kind is FailureKind.MODEL_UNAVAILABLE:
            raise _UnavailableError("the provider does not offer this model (retired, or the id has a typo)") from exc
        if kind is FailureKind.BAD_GENERATION:  # nothing to wait for: the next sample is a new draw
            logger.warning("%s: the provider refused the reply, trying again (attempt %d)", model_id, attempt + 1)
            return 0.0
        seconds = retry_after_seconds(exc)
        if seconds is None:
            seconds = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS)) - 1]
        if seconds > MAX_WAIT_SECONDS:
            raise _UnavailableError(f"asked to wait {seconds:.0f}s, more than {MAX_WAIT_SECONDS:.0f}s") from exc
        logger.warning("%s: %s, waiting %.1fs before attempt %d", model_id, kind.value, seconds, attempt + 1)
        self._sleep(seconds)
        return seconds
