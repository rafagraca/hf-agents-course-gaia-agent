"""Waiting out the daily quota of the models (``run --wait-for-quota``).

Groq's free tier gives each model a daily allowance. When every model of the chain has used it up,
``BudgetedModel`` raises ``NoModelAvailableError``, and ``answer_question`` records it as an error that starts
with ``NoModelAvailableError:``. That error is worth waiting for only when it says a daily quota is the reason
(a rejected key will not mend itself). ``QuotaWaiter`` sleeps in slices of at most fifteen minutes, so that a
quota that comes back is noticed soon, and gives up after a maximum total of time and of pauses.

Both clock and sleep are injected so that tests need not wait.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from gaia_agent._cli import say
from gaia_agent.provider_errors import retry_after_seconds

__all__ = [
    "DEFAULT_MAX_WAIT_HOURS",
    "MAX_PAUSES",
    "MAX_SINGLE_WAIT_SECONDS",
    "MIN_WAIT_SECONDS",
    "QuotaWaiter",
    "describe_duration",
    "is_quota_exhausted",
    "known_retry_after",
]

DEFAULT_MAX_WAIT_HOURS = 36.0
MAX_SINGLE_WAIT_SECONDS = 15 * 60.0
# A hint shorter than this is a per-minute limit mistaken for the daily one: asking again every second would be
# thousands of real requests, so a pause is never shorter than a minute (unless the maximum wait is nearly over).
MIN_WAIT_SECONDS = 60.0
MAX_PAUSES = 200  # in all: a daily quota comes back within a day, which is at most 96 pauses of fifteen minutes

# What ``budget.py`` writes for a model whose daily quota is gone, and the error ``answer_question`` records
# when no model is left (a test pins both against the real ``BudgetedModel``).
NO_MODEL_ERROR_PREFIX = "NoModelAvailableError:"
DAILY_QUOTA_REASON = "daily quota exhausted"


def is_quota_exhausted(error: str | None) -> bool:
    """True when ``error`` (the text recorded for a failed question) says no model is left because of a daily quota."""
    return error is not None and error.startswith(NO_MODEL_ERROR_PREFIX) and DAILY_QUOTA_REASON in error


def known_retry_after(error: str) -> float | None:
    """The wait the provider asked for ("try again in 6m30s") when the error text still carries it."""
    return retry_after_seconds(Exception(error))


class QuotaWaiter:
    """Sleeps while the quota is gone, up to ``max_wait_hours`` counted from the first time it is needed."""

    def __init__(
        self,
        max_wait_hours: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        announce: Callable[[str], None] = say,
    ) -> None:
        if max_wait_hours <= 0:
            raise ValueError(f"the maximum wait must be positive, got {max_wait_hours}")
        self._max_hours = max_wait_hours
        self._clock = clock
        self._sleep = sleep
        self._announce = announce
        self._deadline: float | None = None
        self.pauses = 0
        self.waited_seconds = 0.0

    def wait(self, error: str) -> bool:
        """Sleep once, for ``min(the provider's hint, 15 min)`` (15 min without a hint); False when time is up."""
        now = self._clock()
        if self._deadline is None:
            self._deadline = now + self._max_hours * 3600.0
        remaining = self._deadline - now
        if remaining <= 0:
            self._announce(
                f"Giving up: the models were still out of daily quota after waiting up to {self._max_hours:g} h."
            )
            return False
        if self.pauses >= MAX_PAUSES:
            self._announce(f"Giving up: the models were still out of daily quota after {self.pauses} pauses.")
            return False
        hint = known_retry_after(error)
        seconds = min(MAX_SINGLE_WAIT_SECONDS if hint is None else max(hint, MIN_WAIT_SECONDS), MAX_SINGLE_WAIT_SECONDS)
        seconds = min(seconds, remaining)
        self._announce(
            f"Every model is out of daily quota: waiting {describe_duration(seconds)}, "
            f"then trying the same question again ({remaining / 3600.0:.1f} h of waiting left)."
        )
        self._sleep(seconds)
        self.pauses += 1
        self.waited_seconds += seconds
        return True


def describe_duration(seconds: float) -> str:
    return f"{seconds / 60.0:g} min" if seconds >= 60.0 else f"{seconds:g} s"
