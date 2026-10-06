"""Tests for waiting out the daily quota (``--wait-for-quota``): what counts as a quota error, and how long to wait."""

from __future__ import annotations

import pytest
from budget_fakes import PRIMARY, SECOND, TPD_MESSAGE, FakeClock, FakeModel, rate_limit_error

from gaia_agent._quota import (
    MAX_PAUSES,
    MAX_SINGLE_WAIT_SECONDS,
    MIN_WAIT_SECONDS,
    QuotaWaiter,
    is_quota_exhausted,
    known_retry_after,
)
from gaia_agent.budget import BudgetedModel, NoModelAvailableError
from gaia_agent.config import Settings

HOUR = 3600.0
QUOTA_ERROR = "NoModelAvailableError: No model can answer: groq/a: daily quota exhausted; groq/b: daily quota exhausted"


def real_error_of(settings: Settings, monkeypatch: pytest.MonkeyPatch, errors: dict[str, BaseException]) -> str:
    """The text ``answer_question`` records when ``BudgetedModel`` runs out of models with these provider errors."""
    monkeypatch.setenv("GROQ_API_KEY", "key-for-tests")
    clock = FakeClock()
    model = BudgetedModel(
        settings,
        None,
        model_ids=list(errors),
        model_factory=lambda model_id, key: FakeModel(model_id, [errors[model_id]] * 8),
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(NoModelAvailableError) as caught:
        model.generate([])
    return f"{type(caught.value).__name__}: {caught.value}"


# --- which errors are worth waiting for -------------------------------------------------------------------------


def test_the_error_of_a_budgeted_model_without_daily_quota_is_recognised(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the wording of ``budget.py`` that ``is_quota_exhausted`` depends on."""
    text = real_error_of(
        settings, monkeypatch, {PRIMARY: rate_limit_error(TPD_MESSAGE), SECOND: rate_limit_error(TPD_MESSAGE)}
    )

    assert is_quota_exhausted(text)
    assert known_retry_after(text) is None  # the budget does not pass the provider's hint on


def test_a_rejected_api_key_is_not_worth_waiting_for(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    auth = litellm.AuthenticationError(message="bad key", llm_provider="groq", model="x")

    assert not is_quota_exhausted(real_error_of(settings, monkeypatch, {PRIMARY: auth, SECOND: auth}))


def test_a_missing_api_key_is_not_worth_waiting_for(settings: Settings) -> None:
    text = "NoModelAvailableError: No model can answer: groq/a: GROQ_API_KEY is not set"

    assert not is_quota_exhausted(text)


def test_one_model_out_of_quota_is_enough_when_the_others_are_just_out_of_reach() -> None:
    text = "NoModelAvailableError: No model can answer: groq/a: GROQ_API_KEY is not set; groq/b: daily quota exhausted"

    assert is_quota_exhausted(text)


@pytest.mark.parametrize("text", [None, "", "RuntimeError: daily quota exhausted", "ValueError: boom"])
def test_only_the_no_model_error_counts(text: str | None) -> None:
    assert not is_quota_exhausted(text)


def test_a_retry_hint_in_the_error_is_read() -> None:
    assert known_retry_after(QUOTA_ERROR + ". Please try again in 6m30s.") == 390.0
    assert known_retry_after(QUOTA_ERROR) is None


# --- how long to wait --------------------------------------------------------------------------------------------


def waiter(clock: FakeClock, hours: float = 36.0) -> tuple[QuotaWaiter, list[str]]:
    said: list[str] = []
    return QuotaWaiter(hours, clock=clock, sleep=clock.sleep, announce=said.append), said


def test_without_a_known_retry_after_it_waits_fifteen_minutes() -> None:
    clock = FakeClock()
    quota, said = waiter(clock)

    assert quota.wait(QUOTA_ERROR) is True

    assert clock.sleeps == [MAX_SINGLE_WAIT_SECONDS] == [900.0]
    assert "15 min" in said[0]


def test_a_known_retry_after_shorter_than_fifteen_minutes_is_used() -> None:
    clock = FakeClock()
    quota, _ = waiter(clock)

    quota.wait(QUOTA_ERROR + ". Please try again in 6m30s.")

    assert clock.sleeps == [390.0]


def test_a_known_retry_after_longer_than_fifteen_minutes_is_cut_to_fifteen() -> None:
    clock = FakeClock()
    quota, _ = waiter(clock)

    quota.wait(QUOTA_ERROR + ". Please try again in 3h20m.")

    assert clock.sleeps == [900.0]


@pytest.mark.parametrize("hint", ["0s", "1s", "2s", "45s"])
def test_a_very_short_hint_still_waits_a_full_minute(hint: str) -> None:
    """A hint that short means a per-minute limit was taken for the daily one: not worth a request a second."""
    clock = FakeClock()
    quota, _ = waiter(clock)

    quota.wait(QUOTA_ERROR + f". Please try again in {hint}.")

    assert clock.sleeps == [MIN_WAIT_SECONDS] == [60.0]


def test_short_hints_make_at_most_one_attempt_a_minute_and_a_bounded_number_in_all() -> None:
    clock = FakeClock()
    quota, said = waiter(clock)

    pauses = 0
    while quota.wait(QUOTA_ERROR + ". Please try again in 1s."):
        pauses += 1

    assert pauses == MAX_PAUSES == 200
    assert min(clock.sleeps) >= 60.0
    assert quota.pauses == MAX_PAUSES
    assert "pauses" in said[-1]


def test_the_cap_on_pauses_is_announced_once_per_refusal_and_not_slept() -> None:
    clock = FakeClock()
    quota, _ = waiter(clock)
    quota.pauses = MAX_PAUSES

    assert quota.wait(QUOTA_ERROR) is False
    assert clock.sleeps == []


def test_it_gives_up_when_the_maximum_wait_is_over() -> None:
    clock = FakeClock()
    quota, said = waiter(clock, hours=1.0)

    results = [quota.wait(QUOTA_ERROR) for _ in range(6)]

    assert results == [True, True, True, True, False, False]
    assert clock.slept == HOUR
    assert "1 h" in said[-1]


def test_the_last_wait_is_cut_to_what_is_left() -> None:
    clock = FakeClock()
    quota, _ = waiter(clock, hours=0.3)  # 18 minutes

    assert quota.wait(QUOTA_ERROR) and quota.wait(QUOTA_ERROR)

    assert clock.sleeps == [900.0, 180.0]


def test_time_spent_answering_between_waits_counts_towards_the_maximum() -> None:
    clock = FakeClock()
    quota, _ = waiter(clock, hours=1.0)
    quota.wait(QUOTA_ERROR)

    clock.now += 2700.0  # three quarters of an hour of answering questions

    assert quota.wait(QUOTA_ERROR) is False


def test_a_waiter_that_is_never_asked_does_not_start_its_clock() -> None:
    clock = FakeClock()
    quota, _ = waiter(clock, hours=1.0)

    clock.now += 10 * HOUR

    assert quota.wait(QUOTA_ERROR) is True


def test_a_non_positive_maximum_is_refused() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="positive"):
        QuotaWaiter(0, clock=clock, sleep=clock.sleep)
