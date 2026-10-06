"""Tests for the tokens-per-minute budget and the budgeted model (fake clock, scripted models, no network).

Model selection, the usage log, secrets, threads and the real litellm/smolagents stack are covered in
``test_budget_integration.py``.
"""

from __future__ import annotations

import functools

import pytest
from budget_fakes import (
    PRIMARY,
    SECOND,
    THIRD,
    TPD_MESSAGE,
    FakeClock,
    HttpError,
    litellm,
    make_rig,
    prompt,
    provide_keys,
    rate_limit_error,
    reply,
)
from smolagents.models import ChatMessage, MessageRole

from gaia_agent.budget import NoModelAvailableError, PromptTooLargeError, TokenBudget

SLOW = "Rate limit reached"  # a rate limit without any retry-after hint: the model backs off by itself


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def budget(clock: FakeClock) -> TokenBudget:
    return TokenBudget(8000, clock=clock, sleep=clock.sleep)  # 7600 tokens usable after the 5% margin


@pytest.fixture
def build(settings, monkeypatch: pytest.MonkeyPatch):
    """``build({model_id: [replies or errors]}, ...)``: a budgeted model on scripted fake models."""
    provide_keys(monkeypatch)
    return functools.partial(make_rig, settings)


class TestEstimateTokens:
    def test_it_counts_characters_over_three_and_a_half_rounded_up(self, budget: TokenBudget) -> None:
        assert budget.estimate_tokens([{"role": "user", "content": "x" * 350}]) == 100
        assert budget.estimate_tokens([{"role": "user", "content": "x" * 351}]) == 101

    def test_it_reads_the_text_parts_of_chat_messages(self, budget: TokenBudget) -> None:
        parts = [{"type": "text", "text": "a" * 35}, {"type": "text", "text": "b" * 35}]

        assert budget.estimate_tokens([ChatMessage(role=MessageRole.USER, content=parts)]) == 20

    def test_it_adds_up_every_message(self, budget: TokenBudget) -> None:
        messages = [{"role": "system", "content": "a" * 70}, {"role": "user", "content": "b" * 70}]

        assert budget.estimate_tokens(messages) == 40

    def test_it_ignores_missing_and_non_text_content(self, budget: TokenBudget) -> None:
        messages = [ChatMessage(role=MessageRole.USER, content=None), {"content": [{"type": "image", "image": 1}]}]

        assert budget.estimate_tokens(messages) == 0

    def test_no_messages_cost_nothing(self, budget: TokenBudget) -> None:
        assert budget.estimate_tokens([]) == 0


WAIT_SCENARIOS = [
    # records as (seconds after the start, tokens); when to reserve; how much; expected wait; tokens still used
    pytest.param([(0, 7000)], 0, 1000, 60, 0, id="one old record"),
    pytest.param([(0, 4000), (30, 3000)], 40, 2000, 20, 3000, id="only the first record has to expire"),
    pytest.param([(0, 2000), (10, 2000), (20, 2000), (30, 1600)], 30, 3000, 40, 3600, id="two records have to expire"),
    pytest.param([(0, 7600)], 59, 100, 1, 0, id="a record just under a minute old still counts"),
]


class TestTokenBudget:
    def test_the_usable_budget_keeps_a_five_percent_margin(self, budget: TokenBudget) -> None:
        assert budget.capacity == 7600

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_limit_is_rejected(self, limit: int) -> None:
        with pytest.raises(ValueError, match="tpm_limit"):
            TokenBudget(limit)

    def test_an_empty_window_never_waits(self, budget: TokenBudget, clock: FakeClock) -> None:
        budget.reserve(7600)

        assert clock.sleeps == []

    def test_reserving_does_not_use_up_the_budget(self, budget: TokenBudget, clock: FakeClock) -> None:
        budget.reserve(7000)
        budget.reserve(7000)

        assert clock.sleeps == []
        assert budget.used_tokens == 0

    def test_a_request_larger_than_the_whole_budget_fails_instead_of_waiting(
        self, budget: TokenBudget, clock: FakeClock
    ) -> None:
        with pytest.raises(PromptTooLargeError, match="7601"):
            budget.reserve(7601)

        assert clock.sleeps == []

    def test_reserving_zero_tokens_always_passes(self, budget: TokenBudget, clock: FakeClock) -> None:
        budget.record(7600)

        budget.reserve(0)

        assert clock.sleeps == []

    def test_a_negative_reservation_is_rejected(self, budget: TokenBudget) -> None:
        with pytest.raises(ValueError, match="negative"):
            budget.reserve(-1)

    @pytest.mark.parametrize(("records", "reserve_at", "tokens", "wait", "used_after"), WAIT_SCENARIOS)
    def test_it_waits_until_enough_old_usage_has_left_the_window(
        self, budget: TokenBudget, clock: FakeClock, records, reserve_at, tokens, wait, used_after
    ) -> None:
        start = clock.now
        for offset, used in records:
            clock.now = start + offset
            budget.record(used)
        clock.now = start + reserve_at

        budget.reserve(tokens)

        assert clock.slept == pytest.approx(wait, abs=0.1)
        assert budget.used_tokens == used_after

    def test_usage_a_minute_old_no_longer_counts(self, budget: TokenBudget, clock: FakeClock) -> None:
        budget.record(7600)
        clock.now += 60

        budget.reserve(7600)

        assert clock.sleeps == []

    def test_a_clock_that_never_moves_does_not_trap_the_wait(self) -> None:
        sleeps: list[float] = []
        frozen = TokenBudget(8000, clock=lambda: 1000.0, sleep=sleeps.append)
        frozen.record(7000)

        frozen.reserve(1000)  # a sleep that leaves the clock where it was must not make it spin forever

        assert sleeps == [60.0]

    def test_recorded_usage_adds_up_inside_the_window(self, budget: TokenBudget, clock: FakeClock) -> None:
        budget.record(100)
        clock.now += 10
        budget.record(250)

        assert budget.used_tokens == 350

    def test_recording_nothing_changes_nothing(self, budget: TokenBudget) -> None:
        budget.record(0)

        assert budget.used_tokens == 0

    def test_a_negative_record_is_rejected(self, budget: TokenBudget) -> None:
        with pytest.raises(ValueError, match="negative"):
            budget.record(-5)


class TestGenerate:
    def test_the_request_is_forwarded_and_the_reply_comes_back_unchanged(self, build) -> None:
        expected = reply("hello")
        rig = build({PRIMARY: [expected]})
        messages, tools = prompt(40), [object()]

        result = rig.model.generate(
            messages,
            stop_sequences=["Observation:"],
            response_format={"type": "json_object"},
            tools_to_call_from=tools,
            custom_flag=True,
        )

        (call,) = rig.fakes[PRIMARY].calls
        assert result is expected
        assert call.messages is messages
        assert call.stop_sequences == ["Observation:"]
        assert call.response_format == {"type": "json_object"}
        assert call.tools is tools
        assert call.kwargs["custom_flag"] is True

    def test_calling_the_model_object_is_the_same_as_generate(self, build) -> None:
        expected = reply("hello")

        assert build({PRIMARY: [expected]}).model(prompt(40)) is expected

    @pytest.mark.parametrize(("asked", "sent"), [(None, 2000), (50_000, 2000), (300, 300)])
    def test_max_tokens_never_exceeds_the_configured_output_limit(self, build, asked, sent) -> None:
        rig = build({PRIMARY: [reply()]}, max_output_tokens=2000)

        rig.model.generate(prompt(40), **({} if asked is None else {"max_tokens": asked}))

        assert rig.fakes[PRIMARY].calls[0].kwargs["max_tokens"] == sent

    def test_reasoning_effort_goes_only_to_gpt_oss_models(self, build) -> None:
        scripts = {PRIMARY: [rate_limit_error(TPD_MESSAGE)], SECOND: [rate_limit_error(TPD_MESSAGE)], THIRD: [reply()]}
        rig = build(scripts, reasoning_effort="low")

        rig.model.generate(prompt(40))

        sent = {model_id: fake.calls[0].kwargs.get("reasoning_effort") for model_id, fake in rig.fakes.items()}
        assert sent == {PRIMARY: "low", SECOND: None, THIRD: "low"}

    def test_qwen_never_gets_a_reasoning_effort_not_even_the_callers(self, build) -> None:
        rig = build({SECOND: [reply()]}, model_ids=[SECOND], reasoning_effort="low")

        rig.model.generate(prompt(40), reasoning_effort="high")

        assert "reasoning_effort" not in rig.fakes[SECOND].calls[0].kwargs

    def test_a_reasoning_effort_chosen_by_the_caller_is_kept(self, build) -> None:
        rig = build({PRIMARY: [reply()]})

        rig.model.generate(prompt(40), reasoning_effort="high")

        assert rig.fakes[PRIMARY].calls[0].kwargs["reasoning_effort"] == "high"


class TestBudgeting:
    """Groq admits a request when its prompt fits what is left of the minute (7600 usable here); the declared
    max_tokens is not charged, so a call reserves the estimated prompt alone and books the real usage afterwards."""

    def test_a_prompt_too_big_for_the_minute_is_refused_before_it_is_sent(self, build) -> None:
        rig = build({PRIMARY: [reply()]})

        with pytest.raises(PromptTooLargeError, match="7601"):
            rig.model.generate(prompt(26_601))  # 7601 tokens estimated
        assert rig.fakes[PRIMARY].calls == []
        rig.model.generate(prompt(26_600))  # 7600: fits exactly

        assert len(rig.fakes[PRIMARY].calls) == 1

    def test_the_output_cap_is_not_part_of_the_reservation(self, build) -> None:
        rig = build({PRIMARY: [reply()]}, max_output_tokens=2000)

        rig.model.generate(prompt(21_000))  # 6000 tokens: with the 2000-token cap added it would not fit 7600

        assert rig.clock.sleeps == []
        assert rig.fakes[PRIMARY].calls[0].kwargs["max_tokens"] == 2000

    def test_a_call_does_not_wait_while_the_minute_has_room(self, build) -> None:
        rig = build({PRIMARY: [reply(tokens_in=5900, tokens_out=100), reply()]})

        rig.model.generate(prompt(40))
        rig.model.generate(prompt(40))  # 6000 used + 12 estimated: fits

        assert rig.clock.sleeps == []

    def test_a_call_waits_when_the_last_minute_was_busy(self, build) -> None:
        # A 40-character prompt estimates at 12 tokens: only the real usage (7590) can make the next call wait.
        rig = build({PRIMARY: [reply(tokens_in=7000, tokens_out=590), reply()]})

        rig.model.generate(prompt(40))
        assert rig.clock.sleeps == []
        rig.model.generate(prompt(40))

        assert rig.clock.slept == pytest.approx(60, abs=0.1)

    def test_a_reply_without_usage_is_booked_at_the_worst_case(self, build) -> None:
        rig = build({PRIMARY: [reply(usage=False), reply()]})

        rig.model.generate(prompt(40))  # nothing reported: booked as 12 tokens plus the 2000-token cap
        rig.model.generate(prompt(19_700))  # 5629 more would not fit: it waits for the first call to age out

        assert rig.clock.slept == pytest.approx(60, abs=0.1)


class TestRateLimits:
    def test_the_retry_after_header_is_respected(self, build) -> None:
        rig = build({PRIMARY: [rate_limit_error("slow down", retry_after="7"), reply("done")]})

        assert rig.model.generate(prompt(40)).content == "done"
        assert rig.clock.sleeps == [7.0]
        assert len(rig.fakes[PRIMARY].calls) == 2

    def test_the_time_in_the_message_is_used_when_there_is_no_header(self, build) -> None:
        rig = build({PRIMARY: [rate_limit_error(), reply()]})  # the message says "try again in 3.0s"

        rig.model.generate(prompt(40))

        assert rig.clock.sleeps == [3.0]

    def test_without_any_hint_it_backs_off_exponentially(self, build) -> None:
        rig = build({PRIMARY: [rate_limit_error(SLOW), rate_limit_error(SLOW), reply()]})

        rig.model.generate(prompt(40))

        assert rig.clock.sleeps == [2.0, 4.0]

    def test_it_gives_up_after_five_attempts_and_raises_the_provider_error(self, build) -> None:
        rig = build({PRIMARY: [rate_limit_error(SLOW) for _ in range(5)]})

        with pytest.raises(litellm.RateLimitError):
            rig.model.generate(prompt(40))

        assert rig.clock.sleeps == [2.0, 4.0, 8.0, 16.0]
        assert len(rig.fakes[PRIMARY].calls) == 5

    def test_a_wait_of_minutes_counts_as_an_exhausted_model(self, build) -> None:
        long_wait = rate_limit_error(f"{SLOW}. Please try again in 3m0s.")
        rig = build({PRIMARY: [long_wait], SECOND: [reply("from second")]})

        assert rig.model.generate(prompt(40)).content == "from second"
        assert rig.clock.sleeps == []

    def test_a_server_error_is_retried_like_a_rate_limit(self, build) -> None:
        rig = build({PRIMARY: [HttpError(503, "service unavailable"), reply("done")]})

        assert rig.model.generate(prompt(40)).content == "done"
        assert rig.clock.sleeps == [2.0]

    def test_a_server_error_that_keeps_coming_is_raised_after_five_attempts(self, build) -> None:
        rig = build({PRIMARY: [HttpError(500, "boom") for _ in range(5)]})

        with pytest.raises(HttpError):
            rig.model.generate(prompt(40))

        assert rig.clock.sleeps == [2.0, 4.0, 8.0, 16.0]
        assert len(rig.fakes[PRIMARY].calls) == 5


class TestQuotaAndFallback:
    def test_an_exhausted_daily_quota_moves_on_to_the_next_model_at_once(self, build) -> None:
        rig = build({PRIMARY: [rate_limit_error(TPD_MESSAGE, retry_after="390")], SECOND: [reply("from second")]})

        assert rig.model.generate(prompt(40)).content == "from second"
        assert rig.clock.sleeps == []
        assert rig.model.model_id == SECOND

    def test_a_daily_quota_is_recognised_even_when_the_provider_asks_for_a_short_wait(self, build) -> None:
        """A wait of 30 s looks like a per-minute limit, but the text says per day: the model is out for the run."""
        message = TPD_MESSAGE.replace("Please try again in 6m30s.", "Please try again in 30s.")
        rig = build({PRIMARY: [rate_limit_error(message)], SECOND: [reply("from second")]})

        assert rig.model.generate(prompt(40)).content == "from second"
        assert rig.clock.sleeps == []  # it did not wait for the quota to come back

    def test_a_model_without_quota_is_not_tried_again(self, build) -> None:
        rig = build({PRIMARY: [rate_limit_error(TPD_MESSAGE)], SECOND: [reply(), reply()]})

        rig.model.generate(prompt(40))
        rig.model.generate(prompt(40))

        assert len(rig.fakes[PRIMARY].calls) == 1
        assert len(rig.fakes[SECOND].calls) == 2

    def test_when_every_model_is_exhausted_the_error_says_why_and_stays_cheap(self, build) -> None:
        rig = build({model_id: [rate_limit_error(TPD_MESSAGE)] for model_id in (PRIMARY, SECOND, THIRD)})

        with pytest.raises(NoModelAvailableError) as first:
            rig.model.generate(prompt(40))
        with pytest.raises(NoModelAvailableError):
            rig.model.generate(prompt(40))

        assert all(model_id in str(first.value) for model_id in (PRIMARY, SECOND, THIRD))
        assert all(len(fake.calls) == 1 for fake in rig.fakes.values())
