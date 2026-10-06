"""How the budgeted model reacts to failures that are not rate limits: oversized requests, refused generations and
errors it must not touch (scripted fake models, fake clock, no network)."""

from __future__ import annotations

import functools

import pytest
from budget_fakes import (
    GROQ_SECRET,
    PRIMARY,
    SECOND,
    THIRD,
    TOO_LARGE_MESSAGE,
    HttpError,
    bad_generation_error,
    litellm,
    make_rig,
    prompt,
    provide_keys,
    rate_limit_error,
    reply,
)

from gaia_agent.budget import NoModelAvailableError, PromptTooLargeError


@pytest.fixture
def build(settings, monkeypatch: pytest.MonkeyPatch):
    """``build({model_id: [replies or errors]}, ...)``: a budgeted model on scripted fake models."""
    provide_keys(monkeypatch)
    return functools.partial(make_rig, settings)


class TestOversizedRequests:
    def test_a_413_reported_by_litellm_as_a_rate_limit_is_not_retried(self, build) -> None:
        rig = build({PRIMARY: [rate_limit_error(TOO_LARGE_MESSAGE)], SECOND: [reply()]})

        with pytest.raises(PromptTooLargeError, match="9071"):
            rig.model.generate(prompt(40))

        assert rig.clock.sleeps == []
        assert (len(rig.fakes[PRIMARY].calls), rig.fakes[SECOND].calls) == (1, [])

    @pytest.mark.parametrize(
        "error",
        [
            HttpError(413, "payload too big"),
            litellm.ContextWindowExceededError(message="too long", model="m", llm_provider="groq"),
        ],
        ids=["status 413", "context window"],
    )
    def test_other_oversized_request_errors_raise_prompt_too_large(self, build, error) -> None:
        with pytest.raises(PromptTooLargeError):
            build({PRIMARY: [error]}).model.generate(prompt(40))


class TestRefusedGenerations:
    """Groq answers HTTP 400 ``tool_use_failed`` when gpt-oss replies with a tool call although none was offered."""

    def test_it_is_retried_at_once(self, build) -> None:
        rig = build({PRIMARY: [bad_generation_error(), bad_generation_error(), reply("fine")]})

        assert rig.model.generate(prompt(40)).content == "fine"
        assert len(rig.fakes[PRIMARY].calls) == 3
        assert rig.clock.sleeps == []  # a new sample is a new draw: nothing to wait for

    def test_it_gives_up_after_five_attempts_and_raises_the_provider_error(self, build) -> None:
        rig = build({PRIMARY: [bad_generation_error() for _ in range(5)], SECOND: [reply()]})

        with pytest.raises(litellm.BadRequestError):
            rig.model.generate(prompt(40))

        assert len(rig.fakes[PRIMARY].calls) == 5
        assert rig.fakes[SECOND].calls == []


class TestOtherErrors:
    def test_any_other_error_passes_through_without_retries(self, build) -> None:
        rig = build({PRIMARY: [ValueError("boom")], SECOND: [reply()]})

        with pytest.raises(ValueError, match="boom"):
            rig.model.generate(prompt(40))

        assert rig.clock.sleeps == []
        assert (len(rig.fakes[PRIMARY].calls), rig.fakes[SECOND].calls) == (1, [])


def not_found_error() -> litellm.NotFoundError:
    return litellm.NotFoundError(message="The model does not exist", llm_provider="groq", model="m")


def decommissioned_error() -> litellm.BadRequestError:
    message = '{"error":{"message":"The model has been decommissioned","code":"model_decommissioned"}}'
    return litellm.BadRequestError(message=message, model="m", llm_provider="groq")


def auth_error() -> litellm.AuthenticationError:
    return litellm.AuthenticationError(message=f"Invalid API Key {GROQ_SECRET}", llm_provider="groq", model="m")


class TestModelsThatCannotServe:
    """A fallback model that was retired (or mistyped) must not stop the chain, and a bad key must say so."""

    @pytest.mark.parametrize("error", [not_found_error(), decommissioned_error()], ids=["404", "decommissioned"])
    def test_a_model_that_does_not_exist_hands_over_to_the_next_one(self, build, error) -> None:
        rig = build({PRIMARY: [error], SECOND: [reply("from the second")]})

        assert rig.model.generate(prompt(40)).content == "from the second"
        assert rig.clock.sleeps == []

    def test_a_model_that_does_not_exist_is_not_asked_again_in_the_same_run(self, build) -> None:
        rig = build({PRIMARY: [not_found_error()], SECOND: [reply("one"), reply("two")]})

        rig.model.generate(prompt(40))
        rig.model.generate(prompt(40))

        assert len(rig.fakes[PRIMARY].calls) == 1

    def test_a_rejected_api_key_hands_over_to_the_next_model_too(self, build) -> None:
        rig = build({PRIMARY: [auth_error()], SECOND: [reply("from the second")]})

        assert rig.model.generate(prompt(40)).content == "from the second"

    def test_when_no_model_can_serve_the_reasons_are_all_listed(self, build) -> None:
        rig = build({PRIMARY: [auth_error()], SECOND: [not_found_error()], THIRD: [auth_error()]})

        with pytest.raises(NoModelAvailableError) as caught:
            rig.model.generate(prompt(40))

        message = str(caught.value)
        assert "rejected the API key" in message
        assert "does not offer this model" in message
        assert GROQ_SECRET not in message  # the provider's text, which may echo a key, is never repeated

    def test_the_first_model_that_works_after_a_failure_serves_the_whole_run(self, build) -> None:
        rig = build({PRIMARY: [auth_error()], SECOND: [reply("a"), reply("b")], THIRD: []})

        rig.model.generate(prompt(40))
        rig.model.generate(prompt(40))

        assert (len(rig.fakes[PRIMARY].calls), len(rig.fakes[SECOND].calls), rig.fakes[THIRD].calls) == (1, 2, [])
