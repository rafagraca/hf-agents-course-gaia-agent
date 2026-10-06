"""Budgeted model tests that go beyond scripted fakes: model selection, the usage log, secrets, threads, a real
``CodeAgent`` and the real litellm stack (stubbed at ``completion`` or talking to a loopback server)."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
from budget_fakes import (
    GEMINI_SECRET,
    GROQ_SECRET,
    PRIMARY,
    SECOND,
    THIRD,
    TOO_LARGE_MESSAGE,
    TPD_MESSAGE,
    FakeClock,
    FakeModel,
    GroqStub,
    bad_generation_error,
    chat_reply,
    completion_response,
    error_reply,
    litellm,
    make_rig,
    prompt,
    provide_keys,
    rate_limit_error,
    refused_generation_reply,
    reply,
    stub_litellm_completion,
)
from smolagents import CodeAgent
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import LogLevel

from gaia_agent.budget import (
    REQUEST_TIMEOUT_SECONDS,
    BudgetedModel,
    NoModelAvailableError,
    PromptTooLargeError,
)

GEMINI = "gemini/gemini-2.5-flash"
LOCAL = "ollama/llama3"
SLOW = "Rate limit reached"


@pytest.fixture
def build(settings, monkeypatch: pytest.MonkeyPatch):
    """``build({model_id: [replies or errors]}, ...)``: a budgeted model on scripted fake models."""
    provide_keys(monkeypatch)
    return functools.partial(make_rig, settings)


class TestModelSelection:
    def test_the_models_default_to_the_primary_then_the_fallbacks(self, settings) -> None:
        config = dataclasses.replace(settings, model_id=THIRD, fallback_model_ids=(PRIMARY, SECOND))

        model = BudgetedModel(config)

        assert model.model_ids == (THIRD, PRIMARY, SECOND)
        assert model.model_id == THIRD

    def test_explicit_model_ids_replace_the_defaults_and_drop_duplicates(self, build) -> None:
        assert build({}, model_ids=[THIRD, THIRD, PRIMARY]).model.model_ids == (THIRD, PRIMARY)

    def test_at_least_one_model_id_is_required(self, build) -> None:
        with pytest.raises(ValueError, match="model id"):
            build({}, model_ids=[])

    def test_no_model_is_built_before_the_first_call(self, build) -> None:
        assert build({PRIMARY: [reply()]}).factory.built == []

    def test_each_provider_gets_its_own_key_and_other_providers_none(self, build) -> None:
        scripts = {PRIMARY: [rate_limit_error(TPD_MESSAGE)], GEMINI: [rate_limit_error(TPD_MESSAGE)], LOCAL: [reply()]}
        rig = build(scripts, model_ids=[PRIMARY, GEMINI, LOCAL])

        rig.model.generate(prompt(40))

        assert rig.factory.built == [(PRIMARY, GROQ_SECRET), (GEMINI, GEMINI_SECRET), (LOCAL, None)]

    def test_a_model_without_its_key_is_skipped(self, build, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY")
        rig = build({GEMINI: [reply("from gemini")]}, model_ids=[PRIMARY, GEMINI])

        assert rig.model.generate(prompt(40)).content == "from gemini"
        assert rig.factory.built == [(GEMINI, GEMINI_SECRET)]

    def test_without_any_usable_key_the_error_names_the_missing_variable(
        self, build, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GROQ_API_KEY")
        rig = build({}, model_ids=[PRIMARY, SECOND])

        with pytest.raises(NoModelAvailableError, match="GROQ_API_KEY"):
            rig.model.generate(prompt(40))

    def test_the_default_clock_and_sleep_are_looked_up_when_needed(
        self, settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provide_keys(monkeypatch)
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", slept.append)
        fake = FakeModel(PRIMARY, [rate_limit_error("slow", retry_after="7"), reply()])
        model = BudgetedModel(settings, model_ids=[PRIMARY], model_factory=lambda *_: fake)

        model.generate(prompt(40))

        assert slept == [7.0]


class TestUsageLog:
    def test_each_call_appends_one_json_line(self, build, tmp_path: Path) -> None:
        log = tmp_path / "runs" / "run-1" / "usage.jsonl"  # the folders do not exist yet
        script = [rate_limit_error("slow", retry_after="2"), reply("a", 120, 30), reply("b", 10, 5)]
        rig = build({PRIMARY: script}, usage_log=log, latency=1.5)

        rig.model.generate(prompt(350))
        rig.model.generate(prompt(70))

        first, second = (json.loads(line) for line in log.read_text(encoding="utf-8").splitlines())
        assert first["model"] == PRIMARY
        assert (first["input_tokens"], first["output_tokens"]) == (120, 30)
        assert (first["estimated_input_tokens"], first["max_tokens"]) == (100, 2000)
        assert first["seconds"] == pytest.approx(1.5)
        assert (first["attempts"], first["waited_seconds"]) == (2, pytest.approx(2.0))
        assert (second["input_tokens"], second["output_tokens"], second["attempts"]) == (10, 5, 1)
        datetime.fromisoformat(first["ts"])

    def test_lines_end_with_a_plain_newline_on_every_platform(self, build, tmp_path: Path) -> None:
        log = tmp_path / "usage.jsonl"

        build({PRIMARY: [reply()]}, usage_log=log).model.generate(prompt(40))

        assert b"\r" not in log.read_bytes()

    def test_a_reply_without_usage_is_logged_with_unknown_counts(self, build, tmp_path: Path) -> None:
        log = tmp_path / "usage.jsonl"
        rig = build({PRIMARY: [reply(usage=False)]}, usage_log=log)

        rig.model.generate(prompt(40))

        line = json.loads(log.read_text(encoding="utf-8"))
        assert (line["input_tokens"], line["output_tokens"]) == (None, None)

    def test_without_a_log_path_nothing_is_written(self, build, tmp_path: Path) -> None:
        build({PRIMARY: [reply()]}, usage_log=None).model.generate(prompt(40))

        assert list(tmp_path.rglob("*.jsonl")) == []

    def test_a_log_that_cannot_be_written_warns_but_keeps_the_reply(
        self, build, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        rig = build({PRIMARY: [reply("kept")]}, usage_log=tmp_path)  # a folder, not a file

        with caplog.at_level(logging.WARNING, logger="gaia_agent.usage_log"):
            result = rig.model.generate(prompt(40))

        assert result.content == "kept"
        assert "usage log" in caplog.text


class TestSecrets:
    def test_keys_never_reach_the_logs_the_errors_or_the_usage_file(
        self, build, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        log = tmp_path / "usage.jsonl"
        caplog.set_level(logging.DEBUG)
        script = {
            PRIMARY: [bad_generation_error(), rate_limit_error("slow", retry_after="2"), rate_limit_error(TPD_MESSAGE)],
            SECOND: [reply()],
        }
        build(script, usage_log=log).model.generate(prompt(40))
        exhausted = build({model_id: [rate_limit_error(TPD_MESSAGE)] for model_id in (PRIMARY, SECOND, THIRD)})
        rejected = build({PRIMARY: [rate_limit_error(TOO_LARGE_MESSAGE)]})
        oversized = build({PRIMARY: []})

        with pytest.raises(NoModelAvailableError) as no_model:
            exhausted.model.generate(prompt(40))
        with pytest.raises(PromptTooLargeError) as too_large_for_the_api:
            rejected.model.generate(prompt(40))
        with pytest.raises(PromptTooLargeError) as too_large_for_the_minute:
            oversized.model.generate(prompt(27_000))

        errors = (no_model, too_large_for_the_api, too_large_for_the_minute)
        everything = caplog.text + log.read_text(encoding="utf-8") + "\n".join(str(error.value) for error in errors)
        assert GROQ_SECRET not in everything
        assert GEMINI_SECRET not in everything


class TestConcurrency:
    def test_concurrent_calls_are_serialised(self, build) -> None:
        rig = build({PRIMARY: [reply() for _ in range(12)]})
        fake, state = rig.fakes[PRIMARY], {"active": 0, "peak": 0}
        guard, original = threading.Lock(), rig.fakes[PRIMARY].generate

        def watched(*args, **kwargs):
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.005)
            try:
                return original(*args, **kwargs)
            finally:
                with guard:
                    state["active"] -= 1

        fake.generate = watched
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(3):
                    rig.model.generate(prompt(40))
            except BaseException as exc:  # reported by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert state["peak"] == 1


class TestSmolagentsAgent:
    def test_a_code_agent_runs_on_top_of_the_budgeted_model(self, build) -> None:
        script = [
            reply("Thought: add\n<code>\nprint(21 * 2)\n</code>"),
            reply("Thought: done\n<code>\nfinal_answer(42)\n</code>"),
        ]
        rig = build({PRIMARY: script})
        agent = CodeAgent(tools=[], model=rig.model, max_steps=3, verbosity_level=LogLevel.OFF)

        assert agent.run("Compute 21 * 2.") == 42
        assert len(rig.fakes[PRIMARY].calls) == 2
        assert agent.monitor.total_input_token_count == 200


class TestLiteLLMWithStubbedCompletion:
    def test_the_default_factory_sends_the_budgeted_arguments_to_litellm(
        self, settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", GROQ_SECRET)
        calls = stub_litellm_completion(monkeypatch, lambda _number, _kwargs: completion_response("hi", 11, 7))
        model = BudgetedModel(settings, model_ids=[PRIMARY])
        message = ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])

        result = model.generate([message], stop_sequences=["Observation:"])

        (sent,) = calls
        assert result.content == "hi"
        assert (result.token_usage.input_tokens, result.token_usage.output_tokens) == (11, 7)
        assert (sent["model"], sent["api_key"]) == (PRIMARY, GROQ_SECRET)
        assert sent["max_tokens"] == settings.max_output_tokens
        assert sent["reasoning_effort"] == settings.reasoning_effort
        assert sent["timeout"] == REQUEST_TIMEOUT_SECONDS
        assert sent["stop"] == ["Observation:"]
        assert sent["messages"] == [{"role": "user", "content": "Hello"}]
        assert litellm.suppress_debug_info is True

    def test_litellm_is_not_left_to_retry_on_its_own(self, settings, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", GROQ_SECRET)

        def always_limited(_number, _kwargs):
            raise rate_limit_error(SLOW)

        calls = stub_litellm_completion(monkeypatch, always_limited)
        clock = FakeClock()
        model = BudgetedModel(settings, model_ids=[PRIMARY], clock=clock, sleep=clock.sleep)

        def forbidden(_seconds):
            raise AssertionError("a real sleep: the inner model retried by itself")

        monkeypatch.setattr(time, "sleep", forbidden)

        with pytest.raises(litellm.RateLimitError):
            model.generate(prompt(40))

        assert len(calls) == 5
        assert clock.sleeps == [2.0, 4.0, 8.0, 16.0]


@pytest.mark.filterwarnings("ignore:Item 'summary' on TypedDict:UserWarning")  # pydantic, inside litellm
class TestLiteLLMOverHttp:
    """Real litellm and OpenAI client against a loopback server that answers like Groq."""

    @pytest.fixture(autouse=True)
    def _groq_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", GROQ_SECRET)
        for name in ("NO_PROXY", "no_proxy"):  # a proxy configured on the machine must not capture the loopback calls
            monkeypatch.setenv(name, "127.0.0.1")

    def make_model(self, settings, stub: GroqStub, monkeypatch: pytest.MonkeyPatch, model_ids=(PRIMARY,)):
        monkeypatch.setenv("GROQ_API_BASE", stub.url)
        clock = FakeClock()
        return BudgetedModel(settings, model_ids=model_ids, clock=clock, sleep=clock.sleep), clock

    def test_a_429_is_retried_after_the_advertised_wait(self, settings, monkeypatch: pytest.MonkeyPatch) -> None:
        replies = [error_reply(429, "Rate limit reached. Please try again in 2s.", retry_after="2"), chat_reply("ok")]
        with GroqStub(replies) as stub:
            model, clock = self.make_model(settings, stub, monkeypatch)

            result = model.generate(prompt(40), stop_sequences=["Observation:"])

        assert result.content == "ok"
        assert clock.sleeps == [2.0]
        assert len(stub.requests) == 2  # one HTTP request per attempt: nothing retries behind our back
        body = stub.requests[-1]
        assert body["model"] == "openai/gpt-oss-120b"
        assert (body["max_tokens"], body["reasoning_effort"]) == (settings.max_output_tokens, settings.reasoning_effort)
        assert body["stop"] == ["Observation:"]

    def test_a_503_is_retried_after_a_backoff(self, settings, monkeypatch: pytest.MonkeyPatch) -> None:
        replies = [error_reply(503, "The service is temporarily unavailable."), chat_reply("back again")]
        with GroqStub(replies) as stub:
            model, clock = self.make_model(settings, stub, monkeypatch)

            result = model.generate(prompt(40))

        assert result.content == "back again"
        assert clock.sleeps == [2.0]
        assert len(stub.requests) == 2

    def test_a_refused_generation_is_retried_at_once(self, settings, monkeypatch: pytest.MonkeyPatch) -> None:
        with GroqStub([refused_generation_reply(), chat_reply("recovered")]) as stub:
            model, clock = self.make_model(settings, stub, monkeypatch)

            result = model.generate(prompt(40))

        assert result.content == "recovered"
        assert clock.sleeps == []
        assert len(stub.requests) == 2

    def test_a_413_becomes_prompt_too_large_after_a_single_request(
        self, settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = "Request too large for model `x` on tokens per minute (TPM): Limit 8000, Requested 9071"
        with GroqStub([error_reply(413, message)]) as stub:
            model, clock = self.make_model(settings, stub, monkeypatch)

            with pytest.raises(PromptTooLargeError, match="9071"):
                model.generate(prompt(40))

        assert len(stub.requests) == 1
        assert clock.sleeps == []

    def test_a_daily_quota_moves_on_to_the_next_model(self, settings, monkeypatch: pytest.MonkeyPatch) -> None:
        replies = [error_reply(429, TPD_MESSAGE, retry_after="390"), chat_reply("from second")]
        with GroqStub(replies) as stub:
            model, clock = self.make_model(settings, stub, monkeypatch, model_ids=(PRIMARY, SECOND))

            result = model.generate(prompt(40))

        assert result.content == "from second"
        assert [request["model"] for request in stub.requests] == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        assert clock.sleeps == []
