"""Tests for ``deep_search``: Groq browser search over HTTP faked with ``responses`` (no network).

The reply bodies are invented. Nothing here depends on a real question of the benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests
import responses

from gaia_agent.config import Settings
from gaia_agent.tools import default_tools
from gaia_agent.tools._groq_search import SearchFailure, build_payload
from gaia_agent.tools.deep_search import ENV_DEEP_SEARCH_MAX, DeepSearchTool

URL = "https://api.groq.com/openai/v1/chat/completions"
FAKE_KEY = "fake-groq-key-for-tests"
SOURCE_A = "https://example.org/alpha"
SOURCE_B = "https://example.org/beta"
UNAVAILABLE_TAIL = "Use web_search and read_webpage instead."


@pytest.fixture
def http():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocked:
        yield mocked


@pytest.fixture(autouse=True)
def groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)


class Sleeper:
    """An injectable ``sleep`` that only records what it was asked to wait."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleeper() -> Sleeper:
    return Sleeper()


@pytest.fixture
def tool(sleeper: Sleeper) -> DeepSearchTool:
    return DeepSearchTool(sleep=sleeper)


def reply(
    content: str | None = "An invented answer.",
    *,
    results: tuple[str, ...] = (),
    annotations: tuple[str, ...] = (),
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if results:
        message["executed_tools"] = [
            {"type": "search", "search_results": {"results": [{"title": "t", "url": url} for url in results]}}
        ]
    if annotations:
        message["annotations"] = [
            {"type": "document_citation", "document_citation": {"url": url}} for url in annotations
        ]
    body: dict[str, Any] = {"choices": [{"message": message}]}
    body["usage"] = usage or {"prompt_tokens": 1200, "completion_tokens": 300}
    return body


def rate_limit_body(message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}


MINUTE_MESSAGE = "Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): Limit 8000, Used 7000"
DAILY_MESSAGE = "Rate limit reached for model `openai/gpt-oss-20b` on tokens per day (TPD): Limit 200000, Used 199000"


def sent_json(http: responses.RequestsMock, index: int = 0) -> dict[str, Any]:
    return json.loads(http.calls[index].request.body)


def test_the_request_uses_the_search_model_with_the_browser_tool(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply())

    tool("some query")

    payload = sent_json(http)
    assert payload["model"] == "openai/gpt-oss-20b"
    assert payload["tools"] == [{"type": "browser_search"}]
    assert payload["reasoning_effort"] == "low"
    assert payload["max_completion_tokens"] == 2048
    assert payload["temperature"] <= 0.3
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["messages"][1]["content"] == "some query"


def test_the_system_message_asks_for_a_short_sourced_answer_or_an_admission() -> None:
    system = build_payload("q")["messages"][0]["content"].lower()

    assert "concise" in system
    assert "url" in system
    assert "don't know" in system or "do not know" in system


def test_the_request_carries_the_key_in_the_authorization_header(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply())

    tool("q")

    assert http.calls[0].request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert http.calls[0].request.req_kwargs["timeout"] == 120


def test_the_answer_comes_back_with_the_urls_of_the_executed_search(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply("The invented answer.", results=(SOURCE_A, SOURCE_B)))

    result = tool("q")

    assert result.startswith("The invented answer.")
    assert SOURCE_A in result
    assert SOURCE_B in result


def test_at_most_five_distinct_urls_are_listed(http: responses.RequestsMock, tool: DeepSearchTool):
    urls = tuple(f"https://example.org/page{i}" for i in range(8))
    http.add(responses.POST, URL, json=reply(results=(*urls, urls[0])))

    result = tool("q")

    assert [url for url in urls if url in result] == list(urls[:5])
    assert result.count(urls[0]) == 1


def test_annotation_urls_are_used_too(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply(annotations=(SOURCE_B,)))

    assert SOURCE_B in tool("q")


def test_a_url_quoted_in_the_answer_text_is_listed(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply(f"See {SOURCE_A}, which says so."))

    assert tool("q").count(SOURCE_A) >= 1


def test_a_reply_without_any_source_is_still_returned(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply("I don't know."))

    assert tool("q") == "I don't know."


def test_non_http_links_in_the_tool_results_are_ignored(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply(results=("javascript:alert(1)", "file:///etc/passwd", SOURCE_A)))

    result = tool("q")

    assert SOURCE_A in result
    assert "javascript" not in result
    assert "file:" not in result


def test_a_long_answer_is_cut_to_the_limit_and_keeps_its_sources(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply("word " * 2000, results=(SOURCE_A,)))

    result = tool("q")

    assert len(result) <= 1500
    assert SOURCE_A in result
    assert "truncated" in result


def test_a_reply_without_content_is_reported_as_unavailable(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply(None))

    result = tool("q")

    assert result.startswith("deep_search unavailable:")
    assert result.endswith(UNAVAILABLE_TAIL)


def test_a_body_that_is_not_json_is_reported_as_unavailable(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, body="<html>oops</html>", status=200)

    assert tool("q").startswith("deep_search unavailable:")


def test_a_minute_rate_limit_waits_the_retry_after_and_tries_once_more(
    http: responses.RequestsMock, tool: DeepSearchTool, sleeper: Sleeper
):
    http.add(responses.POST, URL, status=429, json=rate_limit_body(MINUTE_MESSAGE), headers={"retry-after": "7"})
    http.add(responses.POST, URL, json=reply("Second try worked."))

    result = tool("q")

    assert result.startswith("Second try worked.")
    assert sleeper.calls == [7.0]
    assert len(http.calls) == 2


def test_the_wait_can_come_from_the_error_text(http: responses.RequestsMock, tool: DeepSearchTool, sleeper: Sleeper):
    message = f"{MINUTE_MESSAGE}. Please try again in 12.5s."
    http.add(responses.POST, URL, status=429, json=rate_limit_body(message))
    http.add(responses.POST, URL, json=reply())

    tool("q")

    assert sleeper.calls == [12.5]


def test_only_one_retry_is_made_after_a_minute_rate_limit(
    http: responses.RequestsMock, tool: DeepSearchTool, sleeper: Sleeper
):
    http.add(responses.POST, URL, status=429, json=rate_limit_body(MINUTE_MESSAGE), headers={"retry-after": "3"})

    result = tool("q")

    assert len(http.calls) == 2
    assert sleeper.calls == [3.0]
    assert result.startswith("deep_search unavailable:")
    assert result.endswith(UNAVAILABLE_TAIL)


def test_a_wait_longer_than_ninety_seconds_is_not_taken(
    http: responses.RequestsMock, tool: DeepSearchTool, sleeper: Sleeper
):
    http.add(responses.POST, URL, status=429, json=rate_limit_body(MINUTE_MESSAGE), headers={"retry-after": "300"})

    result = tool("q")

    assert sleeper.calls == []
    assert len(http.calls) == 1
    assert result.startswith("deep_search unavailable:")
    assert "300" in result


def test_a_wait_of_exactly_ninety_seconds_is_taken(
    http: responses.RequestsMock, tool: DeepSearchTool, sleeper: Sleeper
):
    http.add(responses.POST, URL, status=429, json=rate_limit_body(MINUTE_MESSAGE), headers={"retry-after": "90"})
    http.add(responses.POST, URL, json=reply())

    tool("q")

    assert sleeper.calls == [90.0]


def test_a_rate_limit_without_a_hint_waits_a_default_while(
    http: responses.RequestsMock, tool: DeepSearchTool, sleeper: Sleeper
):
    http.add(responses.POST, URL, status=429, json=rate_limit_body(MINUTE_MESSAGE))
    http.add(responses.POST, URL, json=reply())

    tool("q")

    assert len(sleeper.calls) == 1
    assert 0 < sleeper.calls[0] <= 90


def test_the_daily_quota_is_reported_without_waiting_or_retrying(
    http: responses.RequestsMock, tool: DeepSearchTool, sleeper: Sleeper
):
    http.add(responses.POST, URL, status=429, json=rate_limit_body(DAILY_MESSAGE), headers={"retry-after": "5000"})

    result = tool("q")

    assert result.startswith("deep_search unavailable:")
    assert "daily" in result.lower()
    assert result.endswith(UNAVAILABLE_TAIL)
    assert sleeper.calls == []
    assert len(http.calls) == 1


def test_after_the_daily_quota_the_network_is_not_tried_again(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, status=429, json=rate_limit_body(DAILY_MESSAGE))

    tool("first")
    tool.reset_question()
    second = tool("second")

    assert len(http.calls) == 1
    assert second.startswith("deep_search unavailable:")


@pytest.mark.parametrize(
    ("status", "fragment"),
    [(401, "API key"), (403, "API key"), (413, "too large"), (500, "HTTP 500"), (400, "HTTP 400")],
)
def test_other_http_errors_are_reported_briefly_and_never_raise(
    http: responses.RequestsMock, tool: DeepSearchTool, status: int, fragment: str
):
    http.add(responses.POST, URL, status=status, json={"error": {"message": "something went wrong"}})

    result = tool("q")

    assert result.startswith("deep_search unavailable:")
    assert fragment in result
    assert result.endswith(UNAVAILABLE_TAIL)


def test_the_key_never_shows_up_in_an_error(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, status=500, json={"error": {"message": f"bad key {FAKE_KEY} was used"}})

    assert FAKE_KEY not in tool("q")


@pytest.mark.parametrize(
    ("exception", "fragment"),
    [(requests.Timeout("slow"), "timed out"), (requests.ConnectionError("down"), "could not reach")],
)
def test_network_failures_are_reported_briefly(
    http: responses.RequestsMock, tool: DeepSearchTool, exception: Exception, fragment: str
):
    http.add(responses.POST, URL, body=exception)

    result = tool("q")

    assert fragment in result
    assert result.startswith("deep_search unavailable:")


def test_without_a_key_nothing_is_sent(http: responses.RequestsMock, tool: DeepSearchTool, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")

    result = tool("q")

    assert len(http.calls) == 0
    assert "GROQ_API_KEY" in result
    assert result.startswith("deep_search unavailable:")


def test_a_blank_query_is_refused_without_a_request(http: responses.RequestsMock, tool: DeepSearchTool):
    result = tool("   ")

    assert len(http.calls) == 0
    assert result.startswith("Error:")


def test_a_question_may_use_the_tool_twice_and_no_more(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply())

    results = [tool(f"q{i}") for i in range(3)]

    assert len(http.calls) == 2
    assert results[2].startswith("deep_search unavailable:")
    assert "question" in results[2]


def test_reset_question_gives_the_next_question_its_two_uses(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply())
    tool("a")
    tool("b")

    tool.reset_question()
    tool("c")

    assert len(http.calls) == 3


def test_the_run_limit_defaults_to_five_and_survives_reset_question(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, json=reply())

    for _ in range(3):
        tool.reset_question()
        tool("a")
        tool("b")
    tool.reset_question()
    refused = tool("one more")

    assert len(http.calls) == 5
    assert refused.startswith("deep_search unavailable:")
    assert "run" in refused


def test_the_run_limit_can_be_set_by_the_environment(
    http: responses.RequestsMock, sleeper: Sleeper, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ENV_DEEP_SEARCH_MAX, "1")
    limited = DeepSearchTool(sleep=sleeper)
    http.add(responses.POST, URL, json=reply())

    limited("a")
    limited.reset_question()
    refused = limited("b")

    assert len(http.calls) == 1
    assert refused.startswith("deep_search unavailable:")


@pytest.mark.parametrize("raw", ["", "abc", "-3", "0.5"])
def test_an_unusable_limit_in_the_environment_means_five(
    http: responses.RequestsMock, sleeper: Sleeper, monkeypatch: pytest.MonkeyPatch, raw: str
):
    monkeypatch.setenv(ENV_DEEP_SEARCH_MAX, raw)
    http.add(responses.POST, URL, json=reply())
    limited = DeepSearchTool(sleep=sleeper)

    for _ in range(6):
        limited.reset_question()
        limited("a")
        limited("b")

    assert len(http.calls) == 5


def test_a_limit_of_zero_switches_the_tool_off(http: responses.RequestsMock, sleeper: Sleeper, monkeypatch):
    monkeypatch.setenv(ENV_DEEP_SEARCH_MAX, "0")

    result = DeepSearchTool(sleep=sleeper)("a")

    assert len(http.calls) == 0
    assert result.startswith("deep_search unavailable:")


def test_a_call_that_fails_still_counts_against_the_limits(http: responses.RequestsMock, tool: DeepSearchTool):
    http.add(responses.POST, URL, status=500, json={"error": {"message": "boom"}})

    tool("a")
    tool("b")
    third = tool("c")

    assert len(http.calls) == 2
    assert "question" in third


def test_the_consumption_is_appended_to_the_usage_log(http: responses.RequestsMock, sleeper: Sleeper, tmp_path: Path):
    log = tmp_path / "run" / "usage.jsonl"
    logged = DeepSearchTool(usage_log=log, sleep=sleeper)
    http.add(responses.POST, URL, json=reply(usage={"prompt_tokens": 19000, "completion_tokens": 700}))

    logged("q")

    (line,) = [json.loads(text) for text in log.read_text(encoding="utf-8").splitlines()]
    assert line["model"] == "openai/gpt-oss-20b (deep_search)"
    assert (line["input_tokens"], line["output_tokens"]) == (19000, 700)
    assert line["max_tokens"] == 2048
    assert FAKE_KEY not in log.read_text(encoding="utf-8")


def test_the_usage_log_can_be_set_after_the_tool_is_built(
    http: responses.RequestsMock, tool: DeepSearchTool, tmp_path: Path
):
    log = tmp_path / "usage.jsonl"
    http.add(responses.POST, URL, json=reply())

    tool.set_usage_log(log)
    tool("q")

    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_a_reply_without_a_usage_block_is_logged_with_unknown_tokens(
    http: responses.RequestsMock, sleeper: Sleeper, tmp_path: Path
):
    log = tmp_path / "usage.jsonl"
    body = reply()
    del body["usage"]
    http.add(responses.POST, URL, json=body)

    DeepSearchTool(usage_log=log, sleep=sleeper)("q")

    line = json.loads(log.read_text(encoding="utf-8"))
    assert line["input_tokens"] is None
    assert line["output_tokens"] is None


def test_without_a_usage_path_nothing_is_written(http: responses.RequestsMock, tool: DeepSearchTool, tmp_path: Path):
    http.add(responses.POST, URL, json=reply())

    tool("q")

    assert list(tmp_path.iterdir()) == []


def test_the_tool_is_offered_only_when_a_groq_key_exists(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    with_key = [tool.name for tool in default_tools(settings)]
    monkeypatch.delenv("GROQ_API_KEY")
    without_key = [tool.name for tool in default_tools(settings)]

    assert "deep_search" in with_key
    assert "deep_search" not in without_key


def test_the_registered_tool_is_a_deep_search_tool_with_its_own_counters(settings: Settings) -> None:
    first = next(t for t in default_tools(settings) if t.name == "deep_search")
    second = next(t for t in default_tools(settings) if t.name == "deep_search")

    assert isinstance(first, DeepSearchTool)
    assert first is not second


def test_the_tool_takes_one_text_input_and_returns_text() -> None:
    assert list(DeepSearchTool.inputs) == ["query"]
    assert DeepSearchTool.output_type == "string"
    assert "read_webpage" in DeepSearchTool.description
    assert len(DeepSearchTool.description) < 500


def test_search_failure_keeps_a_safe_reason() -> None:
    failure = SearchFailure("rate limited")

    assert str(failure) == "rate limited"
    assert failure.retry_after is None


def test_an_error_body_that_is_not_json_is_used_as_text_without_the_key(
    http: responses.RequestsMock, tool: DeepSearchTool
):
    http.add(responses.POST, URL, status=502, body=f"Bad gateway for {FAKE_KEY}")

    result = tool("q")

    assert "HTTP 502" in result
    assert "Bad gateway" in result
    assert FAKE_KEY not in result
