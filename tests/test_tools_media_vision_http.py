"""Tests for ``describe_image``: HTTP failures, odd replies and which media tools are offered (no network)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import requests
import responses
from vision_fakes import FAKE_KEY, completion, make_image

from gaia_agent import config
from gaia_agent.config import Settings
from gaia_agent.tools import _groq_vision
from gaia_agent.tools._groq_vision import CHAT_URL
from gaia_agent.tools.media import AskAboutVideoTool, DescribeImageTool, optional_media_tools

REAL_PAUSE = _groq_vision._pause  # captured before the autouse fixture replaces it


@pytest.fixture
def http():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocked:
        yield mocked


@pytest.fixture
def attachments(settings: Settings) -> Path:
    return settings.files_dir


@pytest.fixture(autouse=True)
def groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)


@pytest.fixture(autouse=True)
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the waits instead of sleeping."""
    waited: list[float] = []
    monkeypatch.setattr(_groq_vision, "_pause", waited.append)
    return waited


@pytest.fixture
def describe(settings: Settings) -> DescribeImageTool:
    return DescribeImageTool(settings)


# --------------------------------------------------------------------------- HTTP failures


def test_a_429_is_retried_once_after_the_advertised_wait(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, pauses: list[float]
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, status=429, headers={"retry-after": "12"}, json={"error": {"message": "slow"}})
    http.add(responses.POST, CHAT_URL, json=completion("after the wait"))

    assert describe(name, "what?") == "after the wait"

    assert pauses == [12.0]
    assert len(http.calls) == 2


def test_a_second_429_is_reported_without_a_third_try(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, pauses: list[float]
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, status=429, headers={"retry-after": "3"}, json={"error": {"message": "slow"}})

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "429" in result
    assert pauses == [3.0]
    assert len(http.calls) == 2


def test_a_429_asking_for_more_than_a_minute_is_not_waited_for(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, pauses: list[float]
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, status=429, headers={"retry-after": "390"}, json={"error": {"message": "tpd"}})

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "390" in result
    assert pauses == []
    assert len(http.calls) == 1


def test_a_429_without_a_usable_wait_hint_uses_a_short_default(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, pauses: list[float]
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, status=429, json={"error": {"message": "slow"}})
    http.add(responses.POST, CHAT_URL, json=completion("ok"))

    assert describe(name, "what?") == "ok"
    assert pauses == [_groq_vision.DEFAULT_RETRY_WAIT_S]


def test_a_413_says_the_request_is_too_large(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, pauses: list[float]
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, status=413, json={"error": {"message": "Request too large"}})

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "too large" in result
    assert "smaller" in result
    assert pauses == []
    assert len(http.calls) == 1


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_reported_without_echoing_it(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, status: int
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, status=status, json={"error": {"message": f"bad key {FAKE_KEY}"}})

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "GROQ_API_KEY" in result
    assert FAKE_KEY not in result


def test_another_error_status_relays_the_message_with_the_key_redacted(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, status=400, json={"error": {"message": f"Bad image near {FAKE_KEY}"}})

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "HTTP 400" in result
    assert "Bad image near [redacted]" in result
    assert FAKE_KEY not in result


def test_a_timeout_and_a_connection_failure_become_short_errors(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, body=requests.Timeout("slow"))
    assert "timed out" in describe(name, "what?")

    http.reset()
    http.add(responses.POST, CHAT_URL, body=requests.ConnectionError(f"boom {FAKE_KEY}"))
    result = describe(name, "what?")
    assert "could not reach" in result
    assert FAKE_KEY not in result


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": ["nope"]},
        {},
    ],
)
def test_a_reply_without_text_is_reported(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, body: dict
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, json=body)

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "no text" in result


@pytest.mark.parametrize("body", ["not json at all", "[1, 2, 3]"])
def test_an_unreadable_reply_is_reported(
    describe: DescribeImageTool, attachments: Path, http: responses.RequestsMock, body: str
) -> None:
    name = make_image(attachments)
    http.add(responses.POST, CHAT_URL, body=body)

    result = describe(name, "what?")

    assert result.startswith("Error:")
    assert "unreadable" in result


def test_the_real_pause_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)

    REAL_PAUSE(1.5)

    assert slept == [1.5]


# --------------------------------------------------------------------------- optional_media_tools


def test_describe_image_is_offered_with_a_groq_key_alone(settings: Settings) -> None:
    tools = optional_media_tools(settings)

    assert [tool.name for tool in tools] == ["describe_image"]
    assert isinstance(tools[0], DescribeImageTool)


def test_nothing_is_offered_without_any_key(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY")

    assert optional_media_tools(settings) == []


def test_a_blank_groq_key_counts_as_missing(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    assert optional_media_tools(settings) == []


def test_the_video_tool_needs_a_gemini_key_and_comes_after_describe_image(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-tests")

    tools = optional_media_tools(settings)

    assert [tool.name for tool in tools] == ["describe_image", "ask_about_video"]
    assert isinstance(tools[1], AskAboutVideoTool)


def test_a_gemini_key_alone_offers_only_the_video_tool(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-tests")

    assert [tool.name for tool in optional_media_tools(settings)] == ["ask_about_video"]


def test_a_groq_key_saved_in_the_windows_user_environment_is_found(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setattr(config, "_persistent_env_reader", lambda name: FAKE_KEY if name == "GROQ_API_KEY" else None)

    assert [tool.name for tool in optional_media_tools(settings)] == ["describe_image"]
