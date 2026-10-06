"""Tests for ``ask_about_video`` (Gemini REST, faked, no network).

HTTP is faked with ``responses`` (no network). The fake API key never shows up in a result.
"""

from __future__ import annotations

import json
import time

import pytest
import requests
import responses

from gaia_agent.tools import _gemini
from gaia_agent.tools.media import AskAboutVideoTool

FAKE_KEY = "fake-gemini-key-for-tests"
MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
VIDEO_ID = "abcDEF12345"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
PNG_BYTES = b"\x89PNG\r\n\x1a\n-invented-image-bytes-"


@pytest.fixture
def http():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocked:
        yield mocked


@pytest.fixture(autouse=True)
def gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)


@pytest.fixture(autouse=True)
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the back-off delays instead of sleeping."""
    delays: list[float] = []
    monkeypatch.setattr(_gemini, "_pause", delays.append)
    return delays


def reply(*texts: str, **extra: object) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text} for text in texts]}, "finishReason": "STOP"}], **extra}


def sent_json(http: responses.RequestsMock, index: int = 0) -> dict:
    return json.loads(http.calls[index].request.body)


REAL_PAUSE = _gemini._pause  # captured before the autouse fixture replaces it


def ask(question: str = "what happens?") -> str:
    return AskAboutVideoTool()(WATCH_URL, question)


# --------------------------------------------------------------------------- ask_about_video


def test_ask_about_video_sends_the_youtube_url_as_file_data(http: responses.RequestsMock) -> None:
    http.add(responses.POST, MODEL_URL, json=reply("Three birds appear"))

    result = AskAboutVideoTool()(f"https://youtu.be/{VIDEO_ID}?si=tracking&t=30", "How many birds appear?")

    assert result == "Three birds appear"
    request = http.calls[0].request
    assert request.headers["x-goog-api-key"] == FAKE_KEY
    parts = sent_json(http)["contents"][0]["parts"]
    assert parts[0] == {"file_data": {"file_uri": WATCH_URL}}
    assert "How many birds appear?" in parts[1]["text"]


def test_ask_about_video_rejects_urls_that_are_not_youtube(http: responses.RequestsMock) -> None:
    result = AskAboutVideoTool()("https://example.com/clip.mp4", "what happens?")

    assert result.startswith("Error:")
    assert "YouTube" in result
    assert not http.calls


def test_ask_about_video_needs_a_question(http: responses.RequestsMock) -> None:
    result = AskAboutVideoTool()(WATCH_URL, "")

    assert result.startswith("Error:")
    assert "question" in result
    assert not http.calls


def test_ask_about_video_without_a_key_says_so(http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")

    result = AskAboutVideoTool()(WATCH_URL, "what happens?")

    assert result.startswith("Error:")
    assert "GEMINI_API_KEY" in result
    assert not http.calls


def test_ask_about_video_reports_api_failures(http: responses.RequestsMock) -> None:
    http.add(responses.POST, MODEL_URL, status=400, json={"error": {"message": "video is private"}})

    result = AskAboutVideoTool()(WATCH_URL, "what happens?")

    assert result.startswith("Error:")
    assert "video is private" in result


# --------------------------------------------------------------------------- HTTP failures


def test_the_back_off_pause_really_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)

    REAL_PAUSE(2.5)

    assert slept == [2.5]


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_reported_without_echoing_it(http: responses.RequestsMock, status: int) -> None:
    http.add(responses.POST, MODEL_URL, status=status, json={"error": {"message": f"key {FAKE_KEY} is bad"}})

    result = ask()

    assert result.startswith("Error:")
    assert "GEMINI_API_KEY" in result
    assert FAKE_KEY not in result
    assert len(http.calls) == 1


def test_an_unknown_model_explains_how_to_fix_it(http: responses.RequestsMock) -> None:
    http.add(responses.POST, MODEL_URL, status=404, json={"error": {"message": "model not found"}})

    result = ask()

    assert "gemini-2.5-flash" in result
    assert "GAIA_GEMINI_MODEL" in result


def test_a_bad_request_relays_the_api_message_with_the_key_redacted(http: responses.RequestsMock) -> None:
    http.add(responses.POST, MODEL_URL, status=400, json={"error": {"message": f"Bad video near {FAKE_KEY}"}})

    result = ask()

    assert "HTTP 400" in result
    assert "Bad video near [redacted]" in result
    assert FAKE_KEY not in result


def test_a_busy_service_is_retried_with_back_off(http: responses.RequestsMock, pauses: list[float]) -> None:
    http.add(responses.POST, MODEL_URL, status=503, json={"error": {"message": "overloaded"}})
    http.add(responses.POST, MODEL_URL, status=500, json={"error": {"message": "slow down"}})
    http.add(responses.POST, MODEL_URL, json=reply("third time lucky"))

    assert ask() == "third time lucky"
    assert len(http.calls) == 3
    assert len(pauses) == 2
    assert pauses[0] < pauses[1]


def test_a_daily_quota_429_is_reported_after_the_retries(http: responses.RequestsMock, pauses: list[float]) -> None:
    http.add(responses.POST, MODEL_URL, status=429, json={"error": {"message": "quota"}})

    result = ask()

    assert result.startswith("Error:")
    assert "429" in result
    assert len(http.calls) == 3
    assert len(pauses) == 2


def test_a_server_error_is_reported(http: responses.RequestsMock) -> None:
    http.add(responses.POST, MODEL_URL, status=500, body="internal failure")

    result = ask()

    assert "HTTP 500" in result
    assert "internal failure" in result


def test_a_timeout_and_a_connection_failure_become_short_errors(http: responses.RequestsMock) -> None:
    http.add(responses.POST, MODEL_URL, body=requests.Timeout("read timed out"))
    assert "timed out" in ask()

    http.reset()
    http.add(responses.POST, MODEL_URL, body=requests.ConnectionError(f"failed for {FAKE_KEY}"))
    result = ask()
    assert "could not reach" in result
    assert FAKE_KEY not in result


# --------------------------------------------------------------------------- the model and the reply


def test_the_model_comes_from_the_environment(http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_GEMINI_MODEL", "models/gemini-custom-9.1")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-custom-9.1:generateContent"
    http.add(responses.POST, url, json=reply("custom model"))

    assert ask() == "custom model"


@pytest.mark.parametrize("model", ["bad/model", "x?key=1", "with space", "../escape"])
def test_a_malformed_model_name_is_refused(
    http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("GAIA_GEMINI_MODEL", model)

    result = ask()

    assert result.startswith("Error:")
    assert "GAIA_GEMINI_MODEL" in result
    assert not http.calls


def test_multi_part_answers_are_joined_and_thoughts_skipped(http: responses.RequestsMock) -> None:
    parts = [{"text": "hidden", "thought": True}, {"text": "Two "}, {"text": "parts"}]
    http.add(responses.POST, MODEL_URL, json={"candidates": [{"content": {"parts": parts}}]})

    assert ask() == "Two parts"


def test_a_very_long_answer_is_truncated(http: responses.RequestsMock) -> None:
    http.add(responses.POST, MODEL_URL, json=reply("word " * 3000))

    result = ask()

    assert "[truncated:" in result
    assert len(result) < 4200


def test_a_blocked_prompt_is_reported(http: responses.RequestsMock) -> None:
    http.add(responses.POST, MODEL_URL, json={"promptFeedback": {"blockReason": "SAFETY"}})

    result = ask()

    assert result.startswith("Error:")
    assert "blocked" in result
    assert "SAFETY" in result


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, "MAX_TOKENS"),
        ({"candidates": []}, "no candidates"),
        ({}, "no candidates"),
    ],
)
def test_a_reply_without_text_is_reported(http: responses.RequestsMock, body: dict, expected: str) -> None:
    http.add(responses.POST, MODEL_URL, json=body)

    result = ask()

    assert result.startswith("Error:")
    assert expected in result


@pytest.mark.parametrize("body", ["not json at all", "[1, 2, 3]"])
def test_an_unreadable_reply_is_reported(http: responses.RequestsMock, body: str) -> None:
    http.add(responses.POST, MODEL_URL, body=body)

    result = ask()

    assert result.startswith("Error:")
    assert "unreadable" in result
