"""Tests for the scoring API client: questions and submission.

Attachments are covered in ``test_api_download.py`` and the answer checks in ``test_submission.py``;
the doubles and fixtures live in ``api_fakes.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests
import responses
from api_fakes import (
    CODE,
    FILE_URL,
    GOOD_ANSWERS,
    QUESTIONS,
    QUESTIONS_JSON,
    QUESTIONS_URL,
    SUBMIT_URL,
    USER,
    WITH_FILE,
    FakeHub,
    FakeResponse,
    FakeSession,
    add_get,
    client,  # noqa: F401  (fixtures)
    hf_token,  # noqa: F401
    hub,  # noqa: F401
    rsps,  # noqa: F401
    seed_cache,
    sleeps,  # noqa: F401
)

from gaia_agent import api
from gaia_agent.api import ApiError, InvalidSubmissionError, ScoringClient
from gaia_agent.config import REPO_ROOT, ConfigError, Settings

Rsps = responses.RequestsMock


# --------------------------------------------------------------------- get_questions


def test_get_questions_downloads_parses_and_caches(client: ScoringClient, rsps: Rsps, settings: Settings) -> None:
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)

    assert client.get_questions() == QUESTIONS
    assert json.loads((settings.cache_dir / "questions.json").read_text(encoding="utf-8")) == QUESTIONS_JSON


def test_get_questions_reads_the_cache_without_network(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)

    assert client.get_questions() == client.get_questions()
    assert len(rsps.calls) == 1


def test_get_questions_refresh_downloads_again(client: ScoringClient, rsps: Rsps, settings: Settings) -> None:
    seed_cache(settings, json.dumps(QUESTIONS_JSON[:1]))
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)

    assert len(client.get_questions()) == 1
    assert len(client.get_questions(refresh=True)) == 3
    assert len(client.get_questions()) == 3


@pytest.mark.parametrize("cached", ["{not json", '{"a": 1}', "[]"], ids=["garbage", "wrong-shape", "empty"])
def test_get_questions_replaces_an_unusable_cache(
    client: ScoringClient, rsps: Rsps, settings: Settings, caplog: pytest.LogCaptureFixture, cached: str
) -> None:
    seed_cache(settings, cached)
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)

    assert client.get_questions() == QUESTIONS
    assert "unusable question cache" in caplog.text


def test_get_questions_accepts_lowercase_level_and_null_file_name(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, QUESTIONS_URL, json=[{"task_id": "t-1", "question": "Q?", "level": 2, "file_name": None}])

    assert [(q.level, q.file_name) for q in client.get_questions()] == [("2", "")]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        [1],
        [{"question": "q"}],
        [{"task_id": "t"}],
        [{"task_id": " ", "question": "q"}],
        [{"task_id": "t", "question": "q", "file_name": 5}],
    ],
    ids=["dict", "empty", "not-an-object", "no-task-id", "no-question", "blank-task-id", "bad-file-name"],
)
def test_get_questions_rejects_a_malformed_list(
    client: ScoringClient, rsps: Rsps, settings: Settings, payload: Any
) -> None:
    add_get(rsps, QUESTIONS_URL, json=payload)

    with pytest.raises(ApiError, match="unexpected question list"):
        client.get_questions()
    assert not (settings.cache_dir / "questions.json").exists()


def test_get_questions_rejects_a_reply_that_is_not_json(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, QUESTIONS_URL, body="<html>oops</html>")

    with pytest.raises(ApiError, match="valid JSON"):
        client.get_questions()


def test_get_questions_does_not_retry_client_errors(client: ScoringClient, rsps: Rsps, sleeps: list[float]) -> None:
    add_get(rsps, QUESTIONS_URL, 404, json={"detail": "no such thing"})

    with pytest.raises(ApiError, match="HTTP 404: no such thing"):
        client.get_questions()
    assert len(rsps.calls) == 1
    assert sleeps == []


def test_error_details_from_the_server_are_short_and_printable(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, QUESTIONS_URL, 400, json={"detail": "line one\nline two\x1b[31m " + "x" * 500})

    with pytest.raises(ApiError) as caught:
        client.get_questions()

    message = str(caught.value)
    assert message.isprintable()
    assert "line one line two" in message
    assert len(message) < 500


def test_get_questions_does_not_retry_permanent_request_errors(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, QUESTIONS_URL, requests.exceptions.TooManyRedirects("loop"))

    with pytest.raises(ApiError, match="TooManyRedirects"):
        client.get_questions()
    assert len(rsps.calls) == 1


def test_get_questions_reports_a_malformed_redirect_address(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, QUESTIONS_URL, 302, headers={"Location": "http://[::1"})  # requests raises a bare ValueError

    with pytest.raises(ApiError, match="ValueError"):
        client.get_questions()
    assert len(rsps.calls) == 1


@pytest.mark.parametrize(
    "failure",
    [503, requests.Timeout("slow"), requests.ConnectionError("down")],
    ids=["http-503", "timeout", "connection"],
)
def test_get_questions_retries_transient_failures_with_growing_waits(
    client: ScoringClient, rsps: Rsps, sleeps: list[float], failure: object
) -> None:
    add_get(rsps, QUESTIONS_URL, failure)
    add_get(rsps, QUESTIONS_URL, failure)
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)

    assert client.get_questions() == QUESTIONS
    assert len(rsps.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_get_questions_gives_up_after_three_attempts(client: ScoringClient, rsps: Rsps, sleeps: list[float]) -> None:
    add_get(rsps, QUESTIONS_URL, 502)

    with pytest.raises(ApiError, match=r"after 3 attempts.*HTTP 502"):
        client.get_questions()
    assert len(rsps.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_get_questions_still_answers_when_the_cache_cannot_be_written(
    client: ScoringClient, rsps: Rsps, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(path: Path, data: bytes) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(api, "write_atomic", refuse)
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)

    assert client.get_questions() == QUESTIONS
    assert "Could not cache the questions" in caplog.text


# ------------------------------------------------------------------------- submit


def test_submit_is_a_dry_run_by_default_and_never_touches_the_network(
    client: ScoringClient, rsps: Rsps, settings: Settings
) -> None:
    seed_cache(settings)

    result = client.submit(USER, CODE, GOOD_ANSWERS)

    assert result == {
        "dry_run": True,
        "url": SUBMIT_URL,
        "payload": {"username": USER, "agent_code": CODE, "answers": GOOD_ANSWERS},
    }
    assert not rsps.calls


@pytest.mark.parametrize("not_false", [None, 0, "", "no"])
def test_only_an_explicit_false_turns_a_dry_run_into_a_real_submission(
    client: ScoringClient, rsps: Rsps, settings: Settings, not_false: Any
) -> None:
    seed_cache(settings)

    assert client.submit(USER, CODE, GOOD_ANSWERS, dry_run=not_false)["dry_run"] is True
    assert not rsps.calls


def test_submit_dry_run_may_read_the_questions_but_never_posts(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)

    client.submit(USER, CODE, GOOD_ANSWERS, dry_run=True)

    assert [call.request.method for call in rsps.calls] == ["GET"]


def test_submit_sends_only_task_id_and_submitted_answer(client: ScoringClient, settings: Settings) -> None:
    seed_cache(settings)
    noisy = [{**GOOD_ANSWERS[0], "trace": "private reasoning", "model": "some-model"}]

    assert client.submit(USER, CODE, noisy)["payload"]["answers"] == [GOOD_ANSWERS[0]]


@pytest.mark.parametrize("dry_run", [True, False])
def test_submit_refuses_invalid_answers_before_any_post(
    client: ScoringClient, rsps: Rsps, settings: Settings, dry_run: bool
) -> None:
    seed_cache(settings)
    bad = [{"task_id": "nope", "submitted_answer": "FINAL ANSWER: x"}]

    with pytest.raises(InvalidSubmissionError) as caught:
        client.submit(USER, CODE, bad, dry_run=dry_run)

    assert len(caught.value.problems) == 2
    assert not rsps.calls


@pytest.mark.parametrize(("username", "agent_code"), [("", CODE), ("  ", CODE), (USER, ""), (None, CODE), (USER, 5)])
def test_submit_requires_a_username_and_an_agent_code(
    client: ScoringClient, settings: Settings, username: Any, agent_code: Any
) -> None:
    seed_cache(settings)

    with pytest.raises(InvalidSubmissionError, match="must be a non-empty string"):
        client.submit(username, agent_code, GOOD_ANSWERS)


@pytest.mark.parametrize("agent_code", ["short", "123456789", "         x"])
def test_submit_refuses_an_agent_code_shorter_than_the_api_minimum(
    client: ScoringClient, rsps: Rsps, settings: Settings, agent_code: str
) -> None:
    """The scoring API answers HTTP 422 below 10 characters; saying so first is kinder than a rejected submission."""
    seed_cache(settings)

    with pytest.raises(InvalidSubmissionError, match="at least 10 characters"):
        client.submit(USER, agent_code, GOOD_ANSWERS, dry_run=False)

    assert len(rsps.calls) == 0


def test_submit_accepts_an_agent_code_of_exactly_the_minimum_length(client: ScoringClient, settings: Settings) -> None:
    seed_cache(settings)

    assert client.submit(USER, "x" * 10, GOOD_ANSWERS)["dry_run"] is True


def test_submit_posts_the_payload_when_dry_run_is_off(client: ScoringClient, rsps: Rsps, settings: Settings) -> None:
    seed_cache(settings)
    reply = {"username": USER, "score": 50.0, "correct_count": 1, "total_attempted": 2, "message": "ok"}
    rsps.add(responses.POST, SUBMIT_URL, json=reply)

    assert client.submit(f"  {USER} ", CODE, GOOD_ANSWERS, dry_run=False) == reply

    request = rsps.calls[0].request
    assert json.loads(request.body) == {"username": USER, "agent_code": CODE, "answers": GOOD_ANSWERS}
    assert request.headers["Content-Type"] == "application/json"


def test_submit_retries_a_server_error(
    client: ScoringClient, rsps: Rsps, settings: Settings, sleeps: list[float]
) -> None:
    seed_cache(settings)
    rsps.add(responses.POST, SUBMIT_URL, status=500)
    rsps.add(responses.POST, SUBMIT_URL, json={"score": 100.0})

    assert client.submit(USER, CODE, GOOD_ANSWERS, dry_run=False) == {"score": 100.0}
    assert len(rsps.calls) == 2
    assert sleeps == [1.0]


def test_submit_gives_up_after_three_attempts(client: ScoringClient, rsps: Rsps, settings: Settings) -> None:
    seed_cache(settings)
    rsps.add(responses.POST, SUBMIT_URL, status=503)

    with pytest.raises(ApiError, match="after 3 attempts"):
        client.submit(USER, CODE, GOOD_ANSWERS, dry_run=False)
    assert len(rsps.calls) == 3


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_submit_never_follows_a_redirect_with_the_answers(
    client: ScoringClient, rsps: Rsps, settings: Settings, status: int
) -> None:
    """A 307 or 308 would send the username and the answers again to wherever the server points."""
    seed_cache(settings)
    rsps.add(responses.POST, SUBMIT_URL, status=status, headers={"Location": "https://elsewhere.example.test/collect"})

    with pytest.raises(ApiError, match=f"HTTP {status}"):
        client.submit(USER, CODE, GOOD_ANSWERS, dry_run=False)

    assert len(rsps.calls) == 1
    assert rsps.calls[0].request.url == SUBMIT_URL


def test_submit_reports_the_server_detail_without_retrying(
    client: ScoringClient, rsps: Rsps, settings: Settings
) -> None:
    seed_cache(settings)
    rsps.add(responses.POST, SUBMIT_URL, status=400, json={"detail": "Invalid task_id"})

    with pytest.raises(ApiError, match="HTTP 400: Invalid task_id"):
        client.submit(USER, CODE, GOOD_ANSWERS, dry_run=False)
    assert len(rsps.calls) == 1


@pytest.mark.parametrize(
    ("body", "message"), [("<html>", "valid JSON"), ("[1, 2]", "JSON object")], ids=["not-json", "not-an-object"]
)
def test_submit_rejects_an_unusable_reply(
    client: ScoringClient, rsps: Rsps, settings: Settings, body: str, message: str
) -> None:
    seed_cache(settings)
    rsps.add(responses.POST, SUBMIT_URL, body=body)

    with pytest.raises(ApiError, match=message):
        client.submit(USER, CODE, GOOD_ANSWERS, dry_run=False)


# ----------------------------------------------------------------- the client itself


def test_each_call_uses_its_own_timeout(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)
    add_get(rsps, FILE_URL, body=b"x")
    rsps.add(responses.POST, SUBMIT_URL, json={"score": 0.0})

    client.get_questions()
    client.download_file(WITH_FILE)
    client.submit(USER, CODE, GOOD_ANSWERS, dry_run=False)

    assert [call.request.req_kwargs["timeout"] for call in rsps.calls] == [30, 120, 180]
    assert rsps.calls[1].request.req_kwargs["stream"] is True


def test_the_hf_token_never_goes_to_the_scoring_api(
    client: ScoringClient, rsps: Rsps, hub: FakeHub, hf_token: str
) -> None:
    add_get(rsps, QUESTIONS_URL, json=QUESTIONS_JSON)
    add_get(rsps, FILE_URL, 404)

    client.get_questions()
    client.download_file(WITH_FILE)

    assert hub.calls[0]["token"] == hf_token
    assert all("Authorization" not in call.request.headers for call in rsps.calls)


def test_the_client_uses_the_session_it_is_given(settings: Settings) -> None:
    session = FakeSession(FakeResponse())

    assert ScoringClient(settings, session).get_questions() == QUESTIONS
    assert [(method, url) for method, url, _ in session.calls] == [("GET", f"{settings.api_url}/questions")]


def test_the_client_refuses_a_data_folder_inside_the_repository() -> None:
    with pytest.raises(ConfigError, match="outside the repository"):
        ScoringClient(Settings(data_dir=REPO_ROOT / "data"))
