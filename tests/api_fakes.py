"""Doubles, constants and fixtures shared by the scoring API tests.

``test_api.py`` and ``test_api_download.py`` import what they need from here. The network
never opens: ``responses`` fakes ``requests``, ``FakeHub`` replaces ``hf_hub_download`` and
the retry waits are recorded instead of slept. Every question, task id and file is invented.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import responses

from gaia_agent import api
from gaia_agent.api import Question, ScoringClient
from gaia_agent.config import Settings

API_URL = "https://scoring.example.test"
QUESTIONS_URL = f"{API_URL}/questions"
FILE_URL = f"{API_URL}/files/task-002"
SUBMIT_URL = f"{API_URL}/submit"
TOKEN = "fake-hf-token-for-tests"
QUESTIONS_JSON = [
    {"task_id": "task-001", "question": "What is 2 + 2?", "Level": "1", "file_name": ""},
    {"task_id": "task-002", "question": "How many legs has a spider?", "Level": "1", "file_name": "task-002.png"},
    {"task_id": "task-003", "question": "Name the capital of Portugal.", "Level": "2", "file_name": "task-003.xlsx"},
]
QUESTIONS = [Question(q["task_id"], q["question"], q["Level"], q["file_name"]) for q in QUESTIONS_JSON]
WITH_FILE = QUESTIONS[1]
USER, CODE = "some-user", "https://code.example.test/tree/main"
GOOD_ANSWERS = [{"task_id": "task-001", "submitted_answer": "4"}, {"task_id": "task-003", "submitted_answer": "Lisbon"}]


@pytest.fixture
def rsps() -> Iterator[responses.RequestsMock]:
    """The fake ``requests`` transport; ``rsps.calls`` records what the client sent."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


@pytest.fixture
def sleeps() -> list[float]:
    """The waits the client asked for between retries (it never really sleeps)."""
    return []


@pytest.fixture
def client(settings: Settings, sleeps: list[float]) -> ScoringClient:
    """A client for ``API_URL`` (written with a trailing slash on purpose) that records its waits."""
    return ScoringClient(dataclasses.replace(settings, api_url=API_URL + "/"), sleep=sleeps.append)


class FakeHub:
    """Stands in for ``hf_hub_download``: records the call and writes the file where the real one would."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.content = b"from-the-dataset"
        self.error: Exception | None = None
        self.returns: Path | None = None

    def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content)
        return str(self.returns or path)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub()
    monkeypatch.setattr(api, "hf_hub_download", fake)
    return fake


@pytest.fixture
def hf_token(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    return TOKEN


class FakeResponse:
    """The members of ``requests.Response`` the client uses; a stream can break after its chunks."""

    def __init__(self, chunks: tuple[bytes, ...] = (), error: Exception | None = None) -> None:
        self.status_code, self.headers, self.closed = 200, {}, False
        self._chunks, self._error = chunks, error

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        yield from self._chunks
        if self._error is not None:
            raise self._error

    def json(self) -> Any:
        return QUESTIONS_JSON

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """Replays queued outcomes: an exception is raised, anything else is returned."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes: list[object] = list(outcomes)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def add_get(rsps: responses.RequestsMock, url: str, failure: object = None, **kwargs: Any) -> None:
    """Register a GET reply; ``failure`` is an HTTP status to answer with or an exception to raise."""
    if isinstance(failure, int):
        kwargs["status"] = failure
    elif failure is not None:
        kwargs["body"] = failure
    rsps.add(responses.GET, url, **kwargs)


def seed_cache(settings: Settings, text: str | None = None) -> None:
    """Pre-fill the question cache (with ``QUESTIONS_JSON`` unless ``text`` is given)."""
    (settings.cache_dir / "questions.json").write_text(text or json.dumps(QUESTIONS_JSON), encoding="utf-8")
