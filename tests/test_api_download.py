"""Tests for ``ScoringClient.download_file`` and the helpers that keep attachments safe.

Covers the two sources (the scoring API, then the dataset on the Hub), the 20 MB limit, the cache,
the retries and the handling of hostile file names. Doubles and fixtures live in ``api_fakes.py``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import responses
from api_fakes import (
    FILE_URL,
    TOKEN,
    WITH_FILE,
    FakeHub,
    FakeResponse,
    FakeSession,
    add_get,
    client,  # noqa: F401  (fixtures)
    hf_token,  # noqa: F401
    hub,  # noqa: F401
    rsps,  # noqa: F401
    sleeps,  # noqa: F401
)

from gaia_agent import _attachments as attachments
from gaia_agent import api
from gaia_agent.api import ApiError, Question, ScoringClient, UnsafeFileNameError, sanitize_file_name
from gaia_agent.config import Settings

Rsps = responses.RequestsMock
PNG = b"png-bytes"
UNSAFE_NAMES = [
    "../../evil.txt", "..\\..\\evil.txt", "..", ".", "a/../b.txt", "report..xlsx", "C:\\", "evil:stream.txt",
    "what?.txt", "a|b.txt", "a\x00b.txt", "NUL", "con.txt", "COM1",
    # The metadata of the benchmark holds the answer key: if the API served it as an "attachment", it is not saved.
    "metadata.jsonl", "Metadata.PARQUET", "metadata.level1.parquet", "metadata",
    # Windows drops a trailing dot or space, so these would land under another name than the one that was checked.
    "report.xlsx.", "report.xlsx ",
]  # fmt: skip


def named(file_name: str) -> Question:
    return dataclasses.replace(WITH_FILE, file_name=file_name)


# --------------------------------------------------------------------- the API route


def test_download_file_without_attachment_does_nothing(client: ScoringClient, rsps: Rsps, hub: FakeHub) -> None:
    assert client.download_file(dataclasses.replace(WITH_FILE, file_name="")) is None
    assert not rsps.calls
    assert hub.calls == []


def test_download_file_saves_the_api_answer_in_the_files_folder(
    client: ScoringClient, rsps: Rsps, settings: Settings
) -> None:
    add_get(rsps, FILE_URL, body=PNG)

    path = client.download_file(WITH_FILE)

    assert path == settings.files_dir / "task-002.png"
    assert path.read_bytes() == PNG
    assert [p.name for p in settings.files_dir.iterdir()] == ["task-002.png"]


def test_download_file_reuses_the_cached_file(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, FILE_URL, body=PNG)

    assert client.download_file(WITH_FILE) == client.download_file(WITH_FILE)
    assert len(rsps.calls) == 1


def test_download_file_ignores_an_empty_leftover_file(client: ScoringClient, rsps: Rsps, settings: Settings) -> None:
    (settings.files_dir / "task-002.png").write_bytes(b"")
    add_get(rsps, FILE_URL, body=PNG)

    assert client.download_file(WITH_FILE).read_bytes() == PNG


def test_download_file_retries_server_errors(client: ScoringClient, rsps: Rsps, sleeps: list[float]) -> None:
    add_get(rsps, FILE_URL, 500)
    add_get(rsps, FILE_URL, body=PNG)

    assert client.download_file(WITH_FILE).read_bytes() == PNG
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    "error",
    [requests.ConnectionError("reset"), requests.exceptions.ChunkedEncodingError("cut"), requests.ReadTimeout("slow")],
)
def test_download_file_retries_a_body_that_breaks_midway(
    settings: Settings, sleeps: list[float], error: Exception
) -> None:
    broken, whole = FakeResponse((b"ab",), error), FakeResponse((b"ab", b"cd"))
    client = ScoringClient(settings, FakeSession(broken, whole), sleep=sleeps.append)

    path = client.download_file(WITH_FILE)

    assert path.read_bytes() == b"abcd"
    assert broken.closed
    assert sleeps == [1.0]


def test_download_file_returns_none_when_the_files_folder_cannot_be_used(
    client: ScoringClient, rsps: Rsps, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(root: Path, split_dir: str, name: str) -> Path:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(api, "cached_copy", refuse)

    assert client.download_file(WITH_FILE) is None
    assert not rsps.calls
    assert "Cannot fetch the attachment" in caplog.text


def test_download_file_survives_a_malformed_redirect_address(client: ScoringClient, rsps: Rsps) -> None:
    add_get(rsps, FILE_URL, 302, headers={"Location": "http://[::1"})  # requests raises a bare ValueError

    assert client.download_file(WITH_FILE) is None


def test_download_file_quotes_the_task_id_in_the_url(settings: Settings, sleeps: list[float]) -> None:
    session = FakeSession(FakeResponse((b"x",)))
    client = ScoringClient(settings, session, sleep=sleeps.append)

    client.download_file(dataclasses.replace(WITH_FILE, task_id="a/b?c#d"))

    assert session.calls[0][1] == f"{settings.api_url}/files/a%2Fb%3Fc%23d"


# ------------------------------------------------------------------ the dataset route


def test_download_file_404_without_token_returns_none_and_says_why(
    client: ScoringClient, rsps: Rsps, hub: FakeHub, caplog: pytest.LogCaptureFixture
) -> None:
    add_get(rsps, FILE_URL, 404, json={"detail": "No file path associated with task_id task-002."})

    assert client.download_file(WITH_FILE) is None
    assert hub.calls == []
    assert "HTTP 404" in caplog.text
    assert "HF_TOKEN" in caplog.text


def test_download_file_404_with_token_falls_back_to_the_dataset(
    client: ScoringClient, rsps: Rsps, hub: FakeHub, hf_token: str, settings: Settings
) -> None:
    add_get(rsps, FILE_URL, 404, json={"detail": "missing"})

    path = client.download_file(WITH_FILE)

    assert hub.calls == [
        {
            "repo_id": "gaia-benchmark/GAIA",
            "repo_type": "dataset",
            "filename": "2023/validation/task-002.png",
            "token": hf_token,
            "local_dir": settings.files_dir,
        }
    ]
    assert path == settings.files_dir / "2023" / "validation" / "task-002.png"
    assert path.read_bytes() == b"from-the-dataset"


def test_download_file_reuses_the_dataset_copy(client: ScoringClient, rsps: Rsps, hub: FakeHub, hf_token: str) -> None:
    add_get(rsps, FILE_URL, 404)

    assert client.download_file(WITH_FILE) == client.download_file(WITH_FILE)
    assert len(rsps.calls) == 1
    assert len(hub.calls) == 1


def test_download_file_falls_back_after_repeated_server_errors(
    client: ScoringClient, rsps: Rsps, hub: FakeHub, hf_token: str
) -> None:
    add_get(rsps, FILE_URL, 500)

    assert client.download_file(WITH_FILE) is not None
    assert len(rsps.calls) == 3
    assert len(hub.calls) == 1


def test_download_file_asks_the_dataset_for_the_sanitised_name(
    client: ScoringClient, rsps: Rsps, hub: FakeHub, hf_token: str
) -> None:
    add_get(rsps, FILE_URL, 404)

    client.download_file(named("odd/dir\\task-002.png"))

    assert hub.calls[0]["filename"] == "2023/validation/task-002.png"


def test_download_file_dataset_failure_returns_none_and_hides_the_token(
    client: ScoringClient, rsps: Rsps, hub: FakeHub, hf_token: str, caplog: pytest.LogCaptureFixture
) -> None:
    add_get(rsps, FILE_URL, 404)
    hub.error = RuntimeError(f"401 Unauthorized for token {TOKEN}")

    assert client.download_file(WITH_FILE) is None
    assert TOKEN not in caplog.text
    assert "RuntimeError" in caplog.text
    assert "***" in caplog.text


def test_the_token_is_not_in_the_exception_chain_of_a_dataset_failure(
    client: ScoringClient, hub: FakeHub, hf_token: str, settings: Settings
) -> None:
    """The original exception quotes the token; ``from None`` keeps it out of every traceback."""
    hub.error = RuntimeError(f"401 Unauthorized for token {TOKEN}")

    with pytest.raises(ApiError) as caught:
        client._download_from_dataset("task-002.png", settings.files_dir)

    assert TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_download_file_refuses_a_dataset_path_outside_the_files_folder(
    client: ScoringClient, rsps: Rsps, hub: FakeHub, hf_token: str, tmp_path: Path
) -> None:
    add_get(rsps, FILE_URL, 404)
    hub.returns = tmp_path / "elsewhere.png"
    hub.returns.write_bytes(b"x")

    assert client.download_file(WITH_FILE) is None


# --------------------------------------------------------------------- the size limit


@pytest.mark.parametrize(
    ("limit", "size", "saved"), [(10, 10, True), (10, 11, False)], ids=["at-the-limit", "over-the-limit"]
)
def test_download_file_enforces_the_size_limit_while_streaming(
    client: ScoringClient,
    rsps: Rsps,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
    size: int,
    saved: bool,
) -> None:
    monkeypatch.setattr(api, "MAX_FILE_BYTES", limit)
    add_get(rsps, FILE_URL, body=b"x" * size)

    assert (client.download_file(WITH_FILE) is not None) is saved
    assert [p.name for p in settings.files_dir.iterdir()] == (["task-002.png"] if saved else [])


def test_download_file_refuses_a_declared_size_over_the_limit(
    client: ScoringClient, rsps: Rsps, settings: Settings
) -> None:
    add_get(rsps, FILE_URL, body=b"x", headers={"Content-Length": str(api.MAX_FILE_BYTES + 1)})

    assert client.download_file(WITH_FILE) is None
    assert list(settings.files_dir.iterdir()) == []


def test_a_download_that_drips_on_forever_is_abandoned() -> None:
    """The timeout of ``requests`` is per socket read: a server that sends a byte every few seconds never trips it."""
    ticks = iter(range(0, 10_000, 100))
    dripping = FakeResponse(tuple(b"x" for _ in range(50)))

    with pytest.raises(ApiError, match="took too long"):
        api._read_limited(dripping, deadline_s=250, clock=lambda: next(ticks))


def test_a_download_within_its_deadline_is_read_whole() -> None:
    steady = FakeResponse((b"ab", b"cd"))

    assert api._read_limited(steady, deadline_s=250, clock=lambda: 0.0) == b"abcd"


def test_download_file_gives_up_on_a_slow_download(
    client: ScoringClient, rsps: Rsps, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "DOWNLOAD_DEADLINE_SECONDS", -1.0)  # every download is already too slow
    add_get(rsps, FILE_URL, body=b"x" * 10)

    assert client.download_file(WITH_FILE) is None
    assert list(settings.files_dir.iterdir()) == []


def test_download_file_removes_an_oversized_dataset_file(
    client: ScoringClient,
    rsps: Rsps,
    hub: FakeHub,
    hf_token: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "MAX_FILE_BYTES", 5)
    add_get(rsps, FILE_URL, 404)

    assert client.download_file(WITH_FILE) is None
    assert not (settings.files_dir / "2023" / "validation" / "task-002.png").exists()


def test_download_file_leaves_no_partial_file_when_the_write_fails(
    client: ScoringClient, rsps: Rsps, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(source: str, target: Path) -> None:
        raise OSError("the disk said no")

    monkeypatch.setattr(attachments, "os", SimpleNamespace(fdopen=os.fdopen, replace=refuse))
    add_get(rsps, FILE_URL, body=PNG)

    assert client.download_file(WITH_FILE) is None
    assert list(settings.files_dir.iterdir()) == []


# ------------------------------------------------------------------------ file names


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_download_file_refuses_unsafe_names_before_touching_anything(
    client: ScoringClient,
    rsps: Rsps,
    hub: FakeHub,
    hf_token: str,
    settings: Settings,
    tmp_path: Path,
    name: str,
) -> None:
    assert client.download_file(named(name)) is None

    assert not rsps.calls
    assert hub.calls == []
    assert list(settings.files_dir.iterdir()) == []
    assert not list(tmp_path.rglob("evil*"))


@pytest.mark.parametrize("name", ["sub/dir/data.xlsx", "C:\\Windows\\data.xlsx", "/etc/data.xlsx", "C:data.xlsx"])
def test_download_file_keeps_only_the_last_path_component(
    client: ScoringClient, rsps: Rsps, settings: Settings, name: str
) -> None:
    add_get(rsps, FILE_URL, body=b"data")

    assert client.download_file(named(name)) == settings.files_dir / "data.xlsx"


def test_download_file_checks_the_destination_even_if_the_sanitiser_is_bypassed(
    client: ScoringClient,
    rsps: Rsps,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(api, "sanitize_file_name", lambda raw: "../escape.txt")

    assert client.download_file(WITH_FILE) is None
    assert not rsps.calls
    assert not (settings.files_dir.parent / "escape.txt").exists()
    assert "files folder" in caplog.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("task-002.png", "task-002.png"),
        ("a/b\\c.txt", "c.txt"),
        ("résumé (1).pdf", "résumé (1).pdf"),
        ("console.txt", "console.txt"),
        (".hidden", ".hidden"),
    ],
)
def test_sanitize_file_name_accepts_ordinary_names(raw: str, expected: str) -> None:
    assert sanitize_file_name(raw) == expected


@pytest.mark.parametrize("raw", ["", *UNSAFE_NAMES])
def test_sanitize_file_name_refuses_unsafe_names(raw: str) -> None:
    with pytest.raises(UnsafeFileNameError):
        sanitize_file_name(raw)
