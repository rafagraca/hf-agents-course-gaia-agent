"""Tests for ``run --final``: only a clean, tagged commit may make the run whose answers are submitted."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from agent_fakes import PLAIN, SHEET, final_reply
from final_fakes import COMMIT, OTHER_COMMIT, TAG, FakeGit
from test_run_cli import Harness, harness, only_run_dir, records  # noqa: F401  (fixture)

from gaia_agent import run
from gaia_agent.config import Settings
from gaia_agent.runs import append_record, new_run_dir

NOW = datetime(2026, 10, 7, 9, 30, 0, tzinfo=UTC)
NOW_TEXT = "2026-10-07T09:30:00+00:00"


def run_info(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_final(harness: Harness, settings: Settings, *extra: str) -> Path:
    """A final run that answers one question, which can be resumed later."""
    harness.client.questions = [PLAIN, SHEET]
    harness.replies[:] = [final_reply("a")]
    assert run.main(["--final", "--limit", "1", *extra], git=FakeGit(), now=lambda: NOW) == 0
    return only_run_dir(settings)


def test_a_final_run_writes_run_json_before_it_answers(
    harness: Harness, settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GAIA_ALLOW_RUN_PYTHON", "1")
    harness.client.questions = [PLAIN]
    harness.replies[:] = [final_reply("a")]

    assert run.main(["--final"], git=FakeGit(), now=lambda: NOW) == 0

    run_dir = only_run_dir(settings)
    info = run_info(run_dir)
    assert info.pop("environment")["GAIA_ALLOW_RUN_PYTHON"] == "1"
    assert info == {
        "final": True,
        "commit": COMMIT,
        "tag": TAG,
        "model_id": settings.model_id,
        "fallbacks": list(settings.fallback_model_ids),
        "max_steps": settings.max_steps,
        "started_at": NOW_TEXT,
        "tools": ["echo_tool"],
        "answers_sha256": sha256_of(run_dir / "answers.jsonl"),
        "sealed_at": NOW_TEXT,
    }
    assert [item["task_id"] for item in records(run_dir)] == [PLAIN.task_id]
    out = capsys.readouterr().out
    assert f"Final run on tag {TAG}" in out
    assert COMMIT[:7] in out


def test_a_dirty_tree_stops_a_final_run_before_anything_is_created(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run.main(["--final"], git=FakeGit(status=" M gaia_agent/run.py")) == 2

    assert "not clean" in capsys.readouterr().err
    assert list(settings.runs_dir.iterdir()) == []
    assert harness.models == [] and harness.client.downloads == []


def test_a_head_without_a_tag_stops_a_final_run(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run.main(["--final"], git=FakeGit(tag=None)) == 2

    assert "not on a tag" in capsys.readouterr().err
    assert list(settings.runs_dir.iterdir()) == []


def test_a_run_that_is_not_final_never_asks_git_and_writes_no_run_json(harness: Harness, settings: Settings) -> None:
    git = FakeGit()
    harness.client.questions = [PLAIN]
    harness.replies[:] = [final_reply("a")]

    assert run.main([], git=git) == 0

    assert git.calls == []
    assert not (only_run_dir(settings) / "run.json").exists()


def test_a_final_run_can_be_resumed_on_the_same_commit_and_the_resume_is_recorded(
    harness: Harness, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = start_final(harness, settings)
    monkeypatch.setenv("GAIA_DEEP_SEARCH_MAX", "2")
    harness.replies[:] = [final_reply("b")]

    assert run.main(["--final", "--run-dir", run_dir.name], git=FakeGit(), now=lambda: NOW) == 0

    assert [item["task_id"] for item in records(run_dir)] == [PLAIN.task_id, SHEET.task_id]
    info = run_info(run_dir)
    assert info["started_at"] == NOW_TEXT
    [resumed] = info["resumed_at"]
    assert resumed["at"] == NOW_TEXT
    assert resumed["environment"]["GAIA_DEEP_SEARCH_MAX"] == "2"
    assert resumed["tools"] == ["echo_tool"]
    assert resumed["model_id"] == settings.model_id
    assert info["answers_sha256"] == sha256_of(run_dir / "answers.jsonl")  # sealed again, with the new answer


def test_resuming_a_final_run_checks_the_code_even_without_the_final_flag(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = start_final(harness, settings)
    models_before = len(harness.models)

    assert run.main(["--run-dir", run_dir.name], git=FakeGit(status=" M x.py")) == 2

    assert "not clean" in capsys.readouterr().err
    assert len(harness.models) == models_before
    assert "resumed_at" not in run_info(run_dir)


def test_a_final_run_is_not_resumed_when_head_changed(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = start_final(harness, settings)
    models_before = len(harness.models)

    assert run.main(["--final", "--run-dir", run_dir.name], git=FakeGit(commit=OTHER_COMMIT)) == 2

    assert "HEAD changed" in capsys.readouterr().err
    assert len(harness.models) == models_before
    assert [item["task_id"] for item in records(run_dir)] == [PLAIN.task_id]


def test_a_final_run_cannot_adopt_a_run_that_was_not_started_as_final(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = new_run_dir(settings.runs_dir, datetime(2026, 1, 1))
    append_record(run_dir / "answers.jsonl", {"task_id": PLAIN.task_id, "answer": "x", "error": None})

    assert run.main(["--final", "--run-dir", run_dir.name], git=FakeGit()) == 2

    assert "not a final run" in capsys.readouterr().err
    assert not (run_dir / "run.json").exists()


def test_a_spoiled_run_json_stops_a_resume(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = new_run_dir(settings.runs_dir, datetime(2026, 1, 1))
    (run_dir / "run.json").write_text("{not json", encoding="utf-8")

    assert run.main(["--run-dir", run_dir.name], git=FakeGit()) == 2

    assert "run.json" in capsys.readouterr().err


def test_git_is_asked_in_the_repository_by_default(
    harness: Harness, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[Path] = []

    def runner(cwd: Path) -> FakeGit:
        asked.append(cwd)
        return FakeGit()

    monkeypatch.setattr(run, "git_runner", runner)
    harness.client.questions = [PLAIN]
    harness.replies[:] = [final_reply("a")]

    assert run.main(["--final"]) == 0

    assert asked == [run.REPO_ROOT]


def test_a_resume_that_cannot_be_recorded_stops_the_run_before_it_answers(
    harness: Harness, settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = start_final(harness, settings)
    models_before = len(harness.models)

    def refuse(run_dir: Path, now: datetime, **_: Any) -> None:
        raise run.FinalRunError("run.json went missing")

    monkeypatch.setattr(run, "record_resume", refuse)

    assert run.main(["--run-dir", run_dir.name], git=FakeGit()) == 2

    assert "run.json went missing" in capsys.readouterr().err
    assert len(harness.models) == models_before


def test_the_seal_is_made_even_when_the_run_is_interrupted(
    harness: Harness, settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.client.questions = [PLAIN, SHEET]
    harness.replies[:] = [final_reply("a")]
    real = run._attempt
    calls: list[str] = []

    def interrupt_second(session: Any, question: Any, attachment: Any) -> Any:
        calls.append(question.task_id)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(session, question, attachment)

    monkeypatch.setattr(run, "_attempt", interrupt_second)

    assert run.main(["--final"], git=FakeGit(), now=lambda: NOW) == 130

    run_dir = only_run_dir(settings)
    assert run_info(run_dir)["answers_sha256"] == sha256_of(run_dir / "answers.jsonl")


def test_a_run_that_is_not_final_is_never_sealed(harness: Harness, settings: Settings) -> None:
    harness.client.questions = [PLAIN]
    harness.replies[:] = [final_reply("a")]

    assert run.main([]) == 0

    assert not (only_run_dir(settings) / "run.json").exists()


def test_a_seal_that_cannot_be_written_is_a_warning_and_not_a_crash(
    harness: Harness, settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(run_dir: Path, now: datetime) -> None:
        raise run.FinalRunError("run.json went missing")

    monkeypatch.setattr(run, "seal_answers", refuse)
    harness.client.questions = [PLAIN]
    harness.replies[:] = [final_reply("a")]

    assert run.main(["--final"], git=FakeGit(), now=lambda: NOW) == 0

    err = capsys.readouterr().err
    assert "could not seal" in err
    assert "run.json went missing" in err
