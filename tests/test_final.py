"""Tests for the helpers of a final run: the git checks and ``run.json``."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from final_fakes import COMMIT, OTHER_COMMIT, TAG, FakeGit

from gaia_agent import _final
from gaia_agent._final import FinalRunError, GitError, GitState
from gaia_agent.config import Settings

NOW = datetime(2026, 10, 7, 9, 30, 0, tzinfo=UTC)
NOW_TEXT = "2026-10-07T09:30:00+00:00"
HAS_GIT = shutil.which("git") is not None
OTHER_VARIABLES = ("GAIA_MODEL_ID", "GAIA_FALLBACK_MODELS", "GAIA_GEMINI_MODEL")  # unset in these tests
NONE_OF_THE_OTHERS = dict.fromkeys(OTHER_VARIABLES)


# --- the checks on the repository ------------------------------------------------------------------------------


def test_a_clean_tagged_head_is_a_release_state() -> None:
    assert _final.require_release_state(FakeGit()) == GitState(commit=COMMIT, tag=TAG)


def test_a_dirty_tree_is_refused() -> None:
    with pytest.raises(FinalRunError, match="not clean"):
        _final.require_release_state(FakeGit(status=" M gaia_agent/run.py\n"))


def test_untracked_files_make_the_tree_dirty_too() -> None:
    with pytest.raises(FinalRunError, match="not clean"):
        _final.require_release_state(FakeGit(status="?? notes.txt\n"))


def test_a_head_without_a_tag_is_refused() -> None:
    with pytest.raises(FinalRunError, match="tag"):
        _final.require_release_state(FakeGit(tag=None))


def test_git_that_cannot_run_is_a_final_run_error() -> None:
    def broken(args: object) -> str:
        raise GitError("git is not installed")

    with pytest.raises(FinalRunError, match="git is not installed"):
        _final.require_release_state(broken)


def test_a_resume_needs_the_same_commit_and_a_clean_tree() -> None:
    info = {"final": True, "commit": COMMIT, "tag": TAG}

    _final.verify_unchanged(info, FakeGit())
    with pytest.raises(FinalRunError, match="HEAD changed"):
        _final.verify_unchanged(info, FakeGit(commit=OTHER_COMMIT))
    with pytest.raises(FinalRunError, match="not clean"):
        _final.verify_unchanged(info, FakeGit(status=" M x.py"))


@pytest.mark.skipif(not HAS_GIT, reason="git is not installed")
def test_the_real_git_runner_reads_a_temporary_repository(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.test")
    git("config", "user.name", "Tester")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-q", "-m", "first")
    runner = _final.git_runner(tmp_path)

    with pytest.raises(FinalRunError, match="tag"):
        _final.require_release_state(runner)
    git("tag", "v9.9.9")
    state = _final.require_release_state(runner)
    assert state.tag == "v9.9.9"
    assert len(state.commit) == 40
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    with pytest.raises(FinalRunError, match="not clean"):
        _final.require_release_state(runner)


def test_the_real_git_runner_reports_a_folder_that_is_not_a_repository(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        _final.git_runner(tmp_path / "missing")(["status", "--porcelain"])


def test_the_real_git_runner_reports_a_missing_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)

    with pytest.raises(GitError, match="not installed"):
        _final.git_runner(tmp_path)(["status"])


def test_the_real_git_runner_reports_a_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def slow(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("git", 30)

    monkeypatch.setattr(subprocess, "run", slow)

    with pytest.raises(GitError, match="too long"):
        _final.git_runner(tmp_path)(["status"])


# --- run.json -------------------------------------------------------------------------------------------------


def test_the_behaviour_variables_are_recorded_as_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_ALLOW_RUN_PYTHON", " 1 ")
    monkeypatch.delenv("GAIA_DEEP_SEARCH_MAX", raising=False)
    for name in OTHER_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    assert _final.behaviour_environment() == {
        "GAIA_ALLOW_RUN_PYTHON": "1",
        "GAIA_DEEP_SEARCH_MAX": None,
        **NONE_OF_THE_OTHERS,
    }


def test_a_blank_variable_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_DEEP_SEARCH_MAX", "  ")

    assert _final.behaviour_environment()["GAIA_DEEP_SEARCH_MAX"] is None


def test_the_description_of_a_final_run_says_what_it_ran_with(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GAIA_DEEP_SEARCH_MAX", "3")
    monkeypatch.delenv("GAIA_ALLOW_RUN_PYTHON", raising=False)
    for name in OTHER_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    info = _final.start_info(settings, GitState(COMMIT, TAG), NOW)

    assert info == {
        "final": True,
        "commit": COMMIT,
        "tag": TAG,
        "model_id": settings.model_id,
        "fallbacks": list(settings.fallback_model_ids),
        "max_steps": settings.max_steps,
        "started_at": NOW_TEXT,
        "environment": {"GAIA_ALLOW_RUN_PYTHON": None, "GAIA_DEEP_SEARCH_MAX": "3", **NONE_OF_THE_OTHERS},
        "tools": [],
    }


def test_run_info_round_trips(tmp_path: Path) -> None:
    info = {"final": True, "commit": COMMIT, "tag": TAG}

    _final.write_run_info(tmp_path, info)

    assert _final.read_run_info(tmp_path) == info
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8")) == info


def test_a_run_without_run_json_has_no_info(tmp_path: Path) -> None:
    assert _final.read_run_info(tmp_path) is None


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '"text"'])
def test_a_spoiled_run_json_is_an_error_not_a_non_final_run(tmp_path: Path, text: str) -> None:
    (tmp_path / "run.json").write_text(text, encoding="utf-8")

    with pytest.raises(FinalRunError, match="run.json"):
        _final.read_run_info(tmp_path)


def test_is_final_needs_a_true_flag() -> None:
    assert _final.is_final({"final": True})
    assert not _final.is_final({"final": "yes"})
    assert not _final.is_final({"final": False})
    assert not _final.is_final({})
    assert not _final.is_final(None)


def test_a_resume_is_added_to_run_json_with_the_variables_of_that_moment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _final.write_run_info(tmp_path, {"final": True, "commit": COMMIT, "tag": TAG, "started_at": "x"})
    monkeypatch.setenv("GAIA_ALLOW_RUN_PYTHON", "1")
    monkeypatch.delenv("GAIA_DEEP_SEARCH_MAX", raising=False)
    for name in OTHER_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    _final.record_resume(tmp_path, NOW)
    _final.record_resume(tmp_path, NOW)

    info = _final.read_run_info(tmp_path)
    assert info is not None
    assert info["started_at"] == "x"
    environment = {"GAIA_ALLOW_RUN_PYTHON": "1", "GAIA_DEEP_SEARCH_MAX": None, **NONE_OF_THE_OTHERS}
    assert info["resumed_at"] == [{"at": NOW_TEXT, "environment": environment, "tools": [], "model_id": None}] * 2


def test_recording_a_resume_without_run_json_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FinalRunError, match="run.json"):
        _final.record_resume(tmp_path, NOW)


def test_git_that_reports_nothing_for_head_is_refused() -> None:
    with pytest.raises(FinalRunError, match="did not report"):
        _final.require_release_state(FakeGit(commit=""))
