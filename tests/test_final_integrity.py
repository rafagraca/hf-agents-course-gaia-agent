"""The proofs of a final run: code that is not hidden in ignored files, the tools that ran, the answers file.

Nothing here judges an answer: it only makes it harder for the code or the answers to change without a trace.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from final_fakes import COMMIT, OTHER_COMMIT, TAG, FakeGit

from gaia_agent import _final
from gaia_agent._final import FinalRunError, GitState
from gaia_agent.config import Settings

NOW = datetime(2026, 10, 7, 9, 30, 0, tzinfo=UTC)
NOW_TEXT = "2026-10-07T09:30:00+00:00"
HAS_GIT = shutil.which("git") is not None


# --- ignored files inside the package -------------------------------------------------------------------------


def test_an_ignored_file_inside_the_package_is_refused() -> None:
    git = FakeGit(ignored="!! gaia_agent/data/\n")

    with pytest.raises(FinalRunError, match=r"ignored.*gaia_agent/data/"):
        _final.require_release_state(git)


def test_an_ignored_file_also_stops_a_resume() -> None:
    with pytest.raises(FinalRunError, match="ignored"):
        _final.verify_unchanged({"commit": COMMIT, "tag": TAG}, FakeGit(ignored="!! gaia_agent/cache/\n"))


def test_bytecode_caches_are_not_a_problem() -> None:
    noise = "!! gaia_agent/__pycache__/\n!! gaia_agent/tools/__pycache__/\n"

    assert _final.require_release_state(FakeGit(ignored=noise)) == GitState(COMMIT, TAG)


def test_the_ignored_files_are_looked_for_in_the_package_only() -> None:
    git = FakeGit()

    _final.require_release_state(git)

    [asked] = [call for call in git.calls if "--ignored" in call]
    assert asked == ("status", "--porcelain", "--ignored", "--", "gaia_agent")


@pytest.mark.skipif(not HAS_GIT, reason="git is not installed")
def test_the_real_git_sees_an_ignored_folder_in_the_package_but_not_the_bytecode(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.test")
    git("config", "user.name", "Tester")
    (tmp_path / ".gitignore").write_text("data/\n__pycache__/\n", encoding="utf-8")
    (tmp_path / "gaia_agent" / "__pycache__").mkdir(parents=True)
    (tmp_path / "gaia_agent" / "__pycache__" / "x.pyc").write_bytes(b"x")
    (tmp_path / "gaia_agent" / "a.py").write_text("a = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "first")
    git("tag", "v9.9.9")
    runner = _final.git_runner(tmp_path)

    assert _final.require_release_state(runner).tag == "v9.9.9"
    (tmp_path / "gaia_agent" / "data").mkdir()
    (tmp_path / "gaia_agent" / "data" / "hidden.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(FinalRunError, match="ignored"):
        _final.require_release_state(runner)


@pytest.mark.skipif(not HAS_GIT, reason="git is not installed")
def test_the_real_git_runner_ignores_the_git_location_variables_of_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    for folder in (repository, elsewhere):
        folder.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=folder, check=True, capture_output=True)
    monkeypatch.setenv("GIT_DIR", str(elsewhere / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(elsewhere))
    monkeypatch.setenv("GIT_INDEX_FILE", str(elsewhere / ".git" / "index"))

    top = _final.git_runner(repository)(["rev-parse", "--show-toplevel"]).strip()

    assert Path(top).resolve() == repository.resolve()


# --- the tag still names the commit of the run ---------------------------------------------------------------


def test_the_commit_of_a_tag_is_read_from_the_tag() -> None:
    git = FakeGit(tag_commit=OTHER_COMMIT)

    assert _final.tag_commit(git, TAG) == OTHER_COMMIT
    assert ("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}^{{commit}}") in git.calls


def test_a_tag_that_does_not_exist_is_an_error() -> None:
    with pytest.raises(FinalRunError, match=r"tag v1\.0\.0 does not exist"):
        _final.tag_commit(FakeGit(tag_commit=""), TAG)


# --- what ran -------------------------------------------------------------------------------------------------


def test_the_variables_that_choose_the_models_and_the_gemini_model_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {"GAIA_MODEL_ID": "m", "GAIA_FALLBACK_MODELS": "a,b", "GAIA_GEMINI_MODEL": "g"}.items():
        monkeypatch.setenv(name, value)

    recorded = _final.behaviour_environment()

    assert recorded["GAIA_MODEL_ID"] == "m"
    assert recorded["GAIA_FALLBACK_MODELS"] == "a,b"
    assert recorded["GAIA_GEMINI_MODEL"] == "g"


def test_no_key_value_is_ever_among_the_recorded_variables() -> None:
    assert not [name for name in _final.BEHAVIOUR_VARIABLES if "KEY" in name or "TOKEN" in name]


def test_a_final_run_records_the_names_of_its_tools(settings: Settings) -> None:
    info = _final.start_info(settings, GitState(COMMIT, TAG), NOW, tool_names=("web_search", "deep_search"))

    assert info["tools"] == ["web_search", "deep_search"]


def test_a_resume_records_the_tools_and_the_model_of_that_moment(tmp_path: Path) -> None:
    _final.write_run_info(tmp_path, {"final": True, "commit": COMMIT, "tag": TAG})

    _final.record_resume(tmp_path, NOW, tool_names=("web_search",), model_id="some/model")

    info = _final.read_run_info(tmp_path)
    assert info is not None
    [entry] = info["resumed_at"]
    assert (entry["at"], entry["tools"], entry["model_id"]) == (NOW_TEXT, ["web_search"], "some/model")


# --- the answers file ---------------------------------------------------------------------------------------


def write_final_run(folder: Path, answers: str = '{"task_id": "t1", "answer": "x"}\n') -> Path:
    (folder / "answers.jsonl").write_text(answers, encoding="utf-8")
    _final.write_run_info(folder, {"final": True, "commit": COMMIT, "tag": TAG})
    return folder / "answers.jsonl"


def test_sealing_records_the_hash_of_the_answers_file(tmp_path: Path) -> None:
    answers = write_final_run(tmp_path)

    _final.seal_answers(tmp_path, NOW)

    info = _final.read_run_info(tmp_path)
    assert info is not None
    assert info["answers_sha256"] == hashlib.sha256(answers.read_bytes()).hexdigest()
    assert info["sealed_at"] == NOW_TEXT
    _final.verify_answers_sealed(info, answers)


def test_an_edited_answers_file_no_longer_matches_its_seal(tmp_path: Path) -> None:
    answers = write_final_run(tmp_path)
    _final.seal_answers(tmp_path, NOW)
    answers.write_text('{"task_id": "t1", "answer": "y"}\n', encoding="utf-8")
    info = _final.read_run_info(tmp_path)

    with pytest.raises(FinalRunError, match="changed since"):
        _final.verify_answers_sealed(info or {}, answers)


def test_a_run_that_was_never_sealed_is_refused(tmp_path: Path) -> None:
    answers = write_final_run(tmp_path)

    with pytest.raises(FinalRunError, match="not sealed"):
        _final.verify_answers_sealed(_final.read_run_info(tmp_path) or {}, answers)


def test_a_missing_answers_file_is_refused_when_verifying(tmp_path: Path) -> None:
    with pytest.raises(FinalRunError, match="answers"):
        _final.verify_answers_sealed({"answers_sha256": "0" * 64}, tmp_path / "answers.jsonl")


def test_sealing_without_run_json_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "answers.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(FinalRunError, match="run.json"):
        _final.seal_answers(tmp_path, NOW)


def test_sealing_a_folder_without_answers_is_an_error(tmp_path: Path) -> None:
    _final.write_run_info(tmp_path, {"final": True})

    with pytest.raises(FinalRunError, match="answers"):
        _final.seal_answers(tmp_path, NOW)
