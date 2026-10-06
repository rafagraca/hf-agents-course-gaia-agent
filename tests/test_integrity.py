"""Guard rails for the repository's integrity rules.

The repository is public, so it must never contain benchmark task ids (UUIDs), the text of the benchmark's
questions, their answers or credentials. These checks scan everything git tracks, and the whole history, for the
obvious signatures; they complement, not replace, a review.

* every tracked file (not only the Python sources) is scanned for task-id-shaped UUIDs and credentials;
* the history is scanned too: a UUID committed once stays in the history after the file is cleaned;
* when the question list of the benchmark has been downloaded on this machine, nothing tracked may contain a task
  id, its short form (the first 8 characters, which reports use) or a run of 8 words of a question. The check
  reads that file at run time and keeps nothing; it is skipped on a machine that has no such file.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from gaia_agent import config

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNED_DIRS = ("gaia_agent", "tests")
SCANNED_FILES = ("README.md", "pyproject.toml", ".gitignore", ".gitattributes", "LICENSE")
QUESTIONS_FILE = config._default_data_dir() / "cache" / "questions.json"  # the real one, read only by this module
QUESTION_WORDS_RUN = 8

UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
CREDENTIALS = {
    "Groq API key": re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
}
# Patterns that .gitignore must keep: the net under the rule that benchmark files never reach the repository.
REQUIRED_IGNORES = ("data/", "runs/", "cache/", "gaia/", "answers.jsonl", "*.parquet", "*.xlsx", "*.mp3", ".env")


def _git(*args: str) -> str | None:
    """The output of a git command in the repository, or None when git or the repository is not there."""
    try:
        done = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, check=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.decode("utf-8", errors="replace")


def _project_text_files() -> list[Path]:
    listing = _git("ls-files", "-z")
    if listing:
        return [REPO_ROOT / name for name in listing.split("\0") if name and (REPO_ROOT / name).is_file()]
    files = [path for name in SCANNED_DIRS for path in sorted((REPO_ROOT / name).rglob("*.py"))]
    return files + [REPO_ROOT / name for name in SCANNED_FILES if (REPO_ROOT / name).is_file()]


PROJECT_FILES = _project_text_files()


def _ids(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_the_scan_covers_the_project() -> None:
    names = {_ids(path) for path in PROJECT_FILES}

    assert {"gaia_agent/config.py", "tests/conftest.py", "pyproject.toml", ".gitignore", "README.md"} <= names


@pytest.mark.parametrize("path", PROJECT_FILES, ids=_ids)
def test_no_uuid_in_the_project_files(path: Path) -> None:
    found = UUID.findall(_read(path))

    assert not found, f"{_ids(path)} contains UUID-like strings (benchmark task ids?): use short fake ids instead"


@pytest.mark.parametrize("path", PROJECT_FILES, ids=_ids)
def test_no_credentials_in_the_project_files(path: Path) -> None:
    text = _read(path)

    leaks = [label for label, pattern in CREDENTIALS.items() if pattern.search(text)]

    assert not leaks, f"{_ids(path)} looks like it contains a {', '.join(leaks)}"


def test_the_history_holds_no_task_id_and_no_credential() -> None:
    history = _git("log", "-p", "--all", "--no-color", "--format=%H %an %ae %s")
    if history is None:
        pytest.skip("git is not available here")

    assert not UUID.findall(history), "a UUID-like string (a benchmark task id?) is in the git history"
    leaks = [label for label, pattern in CREDENTIALS.items() if pattern.search(history)]
    assert not leaks, f"the git history looks like it contains a {', '.join(leaks)}"


def test_gitignore_keeps_the_benchmark_files_out_of_the_repository() -> None:
    lines = {line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}

    assert [pattern for pattern in REQUIRED_IGNORES if pattern not in lines] == []


def test_no_tracked_file_is_a_run_output_or_a_benchmark_attachment() -> None:
    ignored = _git("ls-files", "-ci", "--exclude-standard")
    if ignored is None:
        pytest.skip("git is not available here")

    assert ignored.split() == [], "tracked files that .gitignore says must never be committed"


# --- against the real question list, when this machine has it ----------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _benchmark_questions() -> list[dict[str, object]]:
    try:
        payload = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


@pytest.mark.skipif(not _benchmark_questions(), reason="the question list is not downloaded on this machine")
def test_nothing_tracked_quotes_a_benchmark_question_or_names_a_task() -> None:
    questions = _benchmark_questions()
    haystack = "\n".join(_read(path) for path in PROJECT_FILES if path.name != "uv.lock")
    lowered = haystack.lower()
    squeezed = " ".join(_words(haystack))
    found: list[str] = []
    for index, item in enumerate(questions, start=1):
        task_id = str(item.get("task_id", "")).lower()
        if task_id and (task_id in lowered or task_id[:8] in lowered):
            found.append(f"question {index}: its task id")
        words = _words(str(item.get("question", "")))
        starts = range(len(words) - QUESTION_WORDS_RUN + 1)
        runs = (" ".join(words[start : start + QUESTION_WORDS_RUN]) for start in starts)
        if any(run in squeezed for run in runs):
            found.append(f"question {index}: a run of {QUESTION_WORDS_RUN} words of its text")

    assert found == [], "benchmark material in the tracked files (" + "; ".join(found) + ")"
