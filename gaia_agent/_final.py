"""What makes a run the final one: a clean tagged commit, and ``run.json`` to prove it.

A final run is the one whose answers are submitted. ``require_release_state`` insists that the code that runs is
exactly the code a tag names (a clean working tree, ``HEAD`` on a tag), ``start_info`` describes the run for
``run.json`` and ``submit`` later checks that the code link it is given points at that tag. Resuming a final run
is allowed only on the same commit; each resume is added to ``run.json`` together with the environment variables
that change the agent's behaviour at that moment.

``git`` is reached through a small callable (``GitRunner``) so that tests can script it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gaia_agent.config import REPO_ROOT, Settings
from gaia_agent.runs import write_text

__all__ = [
    "BEHAVIOUR_VARIABLES",
    "RUN_INFO_FILE",
    "FinalRunError",
    "GitError",
    "GitRunner",
    "GitState",
    "behaviour_environment",
    "git_runner",
    "is_final",
    "read_run_info",
    "record_resume",
    "require_release_state",
    "seal_answers",
    "start_info",
    "tag_commit",
    "verify_answers_sealed",
    "verify_unchanged",
    "write_run_info",
]

RUN_INFO_FILE = "run.json"
ANSWERS_FILE = "answers.jsonl"
# The environment variables that change what the agent does (as opposed to which keys it holds: those are never
# recorded). Which tools ran is recorded separately, by name (``start_info``): it also depends on which keys exist.
BEHAVIOUR_VARIABLES = (
    "GAIA_ALLOW_RUN_PYTHON",
    "GAIA_DEEP_SEARCH_MAX",
    "GAIA_MODEL_ID",
    "GAIA_FALLBACK_MODELS",
    "GAIA_GEMINI_MODEL",
)
GIT_TIMEOUT_SECONDS = 30
# Where git looks for the repository: inherited, they could point the checks at another repository.
GIT_LOCATION_VARIABLES = frozenset(
    {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_NAMESPACE"}
)
PACKAGE_FOLDER = "gaia_agent"  # whose ignored files (the .gitignore hides data, cache, tmp... at any depth) are checked
IGNORED_NOISE = "__pycache__"  # ignored and harmless: the bytecode Python writes next to the code
MAX_IGNORED_SHOWN = 5

GitRunner = Callable[[Sequence[str]], str]


class GitError(RuntimeError):
    """``git`` could not be run, or answered with an error."""


class FinalRunError(RuntimeError):
    """The conditions of a final run are not met; the message says which."""


@dataclass(frozen=True)
class GitState:
    """The commit that is checked out and the tag that names it."""

    commit: str
    tag: str


# --- git --------------------------------------------------------------------------------------------------------


def git_runner(cwd: Path = REPO_ROOT) -> GitRunner:
    """A runner of ``git`` commands in ``cwd``: it returns the standard output and raises ``GitError`` on failure."""

    def run(args: Sequence[str]) -> str:
        if not cwd.is_dir():
            raise GitError(f"the folder {cwd} does not exist")
        environment = {name: value for name, value in os.environ.items() if name not in GIT_LOCATION_VARIABLES}
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GitError("git is not installed or not on the PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git {args[0]} took too long (more than {GIT_TIMEOUT_SECONDS} s)") from exc
        if done.returncode != 0:
            reason = done.stderr.strip() or f"exit code {done.returncode}"
            raise GitError(f"git {' '.join(args)} failed: {reason}")
        return done.stdout

    return run


def _ask(git: GitRunner, *args: str) -> str:
    try:
        return git(args)
    except GitError as exc:
        raise FinalRunError(str(exc)) from exc


def _require_clean_tree(git: GitRunner) -> None:
    if _ask(git, "status", "--porcelain").strip():
        raise FinalRunError(
            "the working tree is not clean (changed or untracked files): commit or remove them first, "
            "a final run must be exactly the code of a tag"
        )
    _require_no_ignored_code(git)


def _require_no_ignored_code(git: GitRunner) -> None:
    """``git status`` leaves ignored files out, so ask for them: code or data in the package that no tag holds."""
    listing = _ask(git, "status", "--porcelain", "--ignored", "--", PACKAGE_FOLDER)
    found = [
        line[3:] for line in listing.splitlines() if line.startswith("!! ") and IGNORED_NOISE not in line[3:].split("/")
    ]
    if found:
        shown = ", ".join(found[:MAX_IGNORED_SHOWN]) + (", ..." if len(found) > MAX_IGNORED_SHOWN else "")
        raise FinalRunError(
            f"there are ignored files inside {PACKAGE_FOLDER}/ ({shown}): they change what runs without being in "
            "the tag; remove them first"
        )


def require_release_state(git: GitRunner) -> GitState:
    """The commit and tag of ``HEAD``; ``FinalRunError`` unless the tree is clean and ``HEAD`` is on a tag."""
    _require_clean_tree(git)
    commit = _ask(git, "rev-parse", "HEAD").strip()
    try:
        tag = git(["describe", "--exact-match", "--tags", "HEAD"]).strip()
    except GitError as exc:
        raise FinalRunError("HEAD is not on a tag: tag the commit (git tag <name>) before a final run") from exc
    if not commit or not tag:
        raise FinalRunError("git did not report a commit and a tag for HEAD")
    return GitState(commit=commit, tag=tag)


def tag_commit(git: GitRunner, tag: str) -> str:
    """The commit that ``tag`` names now; ``FinalRunError`` when this repository has no such tag."""
    try:
        return git(["rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"]).strip()
    except GitError as exc:
        raise FinalRunError(f"the tag {tag} does not exist in this repository") from exc


def verify_unchanged(info: Mapping[str, Any], git: GitRunner) -> None:
    """A final run may only go on with the code it started with: a clean tree and the same commit."""
    _require_clean_tree(git)
    commit = _ask(git, "rev-parse", "HEAD").strip()
    if commit != info.get("commit"):
        raise FinalRunError(
            f"HEAD changed since this final run started (it ran on {info.get('commit')}, now {commit}): "
            f"check out the tag {info.get('tag')} again"
        )


# --- run.json ---------------------------------------------------------------------------------------------------


def behaviour_environment(environ: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """The behaviour-changing variables as they are now (``None`` when unset or blank)."""
    source = os.environ if environ is None else environ
    return {name: (source.get(name) or "").strip() or None for name in BEHAVIOUR_VARIABLES}


def start_info(settings: Settings, state: GitState, now: datetime, tool_names: Sequence[str] = ()) -> dict[str, Any]:
    """The description of a final run that starts ``now`` on ``state``, for ``run.json``.

    ``tool_names`` are the names of the tools the agent has (never a key): they depend on which keys exist.
    """
    return {
        "final": True,
        "commit": state.commit,
        "tag": state.tag,
        "model_id": settings.model_id,
        "fallbacks": list(settings.fallback_model_ids),
        "max_steps": settings.max_steps,
        "started_at": now.isoformat(timespec="seconds"),
        "environment": behaviour_environment(),
        "tools": list(tool_names),
    }


def write_run_info(run_dir: Path, info: Mapping[str, Any]) -> None:
    write_text(run_dir / RUN_INFO_FILE, json.dumps(dict(info), ensure_ascii=False, indent=2) + "\n")


def read_run_info(run_dir: Path) -> dict[str, Any] | None:
    """The content of ``run.json`` in ``run_dir``; None when there is none, ``FinalRunError`` when it is spoiled."""
    path = run_dir / RUN_INFO_FILE
    if not path.is_file():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FinalRunError(f"{RUN_INFO_FILE} in {run_dir.name} cannot be read: {exc}") from exc
    if not isinstance(info, dict):
        raise FinalRunError(f"{RUN_INFO_FILE} in {run_dir.name} is not a JSON object")
    return info


def is_final(info: Mapping[str, Any] | None) -> bool:
    """True when ``info`` is the ``run.json`` of a final run."""
    return info is not None and info.get("final") is True


def record_resume(run_dir: Path, now: datetime, *, tool_names: Sequence[str] = (), model_id: str | None = None) -> None:
    """Add a ``resumed_at`` entry (time, behaviour variables, tools and model) to the ``run.json`` of ``run_dir``."""
    info = read_run_info(run_dir)
    if info is None:
        raise FinalRunError(f"there is no {RUN_INFO_FILE} in {run_dir.name} to add the resume to")
    entry = {
        "at": now.isoformat(timespec="seconds"),
        "environment": behaviour_environment(),
        "tools": list(tool_names),
        "model_id": model_id,
    }
    resumed = info.get("resumed_at")
    info["resumed_at"] = [*(resumed if isinstance(resumed, list) else []), entry]
    write_run_info(run_dir, info)


# --- the answers file ------------------------------------------------------------------------------------------


def _answers_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FinalRunError(f"the answers file {path} cannot be read: {exc}") from exc


def seal_answers(run_dir: Path, now: datetime) -> None:
    """Record in ``run.json`` the SHA-256 of ``answers.jsonl`` as the run left it (a seal against editing by hand)."""
    info = read_run_info(run_dir)
    if info is None:
        raise FinalRunError(f"there is no {RUN_INFO_FILE} in {run_dir.name} to seal the answers in")
    info["answers_sha256"] = _answers_hash(run_dir / ANSWERS_FILE)
    info["sealed_at"] = now.isoformat(timespec="seconds")
    write_run_info(run_dir, info)


def verify_answers_sealed(info: Mapping[str, Any], answers_path: Path) -> None:
    """``FinalRunError`` unless ``answers_path`` is exactly the file the run sealed."""
    sealed = info.get("answers_sha256")
    if not isinstance(sealed, str) or not sealed:
        raise FinalRunError(
            "the answers of this run are not sealed (no answers_sha256 in run.json): finish the run with "
            "`run --final --run-dir <run>`, which seals them"
        )
    if _answers_hash(answers_path) != sealed:
        raise FinalRunError(
            f"{answers_path.name} changed since the run sealed it: submit the answers exactly as the run left them"
        )
