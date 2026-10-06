"""Command line entry point: validate the answers of a final run and submit them to the scoring API.

Usage: ``uv run python -m gaia_agent.submit --username USER --agent-code URL [--run-dir DIR] [--yes]``
(default: the latest run). Nothing is sent unless the dry-run protection is explicitly switched off
with ``--yes``.

Only a final run can be submitted (``run --final`` writes ``run.json``), and ``--agent-code`` must be the link
``https://github.com/<owner>/<repo>/tree/<ref>`` where ``<ref>`` is the tag (or the commit) the run recorded, so that
the code a reader finds is the code that produced the answers. Before anything is sent, ``git`` is asked that this
tag still names the commit the run recorded, and ``answers.jsonl`` must be exactly the file the run sealed (its hash
is in ``run.json``). A run is sent once: ``--yes`` is refused when ``submission.json`` already holds a real reply, so
that the score cannot be used to try variations of the answers. Every question of the scoring API needs an
answer. The attempt (answers, reply or error) is saved as ``submission.json`` in the run folder. The console shows
counts and the score, the number of correct answers, the number attempted and the server's message, never the answers.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gaia_agent._cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, complain, configure_output, say
from gaia_agent._final import (
    FinalRunError,
    GitRunner,
    git_runner,
    is_final,
    read_run_info,
    tag_commit,
    verify_answers_sealed,
)
from gaia_agent.api import ApiError, InvalidSubmissionError, Question, ScoringClient
from gaia_agent.config import REPO_ROOT, ConfigError, Settings, load_settings
from gaia_agent.runs import (
    ANSWERS_FILE,
    SUBMISSION_FILE,
    RunDirError,
    is_answered,
    load_answers,
    pick_run_dir,
    write_text,
)

# https://github.com/<owner>/<repo>/tree/<ref>, nothing before it, nothing after it (no query, fragment or slash).
_AGENT_CODE = re.compile(
    r"https://github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<repo>[A-Za-z0-9._-]{1,100})/tree/(?P<ref>[^\s?#/][^\s?#]*)"
)
_NOT_REPOSITORY_NAMES = frozenset({".", ".."})
MAX_MISSING_SHOWN = 10


def submission_answers(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    """The answers to send: one per answered task, in the order of the answers file."""
    return [
        {"task_id": task_id, "submitted_answer": record["answer"]}
        for task_id, record in records.items()
        if is_answered(record)
    ]


def agent_code_problem(agent_code: str, info: Mapping[str, Any]) -> str | None:
    """Why ``agent_code`` is not the link to the code of the run described by ``info`` (None when it is)."""
    found = _AGENT_CODE.fullmatch(agent_code)
    ref = found.group("ref") if found else None
    if (
        found is None
        or found.group("repo") in _NOT_REPOSITORY_NAMES
        or ref not in (info.get("tag"), info.get("commit"))
    ):
        return (
            "--agent-code must be https://github.com/<owner>/<repo>/tree/<ref> with <ref> the tag of the final run "
            f"(here /tree/{info.get('tag')}) or its commit ({info.get('commit')}), so that the code behind the link "
            "is the code that gave the answers"
        )
    return None


def _unanswered(questions: Sequence[Question], answers: Sequence[Mapping[str, str]]) -> list[str]:
    answered = {answer["task_id"] for answer in answers}
    return [question.task_id for question in questions if question.task_id not in answered]


def _report_unanswered(missing: Sequence[str], total: int) -> None:
    shown = ", ".join(task_id[:8] for task_id in missing[:MAX_MISSING_SHOWN])
    more = f" and {len(missing) - MAX_MISSING_SHOWN} more" if len(missing) > MAX_MISSING_SHOWN else ""
    complain(f"error: {len(missing)} of {total} questions have no answer ({shown}{more}): run them (--run-dir) first")


class _UsageError(ValueError):
    """The arguments or the run cannot be used for a submission."""


@dataclass(frozen=True)
class _Ready:
    """What a submission needs, once the run and the code link have been checked."""

    settings: Settings
    run_dir: Path
    info: Mapping[str, Any]
    client: ScoringClient
    answers: list[dict[str, str]]


def _final_run(settings: Settings, run_dir_name: str | None) -> tuple[Path, dict[str, Any]]:
    """The run to submit and its ``run.json``; ``_UsageError`` when it is not a final run."""
    run_dir = pick_run_dir(settings.runs_dir, run_dir_name)
    info = read_run_info(run_dir)
    if info is None or not is_final(info):
        raise _UsageError(f"{run_dir.name} is not a final run (no final run.json): only `run --final` can be submitted")
    return run_dir, info


def _require_tag_unmoved(info: Mapping[str, Any], git: GitRunner) -> None:
    """The tag the link names must still name the commit that gave the answers (a tag can be moved)."""
    tag = str(info.get("tag"))
    now = tag_commit(git, tag)
    if now != info.get("commit"):
        raise _UsageError(
            f"the tag {tag} now names {now}, not {info.get('commit')} (the commit of the run): the code behind "
            "the link would not be the code that gave the answers"
        )


def _was_submitted(run_dir: Path) -> bool:
    """True when ``submission.json`` holds the server's reply to a real submission; ``_UsageError`` if unreadable."""
    path = run_dir / SUBMISSION_FILE
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _UsageError(f"{SUBMISSION_FILE} in {run_dir.name} cannot be read ({exc}): check it first") from exc
    return isinstance(record, dict) and record.get("dry_run") is False and record.get("reply") is not None


def _require_not_submitted(run_dir: Path) -> None:
    if _was_submitted(run_dir):
        raise _UsageError(
            f"{run_dir.name} was already submitted ({SUBMISSION_FILE} holds the server's reply): a final run is "
            "sent once, so that the score cannot be used to try other answers"
        )


def _keeps_record_of_submission(run_dir: Path) -> bool:
    """True when the record on disk must not be overwritten: it is the real submission (or cannot be read)."""
    try:
        return _was_submitted(run_dir)
    except _UsageError:
        return True


def _prepare(args: argparse.Namespace, git: GitRunner | None) -> _Ready:
    settings = load_settings()
    run_dir, info = _final_run(settings, args.run_dir)
    problem = agent_code_problem(args.agent_code, info)
    if problem is not None:
        raise _UsageError(problem)
    _require_tag_unmoved(info, git or git_runner(REPO_ROOT))
    verify_answers_sealed(info, run_dir / ANSWERS_FILE)
    if args.yes is True:
        _require_not_submitted(run_dir)
    client = ScoringClient(settings)
    answers = submission_answers(load_answers(run_dir / ANSWERS_FILE))
    return _Ready(settings, run_dir, info, client, answers)


def _warn_about_username(username: str) -> None:
    if username != username.lower():
        complain(
            f"warning: the username {username!r} has capital letters; send it exactly as it is on your "
            "Hugging Face profile, the leaderboard may treat a different capitalisation as another user"
        )


def main(argv: list[str] | None = None, *, git: GitRunner | None = None) -> int:
    """Run the CLI with ``argv`` (default: ``sys.argv[1:]``) and return the process exit code.

    ``git`` (default: ``git`` in the repository) is injectable for tests.
    """
    args = _parse_args(argv)
    configure_output()
    try:
        ready = _prepare(args, git)
    except (ConfigError, RunDirError, FinalRunError, _UsageError) as exc:
        complain(f"error: {exc}")
        return EXIT_USAGE
    _warn_about_username(args.username)
    return _submit(args, ready)


def _submit(args: argparse.Namespace, ready: _Ready) -> int:
    """Check that every question has an answer, then validate and (with ``--yes``) send them."""
    send = args.yes is True
    try:
        questions = ready.client.get_questions()
        missing = _unanswered(questions, ready.answers)
        if missing:
            _report_unanswered(missing, len(questions))
            return EXIT_FAILED
        reply = ready.client.submit(args.username, args.agent_code, ready.answers, dry_run=not send)
    except InvalidSubmissionError as exc:
        complain(f"error: the answers cannot be submitted ({len(exc.problems)} problem(s)):")
        for problem in exc.problems:
            complain(f"- {problem}")
        return EXIT_FAILED
    except ApiError as exc:
        _save(ready, args, sent=send, reply=None, error=str(exc))
        complain(f"error: the submission failed: {exc}")
        return EXIT_FAILED
    _save(ready, args, sent=send, reply=reply if send else None, error=None)
    _report(ready, reply if send else None)
    return EXIT_OK


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m gaia_agent.submit",
        description="Validate the answers of a run and, with --yes, submit them to the scoring API.",
    )
    parser.add_argument("--username", required=True, help="your Hugging Face username")
    parser.add_argument("--agent-code", required=True, help="the public URL of the agent's code")
    parser.add_argument("--run-dir", metavar="DIR", help="the run to submit (default: the latest run with answers)")
    parser.add_argument("--yes", action="store_true", help="really send the answers (otherwise: a dry run)")
    return parser.parse_args(argv)


def _submit_url(settings: Settings) -> str:
    return f"{settings.api_url.rstrip('/')}/submit"


def _save(
    ready: _Ready, args: argparse.Namespace, *, sent: bool, reply: dict[str, Any] | None, error: str | None
) -> None:
    """Keep a record of the attempt in the run folder (DATA only: it holds the answers).

    A dry run never replaces the record of a real submission.
    """
    if not sent and _keeps_record_of_submission(ready.run_dir):
        return
    record = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "dry_run": not sent,
        "url": _submit_url(ready.settings),
        "username": args.username,
        "agent_code": args.agent_code,
        "tag": ready.info.get("tag"),
        "commit": ready.info.get("commit"),
        "answers_sha256": ready.info.get("answers_sha256"),
        "answers": ready.answers,
        "reply": reply,
        "error": error,
    }
    try:
        write_text(ready.run_dir / SUBMISSION_FILE, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    except OSError as exc:
        complain(f"warning: could not save {SUBMISSION_FILE}: {exc}")


def _report(ready: _Ready, reply: Mapping[str, Any] | None) -> None:
    """The score, the counts and the server's message only: the answers stay in the run folder."""
    count, name = len(ready.answers), ready.run_dir.name
    if reply is None:
        say(
            f"Dry run: {count} answers from {name} are valid; nothing was sent. "
            f"Add --yes to submit them to {_submit_url(ready.settings)}."
        )
        return
    say(
        f"Submitted {count} answers from {name}: score {reply.get('score')}% "
        f"({reply.get('correct_count')}/{reply.get('total_attempted')} correct)."
    )
    message = reply.get("message")
    if message:
        say(str(message))


if __name__ == "__main__":
    raise SystemExit(main())
