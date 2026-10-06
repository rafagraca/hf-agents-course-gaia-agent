"""Command line entry point: answer the exam questions with the agent.

Usage: ``uv run python -m gaia_agent.run [--task ID]... [--limit N] [--run-dir DIR] [--refresh] [--verbose]
[--final] [--wait-for-quota [--max-wait-hours H]] [--defer-missing-attachments]``.

Answers and traces are stored under ``<data>/runs/<YYYYMMDD-HHMMSS>``; ``--run-dir`` resumes an
existing run instead: questions already answered there (an answer and no error) are skipped and
failed ones are tried again. Each attempt is saved as soon as it ends, one line in
``answers.jsonl`` plus ``traces/<task_id>.md``. The console shows progress and token use, never
the answers, and nothing the agent itself prints unless ``--verbose`` asks for it (the errors of the
agent quote the replies of the model; the traces keep them). One ``BudgetedModel`` serves the whole
run while every question gets a fresh agent, and the run stops early when no model can answer anymore
(daily quotas). It refuses to start while a file that looks like the benchmark answer key is in the data folder.

* ``--final`` starts the run whose answers will be submitted: the working tree must be clean and ``HEAD`` on a
  tag, and ``run.json`` records what ran (see ``_final``) and, at the end of every session, the hash of
  ``answers.jsonl`` that ``submit`` checks. A final run is resumed only on the same commit.
* ``--wait-for-quota`` waits (at most ``--max-wait-hours``, 36 by default) instead of stopping when every model
  is out of daily quota, and asks the same question again afterwards (see ``_quota``).
* ``--defer-missing-attachments`` leaves a question whose attachment cannot be fetched unanswered, instead of
  answering it without the file; a later ``--run-dir`` run answers it once the file can be had.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from smolagents import Model, Tool
from smolagents.monitoring import LogLevel

from gaia_agent._cli import (
    EXIT_FAILED,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_USAGE,
    complain,
    configure_output,
    say,
    short_id,
)
from gaia_agent._final import (
    FinalRunError,
    GitRunner,
    GitState,
    git_runner,
    is_final,
    read_run_info,
    record_resume,
    require_release_state,
    seal_answers,
    start_info,
    verify_unchanged,
    write_run_info,
)
from gaia_agent._quota import QuotaWaiter, is_quota_exhausted
from gaia_agent._run_args import parse_args
from gaia_agent._run_report import Attempt, print_summary, progress_line
from gaia_agent.agent import answer_question, build_agent, is_fatal
from gaia_agent.api import ApiError, Question, ScoringClient
from gaia_agent.budget import BudgetedModel
from gaia_agent.config import REPO_ROOT, ConfigError, Settings, ensure_no_answer_key, load_settings
from gaia_agent.runs import (
    ANSWERS_FILE,
    USAGE_FILE,
    RunDirError,
    append_record,
    is_answered,
    load_answers,
    new_run_dir,
    read_usage,
    resolve_run_dir,
    summarize_usage,
    trace_path,
    write_text,
)
from gaia_agent.tools import default_tools
from gaia_agent.trace import format_trace

__all__ = ["REPO_ROOT", "UnknownTaskError", "attach_usage_log", "main", "reset_tools", "select_questions"]


class UnknownTaskError(ValueError):
    """``--task`` named a task id that is not in the question list."""


@dataclass(frozen=True)
class RunPlan:
    """What a run will do: the questions still to answer, where the results go and how to behave."""

    settings: Settings
    client: ScoringClient
    run_dir: Path
    pending: tuple[Question, ...]
    skipped: int
    defer_missing_attachments: bool = False
    final_info: Mapping[str, Any] | None = None  # the run.json of a final run (new or resumed)
    resuming_final: bool = False


@dataclass
class Session:
    """What lives through a whole run: the model (rebuilt after a quota wait), the tools and the quota waiter."""

    plan: RunPlan
    verbosity: LogLevel
    tools: list[Tool]
    waiter: QuotaWaiter | None
    model: Model

    @classmethod
    def start(cls, plan: RunPlan, verbosity: LogLevel, waiter: QuotaWaiter | None) -> Session:
        tools = default_tools(plan.settings)
        attach_usage_log(tools, plan.run_dir / USAGE_FILE)
        return cls(plan, verbosity, tools, waiter, _new_model(plan))

    def renew_model(self) -> None:
        """A new model, which does not remember that the quotas of the old one were gone."""
        self.model = _new_model(self.plan)


def _new_model(plan: RunPlan) -> Model:
    return BudgetedModel(plan.settings, plan.run_dir / USAGE_FILE)


def select_questions(questions: Sequence[Question], task_ids: Sequence[str]) -> list[Question]:
    """The questions named by ``task_ids``, in that order and without repeats; all of them when none is named."""
    if not task_ids:
        return list(questions)
    by_id = {question.task_id: question for question in questions}
    wanted = list(dict.fromkeys(task_ids))
    unknown = [task_id for task_id in wanted if task_id not in by_id]
    if unknown:
        raise UnknownTaskError(f"unknown task id(s): {', '.join(unknown)}")
    return [by_id[task_id] for task_id in wanted]


def reset_tools(tools: Sequence[Tool]) -> None:
    """Call ``reset_question()`` on every tool that has one (a tool that counts its calls per question starts over)."""
    for tool in tools:
        reset = getattr(tool, "reset_question", None)
        if callable(reset):
            reset()


def attach_usage_log(tools: Sequence[Tool], path: Path) -> None:
    """Point every tool that logs its own model use (it has ``set_usage_log``) at the usage log of the run."""
    for tool in tools:
        attach = getattr(tool, "set_usage_log", None)
        if callable(attach):
            attach(path)


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _tool_names(settings: Settings) -> tuple[str, ...]:
    """The names of the tools the agent has with these settings and keys (``run.json`` records them)."""
    return tuple(tool.name for tool in default_tools(settings))


def _seal(plan: RunPlan, now: Callable[[], datetime]) -> None:
    """At the end of a session of a final run, record the hash of what the answers file holds."""
    if plan.final_info is None:
        return
    try:
        seal_answers(plan.run_dir, now())
    except (FinalRunError, OSError) as exc:
        complain(f"warning: could not seal the answers in run.json ({exc}); submit will refuse the run until it is")


def _answer_session(
    plan: RunPlan, verbosity: LogLevel, waiter: QuotaWaiter | None, now: Callable[[], datetime]
) -> tuple[list[Attempt], int]:
    """One session: record the resume of a final run, answer, and seal the answers however it ends."""
    if plan.resuming_final:
        record_resume(plan.run_dir, now(), tool_names=_tool_names(plan.settings), model_id=plan.settings.model_id)
    try:
        return _answer_all(plan, Session.start(plan, verbosity, waiter))
    finally:
        _seal(plan, now)


def main(
    argv: list[str] | None = None,
    *,
    git: GitRunner | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _local_now,
) -> int:
    """Run the CLI with ``argv`` (default: ``sys.argv[1:]``) and return the process exit code.

    ``git`` (default: ``git`` in the repository), ``clock``, ``sleep`` and ``now`` are injectable for tests.
    """
    args = parse_args(argv)
    configure_output(args.verbose)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # litellm would download its price list
    try:
        plan = _plan(args, git, now)
    except (ConfigError, ApiError, RunDirError, UnknownTaskError, FinalRunError) as exc:
        complain(f"error: {exc}")
        return EXIT_USAGE
    say(f"Run folder: {plan.run_dir}")
    if plan.final_info is not None:
        say(f"Final run on tag {plan.final_info.get('tag')} (commit {str(plan.final_info.get('commit'))[:7]}).")
    if not plan.pending:
        say(f"Nothing to do: {plan.skipped} already answered.")
        return EXIT_OK
    say(f"Answering {len(plan.pending)} question(s); {plan.skipped} already answered.")
    waiter = QuotaWaiter(args.max_wait_hours, clock=clock, sleep=sleep) if args.wait_for_quota else None
    first_usage_line = len(read_usage(plan.run_dir / USAGE_FILE))
    started = clock()
    try:
        attempts, deferred = _answer_session(plan, LogLevel.INFO if args.verbose else LogLevel.OFF, waiter, now)
    except FinalRunError as exc:
        complain(f"error: {exc}")
        return EXIT_USAGE
    except KeyboardInterrupt:
        complain(f"Interrupted: the attempts finished so far are saved in {plan.run_dir / ANSWERS_FILE}")
        return EXIT_INTERRUPTED
    except OSError as exc:
        complain(f"error: could not save the run files: {exc}")
        return EXIT_FAILED
    done = print_summary(
        plan.run_dir,
        pending=len(plan.pending),
        skipped=plan.skipped,
        attempts=attempts,
        deferred=deferred,
        seconds=clock() - started,
        first_usage_line=first_usage_line,
        waiter=waiter,
    )
    return EXIT_OK if done == len(plan.pending) else EXIT_FAILED


def _check_final(
    args: argparse.Namespace, info: Mapping[str, Any] | None, resuming: bool, git: GitRunner | None
) -> GitState | None:
    """The git state of a new final run; for a final run being resumed, check that the code is the same.

    A run that was started as final is held to it, with or without ``--final``. Returns None unless a new
    final run starts.
    """
    if info is not None and is_final(info):
        verify_unchanged(info, git or git_runner(REPO_ROOT))
        return None
    if not args.final:
        return None
    if resuming:
        raise FinalRunError("--final cannot adopt a run that is not a final run: start a new one")
    return require_release_state(git or git_runner(REPO_ROOT))


def _plan(args: argparse.Namespace, git: GitRunner | None, now: Callable[[], datetime]) -> RunPlan:
    """Load the questions, pick the run folder and work out what is left to answer."""
    settings = load_settings()
    ensure_no_answer_key(settings)
    resume_dir = resolve_run_dir(settings.runs_dir, args.run_dir) if args.run_dir else None
    info = read_run_info(resume_dir) if resume_dir is not None else None
    release = _check_final(args, info, resume_dir is not None, git)
    client = ScoringClient(settings)
    chosen = select_questions(client.get_questions(refresh=args.refresh), args.task or [])
    run_dir = resume_dir if resume_dir is not None else new_run_dir(settings.runs_dir, now())
    if release is not None:
        info = start_info(settings, release, now(), _tool_names(settings))
        write_run_info(run_dir, info)
    done = load_answers(run_dir / ANSWERS_FILE)
    pending = [question for question in chosen if not is_answered(done.get(question.task_id))]
    skipped = len(chosen) - len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]
    return RunPlan(
        settings=settings,
        client=client,
        run_dir=run_dir,
        pending=tuple(pending),
        skipped=skipped,
        defer_missing_attachments=args.defer_missing_attachments,
        final_info=info if is_final(info) else None,
        resuming_final=resume_dir is not None and is_final(info),
    )


def _answer_all(plan: RunPlan, session: Session) -> tuple[list[Attempt], int]:
    """Answer the pending questions one by one, stopping early when no model is left.

    Returns the attempts made and how many questions were deferred (not attempted: no attachment).
    """
    attempts: list[Attempt] = []
    deferred = 0
    for index, question in enumerate(plan.pending, start=1):
        attempt = _settle(session, question)
        if attempt is None:
            deferred += 1
            say(f"[{index}/{len(plan.pending)}] {short_id(question.task_id)} DEFERRED (attachment not available)")
            continue
        attempts.append(attempt)
        say(progress_line(index, len(plan.pending), question, attempt))
        if is_fatal(attempt.result):
            say(f"Stopping: no model can answer anymore. Resume later with --run-dir {plan.run_dir.name}")
            break
    return attempts, deferred


def _settle(session: Session, question: Question) -> Attempt | None:
    """Answer ``question`` and save the outcome; None when it is deferred for want of its attachment.

    With ``--wait-for-quota`` an attempt that failed for want of quota is not saved: the run waits, gets a new
    model and asks again, until the question is answered or the waiting time is used up.
    """
    plan = session.plan
    attachment = plan.client.download_file(question) if question.file_name else None
    if question.file_name and attachment is None and plan.defer_missing_attachments:
        return None
    while True:
        attempt = _attempt(session, question, attachment)
        error = attempt.result.error
        if session.waiter is not None and is_quota_exhausted(error) and session.waiter.wait(error or ""):
            session.renew_model()
            continue
        _save(plan, question, attempt, attachment)
        return attempt


def _attempt(session: Session, question: Question, attachment: Path | None) -> Attempt:
    """Answer one question with a fresh agent (nothing is saved here)."""
    plan = session.plan
    usage_path = plan.run_dir / USAGE_FILE
    first_line = len(read_usage(usage_path))
    reset_tools(session.tools)
    agent = build_agent(plan.settings, tools=session.tools, model=session.model, verbosity=session.verbosity)
    result = answer_question(agent, question, attachment)
    usage = summarize_usage(read_usage(usage_path)[first_line:])
    return Attempt(result=result, usage=usage, steps=tuple(agent.memory.steps))


def _save(plan: RunPlan, question: Question, attempt: Attempt, attachment: Path | None) -> None:
    """Keep the answer line and the trace of an attempt."""
    append_record(plan.run_dir / ANSWERS_FILE, dataclasses.asdict(attempt.result))
    trace = format_trace(question, attempt.result, list(attempt.steps), attachment=attachment, usage=attempt.usage)
    write_text(trace_path(plan.run_dir, question.task_id), trace)


if __name__ == "__main__":
    raise SystemExit(main())
