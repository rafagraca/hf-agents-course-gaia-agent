"""What a run tells the console: one line per question and the totals at the end."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gaia_agent._cli import say, short_id
from gaia_agent._quota import QuotaWaiter, describe_duration
from gaia_agent.agent import AnswerResult
from gaia_agent.api import Question
from gaia_agent.runs import ANSWERS_FILE, USAGE_FILE, UsageStats, is_answered, read_usage, summarize_usage

__all__ = ["Attempt", "answered", "print_summary", "progress_line"]


@dataclass(frozen=True)
class Attempt:
    """One answered (or failed) question and the model usage it cost; ``steps`` are the agent's memory steps."""

    result: AnswerResult
    usage: UsageStats
    steps: tuple[Any, ...] = ()


def answered(result: AnswerResult) -> bool:
    """True when ``result`` holds an answer and no error."""
    return is_answered(dataclasses.asdict(result))


def progress_line(index: int, total: int, question: Question, attempt: Attempt) -> str:
    result = attempt.result
    if answered(result):
        status = "answered"
    elif result.error is not None:
        status = f"FAILED ({result.error})"
    else:
        status = "FAILED (empty answer)"
    return (
        f"[{index}/{total}] {short_id(question.task_id)} {status}: {result.steps} steps, "
        f"{result.seconds:.1f} s; {attempt.usage.describe()}"
    )


def print_summary(
    run_dir: Path,
    *,
    pending: int,
    skipped: int,
    attempts: Sequence[Attempt],
    deferred: int,
    seconds: float,
    first_usage_line: int,
    waiter: QuotaWaiter | None,
) -> int:
    """Print the totals of this session and return how many questions it answered."""
    done = sum(answered(attempt.result) for attempt in attempts)
    usage = summarize_usage(read_usage(run_dir / USAGE_FILE)[first_usage_line:])
    say(
        f"Done in {seconds:.0f} s: {done} answered, {len(attempts) - done} failed, "
        f"{pending - len(attempts) - deferred} not attempted, {skipped} skipped (already answered)."
    )
    if deferred:
        say(f"{deferred} deferred (no attachment): resume with --run-dir {run_dir.name} when it can be had.")
    if waiter is not None and waiter.pauses:
        say(f"The run waited {describe_duration(waiter.waited_seconds)} for the quota ({waiter.pauses} pause(s)).")
    say(f"Usage: {usage.describe()}")
    say(f"Answers: {run_dir / ANSWERS_FILE}")
    return done
