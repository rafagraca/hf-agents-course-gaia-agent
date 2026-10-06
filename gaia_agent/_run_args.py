"""The command line of ``python -m gaia_agent.run``."""

from __future__ import annotations

import argparse

from gaia_agent._quota import DEFAULT_MAX_WAIT_HOURS

__all__ = ["parse_args"]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m gaia_agent.run",
        description="Answer the exam questions with the agent and save the answers in a run folder.",
    )
    parser.add_argument("--task", action="append", metavar="TASK_ID", help="answer only this task (repeatable)")
    parser.add_argument("--limit", type=_positive_int, metavar="N", help="attempt at most N questions")
    parser.add_argument("--run-dir", metavar="DIR", help="resume this run (a folder name in <data>/runs or a path)")
    parser.add_argument("--refresh", action="store_true", help="download the question list again")
    parser.add_argument("--verbose", action="store_true", help="show every step of the agent")
    parser.add_argument(
        "--final",
        action="store_true",
        help="the run whose answers will be submitted: needs a clean working tree and HEAD on a tag",
    )
    parser.add_argument(
        "--wait-for-quota",
        action="store_true",
        help="when every model is out of daily quota, wait (at most 15 minutes at a time) and try again",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=_positive_float,
        default=DEFAULT_MAX_WAIT_HOURS,
        metavar="H",
        help=f"stop waiting for the quota after this many hours in all (default {DEFAULT_MAX_WAIT_HOURS:g})",
    )
    parser.add_argument(
        "--defer-missing-attachments",
        action="store_true",
        help="leave a question whose attachment cannot be fetched for a later --run-dir run",
    )
    return parser.parse_args(argv)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text!r}")
    return value
