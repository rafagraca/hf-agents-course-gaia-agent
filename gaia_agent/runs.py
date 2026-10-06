"""The files of a run: its folder, the answers file and the usage log written by the model.

A run lives in ``<data>/runs/<YYYYMMDD-HHMMSS>``, inside the DATA folder and never inside the
repository, because it holds benchmark answers. Inside it:

* ``answers.jsonl``: one JSON line per attempt (``task_id``, ``answer``, ``raw_answer``, ``steps``,
  ``seconds``, ``error``, ``model``); when a task appears more than once, the last line wins;
* ``usage.jsonl``: one line per model call, written by ``BudgetedModel`` and by the tools that call a model
  themselves (``deep_search``);
* ``run.json``: what a final run ran (see ``gaia_agent._final``), and ``submission.json`` once it is submitted;
* ``traces/<task_id>.md``: a short trace of each attempt.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "ANSWERS_FILE",
    "SUBMISSION_FILE",
    "USAGE_FILE",
    "RunDirError",
    "UsageStats",
    "append_record",
    "is_answered",
    "latest_run_dir",
    "load_answers",
    "new_run_dir",
    "pick_run_dir",
    "read_usage",
    "resolve_run_dir",
    "summarize_usage",
    "trace_path",
    "write_text",
]

logger = logging.getLogger(__name__)

ANSWERS_FILE = "answers.jsonl"
USAGE_FILE = "usage.jsonl"
SUBMISSION_FILE = "submission.json"
TRACES_DIR = "traces"
RUN_NAME_FORMAT = "%Y%m%d-%H%M%S"
MAX_SAME_SECOND_RUNS = 100

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_RUN_NAME = re.compile(r"(\d{8}-\d{6})(?:-(\d+))?")
_TOOL_MODEL = re.compile(r" \([a-z_]+\)$")  # "openai/gpt-oss-20b (deep_search)"


class RunDirError(ValueError):
    """A run folder cannot be created or used."""


# --- run folders -------------------------------------------------------------------------------------------------


def new_run_dir(runs_dir: Path, now: datetime) -> Path:
    """Create and return ``runs_dir/<YYYYMMDD-HHMMSS>`` for ``now``, adding ``-2``, ``-3``... if it exists."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = now.strftime(RUN_NAME_FORMAT)
    for number in range(1, MAX_SAME_SECOND_RUNS + 1):
        folder = runs_dir / (stem if number == 1 else f"{stem}-{number}")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise RunDirError(f"too many runs started at {stem} in {runs_dir}")


def resolve_run_dir(runs_dir: Path, value: str | Path) -> Path:
    """The existing run folder named by ``value``: a folder name inside ``runs_dir`` or a path to one.

    Raises ``RunDirError`` when it does not exist or is not strictly inside ``runs_dir``, so that
    answers can never be written to (or read from) anywhere else.
    """
    root = runs_dir.resolve()
    path = Path(value)
    folder = (path if path.is_absolute() else root / path).resolve()
    if folder == root or not folder.is_relative_to(root):
        raise RunDirError(f"the run folder must be inside {root}: {folder}")
    if not folder.is_dir():
        raise RunDirError(f"there is no run folder at {folder}")
    return folder


def _run_order(folder: Path) -> tuple[str, int]:
    """How runs are ordered: by the second they started in, then by the number after it (``-10`` follows ``-9``)."""
    found = _RUN_NAME.fullmatch(folder.name)
    return (found.group(1), int(found.group(2) or 1)) if found else (folder.name, 1)


def latest_run_dir(runs_dir: Path) -> Path | None:
    """The newest run folder that holds an answers file, or None when there is none.

    Newest means started last: the name is a timestamp, and runs started in the same second carry a number.
    """
    if not runs_dir.is_dir():
        return None
    runs = [folder for folder in runs_dir.iterdir() if (folder / ANSWERS_FILE).is_file()]
    return max(runs, key=_run_order, default=None)


def pick_run_dir(runs_dir: Path, value: str | Path | None) -> Path:
    """The run folder named by ``value`` (see :func:`resolve_run_dir`) or, without one, the latest run."""
    if value is not None:
        return resolve_run_dir(runs_dir, value)
    latest = latest_run_dir(runs_dir)
    if latest is None:
        raise RunDirError(f"there is no run with answers yet in {runs_dir}")
    return latest


def trace_path(run_dir: Path, task_id: str) -> Path:
    """Where the trace of ``task_id`` goes; characters unsafe in a file name become ``_``."""
    return run_dir / TRACES_DIR / f"{_UNSAFE_NAME_CHARS.sub('_', task_id)}.md"


def write_text(path: Path, text: str) -> None:
    """Write ``text`` as UTF-8 with ``\\n`` line endings, creating the parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


# --- answers file ------------------------------------------------------------------------------------------------


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append ``record`` to the JSON Lines file ``path`` (created with its folder when missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _json_lines(path: Path) -> list[dict[str, Any]]:
    """The JSON objects of a JSON Lines file, in order; blank lines are ignored, bad ones skipped with a warning."""
    if not path.is_file():
        return []
    items: list[dict[str, Any]] = []
    # Only "\n" ends a record: json.dumps leaves U+2028 and U+0085 inside a string, and str.splitlines would cut
    # there. A byte that is not UTF-8 spoils its own line (skipped below), not the whole file.
    text = path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            item = None
        if isinstance(item, dict):
            items.append(item)
        else:
            logger.warning("Skipping line %d of %s: not a JSON object.", number, path.name)
    return items


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    """The records of an answers file by task id, in first-seen order; the last record of a task wins."""
    answers: dict[str, dict[str, Any]] = {}
    for item in _json_lines(path):
        task_id = item.get("task_id")
        if isinstance(task_id, str) and task_id:
            answers[task_id] = item
        else:
            logger.warning("Skipping line of %s without a task_id.", path.name)
    return answers


def is_answered(record: Mapping[str, Any] | None) -> bool:
    """True when ``record`` holds a non-empty text answer and no error (anything else is tried again)."""
    if record is None or record.get("error") is not None:
        return False
    answer = record.get("answer")
    return isinstance(answer, str) and answer != ""


# --- usage log ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageStats:
    """Model calls and tokens over some lines of the usage log (input figures only count calls that report them).

    The calls a tool makes to its own model (``deep_search``) are counted apart, in ``tool_calls`` and
    ``tool_input``: they do not go through the prompt limit of the agent, so they must not hide its maximum.
    """

    calls: int
    max_input: int
    mean_input: int
    total_input: int
    total_output: int
    waited_seconds: float
    tool_calls: int = 0
    tool_input: int = 0

    def describe(self) -> str:
        """One short line for the console."""
        line = (
            f"{self.calls} model calls, input tokens max {self.max_input} / mean {self.mean_input}, "
            f"output tokens {self.total_output}, waited {self.waited_seconds:.0f} s"
        )
        if self.tool_calls:
            line += f"; {self.tool_calls} tool model calls, input tokens {self.tool_input}"
        return line


def read_usage(path: Path) -> list[dict[str, Any]]:
    """The lines of a usage log, in order (an absent log is empty)."""
    return _json_lines(path)


def _numbers(lines: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [value for line in lines if isinstance(value := line.get(key), int | float) and not isinstance(value, bool)]


def _is_tool_line(line: Mapping[str, Any]) -> bool:
    """A line a tool wrote for its own model: its model reads ``<model id> (<tool name>)``."""
    return bool(_TOOL_MODEL.search(str(line.get("model", ""))))


def summarize_usage(lines: Sequence[Mapping[str, Any]]) -> UsageStats:
    """Totals and extremes over ``lines`` of a usage log, the calls of the tools apart."""
    agent_lines = [line for line in lines if not _is_tool_line(line)]
    inputs = _numbers(agent_lines, "input_tokens")
    return UsageStats(
        calls=len(agent_lines),
        max_input=int(max(inputs, default=0)),
        mean_input=round(sum(inputs) / len(inputs)) if inputs else 0,
        total_input=int(sum(inputs)),
        total_output=int(sum(_numbers(agent_lines, "output_tokens"))),
        waited_seconds=round(sum(_numbers(lines, "waited_seconds")), 3),
        tool_calls=len(lines) - len(agent_lines),
        tool_input=int(sum(_numbers([line for line in lines if _is_tool_line(line)], "input_tokens"))),
    )
