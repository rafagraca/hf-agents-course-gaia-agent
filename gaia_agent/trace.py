"""A short markdown trace of one attempt: the question, the outcome and what happened at each step.

Traces are written to the run folder in DATA (they quote the benchmark question), never to the
repository. Long model outputs and observations are cut, so a trace stays easy to skim.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from smolagents.memory import ActionStep
from smolagents.utils import AgentMaxStepsError

from gaia_agent._secrets import mask_secrets

if TYPE_CHECKING:
    from gaia_agent.agent import AnswerResult
    from gaia_agent.api import Question
    from gaia_agent.runs import UsageStats

__all__ = ["MAX_OBSERVATION_CHARS", "MAX_OUTPUT_CHARS", "format_trace"]

MAX_OUTPUT_CHARS = 2000
MAX_OBSERVATION_CHARS = 1500
MAX_ERROR_CHARS = 500

_BACKTICK_RUN = re.compile(r"`+")


def format_trace(
    question: Question,
    result: AnswerResult,
    steps: Sequence[Any],
    *,
    attachment: Path | None = None,
    usage: UsageStats | None = None,
) -> str:
    """Render the trace of ``result``; ``steps`` is the agent memory (only its action steps are shown).

    No API key or token appears in it, whatever the model printed or the provider quoted in an error.
    """
    action_steps = [step for step in steps if isinstance(step, ActionStep)]
    lines = [f"# Trace of task {question.task_id}", "", *_header(question, result, attachment, usage)]
    if any(isinstance(step.error, AgentMaxStepsError) for step in action_steps):
        lines.append("- Stopped at the step limit: the answer was forced.")
    for step in action_steps:
        lines.extend(["", *_step_lines(step)])
    return mask_secrets("\n".join(lines) + "\n")


def _header(question: Question, result: AnswerResult, attachment: Path | None, usage: UsageStats | None) -> list[str]:
    if attachment is not None:
        attached = attachment.as_posix()
    elif question.file_name:
        attached = f"{question.file_name} (not available)"
    else:
        attached = "none"
    lines = [
        f"- Question (level {question.level or '?'}): {question.question}",
        f"- Attachment: {attached}",
        f"- Answer: `{result.answer}` (raw: `{result.raw_answer}`)",
        f"- Error: {result.error or 'none'}",
        f"- Model: {result.model or '?'}; {result.steps} steps; {result.seconds:.1f} s",
    ]
    if usage is not None:
        lines.append(f"- Usage: {usage.describe()}")
    return lines


def _step_lines(step: ActionStep) -> list[str]:
    duration = step.timing.duration if step.timing is not None else None
    seconds = "?" if duration is None else f"{duration:.1f}"
    usage = step.token_usage
    tokens = "? in / ? out" if usage is None else f"{usage.input_tokens} in / {usage.output_tokens} out"
    lines = [f"## Step {step.step_number} ({seconds} s, {tokens} tokens)"]
    if step.model_output:
        lines.extend(["", "Model output:", _fenced(_cut(str(step.model_output), MAX_OUTPUT_CHARS))])
    if step.observations:
        lines.extend(["", "Observation:", _fenced(_cut(step.observations, MAX_OBSERVATION_CHARS))])
    if step.error is not None:
        lines.extend(["", f"Error: {_cut(str(step.error), MAX_ERROR_CHARS)}"])
    return lines


def _cut(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters, with a note giving the full length."""
    return text if len(text) <= limit else f"{text[:limit]}\n[truncated: {len(text)} chars total]"


def _fenced(text: str) -> str:
    """``text`` in a code fence longer than any run of backticks inside it."""
    longest = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"
