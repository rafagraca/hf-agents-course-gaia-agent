"""Tests for ``gaia_agent.trace.format_trace``: the short markdown trace saved for every attempt."""

from __future__ import annotations

from pathlib import Path

from agent_fakes import AUDIO, PLAIN, SHEET
from smolagents.memory import ActionStep, TaskStep
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentExecutionError, AgentMaxStepsError

from gaia_agent.agent import AnswerResult
from gaia_agent.runs import UsageStats
from gaia_agent.trace import MAX_OBSERVATION_CHARS, MAX_OUTPUT_CHARS, format_trace

QUIET = AgentLogger(level=LogLevel.OFF)
DONE = AnswerResult(task_id=PLAIN.task_id, answer="42", raw_answer="42.", steps=2, seconds=12.34, model="fake/model")


def step(
    number: int, output: str = "Thought: go.\n<code>\nx = 1\n</code>", observations: str | None = "Out: 1", **extra
):
    return ActionStep(
        step_number=number,
        timing=Timing(start_time=100.0, end_time=103.5),
        model_output=output,
        observations=observations,
        token_usage=TokenUsage(input_tokens=1500, output_tokens=200),
        **extra,
    )


def test_the_header_names_the_task_question_answer_and_run_details() -> None:
    usage = UsageStats(calls=2, max_input=1500, mean_input=1400, total_input=2800, total_output=400, waited_seconds=0)

    text = format_trace(PLAIN, DONE, [TaskStep(task=PLAIN.question), step(1)], usage=usage)

    assert text.startswith(f"# Trace of task {PLAIN.task_id}\n")
    assert PLAIN.question in text
    assert "- Answer: `42` (raw: `42.`)" in text
    assert "- Error: none" in text
    assert "- Model: fake/model; 2 steps; 12.3 s" in text
    assert f"- Usage: {usage.describe()}" in text
    assert "- Attachment: none" in text


def test_each_action_step_shows_its_timing_tokens_output_and_observation() -> None:
    text = format_trace(PLAIN, DONE, [step(1, "Thought: search.", "Out: found it"), step(2)])

    assert "## Step 1 (3.5 s, 1500 in / 200 out tokens)" in text
    assert "Thought: search." in text
    assert "Out: found it" in text
    assert "## Step 2" in text


def test_attachments_are_described_whether_present_or_missing(tmp_path: Path) -> None:
    present = format_trace(SHEET, DONE, [], attachment=tmp_path / "task-sheet.xlsx")
    missing = format_trace(AUDIO, DONE, [])

    assert f"- Attachment: {(tmp_path / 'task-sheet.xlsx').as_posix()}" in present
    assert f"- Attachment: {AUDIO.file_name} (not available)" in missing


def test_errors_and_the_step_limit_are_reported() -> None:
    failed = AnswerResult(
        task_id=PLAIN.task_id, answer="", raw_answer="", steps=2, seconds=1.0, error="RuntimeError: x"
    )
    steps = [step(1, error=AgentExecutionError("name 'y' is not defined", QUIET)), step(2, observations=None)]
    steps.append(step(3, output=None, observations=None, error=AgentMaxStepsError("Reached max steps.", QUIET)))

    text = format_trace(PLAIN, failed, steps)

    assert "- Error: RuntimeError: x" in text
    assert "Error: name 'y' is not defined" in text
    assert "- Stopped at the step limit: the answer was forced." in text


def test_long_outputs_and_observations_are_cut_with_a_note() -> None:
    output, observation = "a" * (MAX_OUTPUT_CHARS + 50), "b" * (MAX_OBSERVATION_CHARS + 70)

    text = format_trace(PLAIN, DONE, [step(1, output, observation)])

    assert "a" * MAX_OUTPUT_CHARS + "\n[truncated: " in text
    assert "a" * (MAX_OUTPUT_CHARS + 1) not in text
    assert f"[truncated: {MAX_OBSERVATION_CHARS + 70} chars total]" in text


def test_code_fences_stay_closed_when_the_content_has_backticks() -> None:
    text = format_trace(PLAIN, DONE, [step(1, "Thought: x\n```python\nprint(1)\n```", "````odd````")])

    assert "````\nThought: x" in text
    assert "`````\n````odd````\n`````" in text


def test_steps_without_timing_or_usage_still_render() -> None:
    bare = ActionStep(step_number=1, timing=Timing(start_time=5.0), model_output="Thought: hm.")

    assert "## Step 1 (? s, ? in / ? out tokens)" in format_trace(PLAIN, DONE, [bare])
