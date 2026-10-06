"""Tests for the step callback that shortens the observations kept in the agent's memory."""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace
from typing import Any

import pytest
from smolagents import CodeAgent, Model
from smolagents.memory import ActionStep, CallbackRegistry, FinalAnswerStep, PlanningStep, TaskStep, ToolCall
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import LogLevel, Timing

from gaia_agent.memory import TRIM_MARKER, make_trim_callback, trim_text

LAST_LIMIT = 300
OLD_LIMIT = 60


def make_action_step(number: int, observations: str | None, code: str = "print('hi')") -> ActionStep:
    """Build a realistic action step: generated code, the tool call that ran it and its observations."""
    return ActionStep(
        step_number=number,
        timing=Timing(start_time=0.0, end_time=1.0),
        model_output=f"Thought: step {number}\n<code>\n{code}\n</code>",
        code_action=code,
        tool_calls=[ToolCall(name="python_interpreter", arguments=code, id=f"call_{number}")],
        observations=observations,
    )


def make_agent(*steps: Any) -> SimpleNamespace:
    """A stand-in for the agent: the callback only reads ``agent.memory.steps``."""
    return SimpleNamespace(memory=SimpleNamespace(steps=list(steps)))


CODE_STEPS = [
    "Thought: first\n<code>\nprint('Q' * 3000)\n</code>",
    "Thought: second\n<code>\nprint('R' * 3000)\n</code>",
    "Thought: done\n<code>\nfinal_answer('done')\n</code>",
]


class ScriptedModel(Model):
    """A model that replays canned outputs and remembers the prompts it was given."""

    def __init__(self, outputs: list[str]) -> None:
        super().__init__(model_id="scripted")
        self._outputs = list(outputs)
        self.prompts: list[list[ChatMessage]] = []

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self.prompts.append(list(messages))
        return ChatMessage(role=MessageRole.ASSISTANT, content=self._outputs.pop(0))


def prompt_text(messages: list[ChatMessage]) -> str:
    """Concatenate the text parts of a prompt."""
    parts: list[str] = []
    for message in messages:
        if isinstance(message.content, list):
            parts.extend(part["text"] for part in message.content if part.get("type") == "text")
    return "\n".join(parts)


def longest_run(text: str, char: str) -> int:
    """Length of the longest run of ``char`` in ``text``."""
    return max((len(run) for run in re.findall(f"{re.escape(char)}+", text)), default=0)


@pytest.fixture
def callback():
    return make_trim_callback(max_last_obs_chars=LAST_LIMIT, max_old_obs_chars=OLD_LIMIT)


class TestTrimText:
    def test_short_text_is_returned_unchanged(self) -> None:
        assert trim_text("short", 100) == "short"

    def test_text_exactly_at_the_limit_is_kept(self) -> None:
        text = "x" * 50

        assert trim_text(text, 50) == text

    def test_long_text_is_cut_to_the_limit_and_marked(self) -> None:
        text = "".join(chr(ord("a") + i % 26) for i in range(500))

        trimmed = trim_text(text, 100)

        assert len(trimmed) == 100
        assert trimmed.endswith(TRIM_MARKER)
        assert trimmed.startswith(text[:50])

    def test_the_marker_is_literally_trimmed_in_brackets(self) -> None:
        assert TRIM_MARKER == "[...trimmed]"

    def test_trimming_twice_gives_the_same_result(self) -> None:
        once = trim_text("y" * 1000, 120)

        assert trim_text(once, 120) == once

    def test_trimming_a_trimmed_text_with_a_smaller_limit_keeps_a_single_marker(self) -> None:
        trimmed = trim_text(trim_text("z" * 1000, 400), 100)

        assert len(trimmed) == 100
        assert trimmed.count(TRIM_MARKER) == 1

    def test_a_limit_too_small_for_the_marker_cuts_the_text_hard(self) -> None:
        assert trim_text("abcdefghij", 5) == "abcde"

    def test_a_zero_limit_gives_an_empty_text(self) -> None:
        assert trim_text("abc", 0) == ""

    def test_empty_text_stays_empty(self) -> None:
        assert trim_text("", 10) == ""

    def test_a_negative_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            trim_text("abc", -1)


class TestTrimCallback:
    def test_the_latest_observation_may_keep_the_larger_limit(self, callback) -> None:
        step = make_action_step(1, "o" * 1000)

        callback(step, agent=make_agent())

        assert len(step.observations) == LAST_LIMIT
        assert step.observations.endswith(TRIM_MARKER)

    def test_older_observations_are_cut_to_the_smaller_limit(self, callback) -> None:
        old = make_action_step(1, "a" * 1000)
        current = make_action_step(2, "b" * 1000)

        callback(current, agent=make_agent(old))

        assert len(old.observations) == OLD_LIMIT
        assert old.observations.endswith(TRIM_MARKER)
        assert len(current.observations) == LAST_LIMIT

    def test_short_observations_are_left_alone(self, callback) -> None:
        old = make_action_step(1, "short old")
        current = make_action_step(2, "short new")

        callback(current, agent=make_agent(old))

        assert (old.observations, current.observations) == ("short old", "short new")

    def test_the_previous_latest_observation_becomes_old_on_the_next_step(self, callback) -> None:
        first = make_action_step(1, "f" * 1000)
        second = make_action_step(2, "s" * 1000)
        memory_agent = make_agent()

        callback(first, agent=memory_agent)
        assert len(first.observations) == LAST_LIMIT
        memory_agent.memory.steps.append(first)
        callback(second, agent=memory_agent)

        assert len(first.observations) == OLD_LIMIT
        assert len(second.observations) == LAST_LIMIT

    def test_generated_code_and_tool_calls_are_never_touched(self, callback) -> None:
        long_code = "x = 1\n" * 400
        old = make_action_step(1, "o" * 1000, code=long_code)
        current = make_action_step(2, "c" * 1000, code=long_code)
        expected = copy.deepcopy((old, current))

        callback(current, agent=make_agent(old))

        for step, before in zip((old, current), expected, strict=True):
            assert step.code_action == before.code_action
            assert step.model_output == before.model_output
            assert step.tool_calls == before.tool_calls

    def test_tool_calls_are_kept_unless_the_option_is_on(self, callback) -> None:
        old = make_action_step(1, "o")
        current = make_action_step(2, "c")

        callback(current, agent=make_agent(old))

        assert old.tool_calls is not None
        assert current.tool_calls is not None

    def test_the_option_drops_the_tool_calls_of_older_steps_only(self) -> None:
        drop = make_trim_callback(LAST_LIMIT, OLD_LIMIT, drop_old_tool_calls=True)
        old = make_action_step(1, "o", code="x = 1")
        current = make_action_step(2, "c", code="y = 2")
        code_before = (old.code_action, old.model_output)

        drop(current, agent=make_agent(old))

        assert old.tool_calls is None
        assert current.tool_calls is not None
        assert (old.code_action, old.model_output) == code_before

    def test_the_task_and_the_planning_steps_are_never_touched(self, callback) -> None:
        task = TaskStep(task="t" * 2000)
        plan = PlanningStep(
            model_input_messages=[],
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="p"),
            plan="p" * 2000,
            timing=Timing(start_time=0.0),
        )
        current = make_action_step(1, "c" * 1000)

        callback(current, agent=make_agent(task, plan))

        assert (len(task.task), len(plan.plan)) == (2000, 2000)

    def test_steps_without_observations_are_skipped(self, callback) -> None:
        silent = make_action_step(1, None)
        current = make_action_step(2, None)

        callback(current, agent=make_agent(silent))

        assert (silent.observations, current.observations) == (None, None)

    def test_it_ignores_steps_that_are_not_action_steps(self, callback) -> None:
        old = make_action_step(1, "o" * 1000)
        final = FinalAnswerStep(output="answer")

        callback(final, agent=make_agent(old))

        assert old.observations == "o" * 1000

    def test_it_works_without_an_agent(self, callback) -> None:
        step = make_action_step(1, "o" * 1000)

        callback(step)

        assert len(step.observations) == LAST_LIMIT

    def test_calling_it_again_changes_nothing(self, callback) -> None:
        old = make_action_step(1, "a" * 1000)
        current = make_action_step(2, "b" * 1000)
        callback(current, agent=make_agent(old))
        snapshot = (old.observations, current.observations)

        callback(current, agent=make_agent(old))

        assert (old.observations, current.observations) == snapshot

    def test_the_defaults_are_6000_and_700_characters(self) -> None:
        default = make_trim_callback()
        old = make_action_step(1, "a" * 10_000)
        current = make_action_step(2, "b" * 10_000)

        default(current, agent=make_agent(old))

        assert (len(old.observations), len(current.observations)) == (700, 6000)

    @pytest.mark.parametrize(
        ("last", "old"),
        [(0, 700), (6000, 0), (-5, 700), (6000, -1)],
    )
    def test_non_positive_limits_are_rejected(self, last: int, old: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            make_trim_callback(max_last_obs_chars=last, max_old_obs_chars=old)


class TestSmolagentsIntegration:
    """The callback must fit the way the installed smolagents version calls its step callbacks."""

    def test_the_registry_calls_it_with_the_step_and_the_agent(self, callback) -> None:
        old = make_action_step(1, "a" * 1000)
        current = make_action_step(2, "b" * 1000)
        registry = CallbackRegistry()
        registry.register(ActionStep, callback)

        registry.callback(current, agent=make_agent(old))

        assert (len(old.observations), len(current.observations)) == (OLD_LIMIT, LAST_LIMIT)

    def test_a_code_agent_run_sends_trimmed_history_to_the_model(self) -> None:
        model = ScriptedModel(CODE_STEPS)
        agent = CodeAgent(
            tools=[],
            model=model,
            step_callbacks=[make_trim_callback(max_last_obs_chars=1000, max_old_obs_chars=100)],
            max_steps=5,
            verbosity_level=LogLevel.OFF,
        )

        answer = agent.run("Say done.")

        assert answer == "done"
        third_prompt = prompt_text(model.prompts[2])
        assert longest_run(third_prompt, "Q") <= 100
        assert longest_run(third_prompt, "R") <= 1000
        assert longest_run(third_prompt, "R") > 100
        assert TRIM_MARKER in third_prompt

    @pytest.mark.parametrize(("drop", "blocks"), [(False, 2), (True, 1)])
    def test_dropping_old_tool_calls_removes_the_repeated_code_from_the_prompt(self, drop: bool, blocks: int) -> None:
        model = ScriptedModel(CODE_STEPS)
        agent = CodeAgent(
            tools=[],
            model=model,
            step_callbacks=[make_trim_callback(drop_old_tool_calls=drop)],
            max_steps=5,
            verbosity_level=LogLevel.OFF,
        )

        agent.run("Say done.")

        # In the third prompt steps 1 and 2 are remembered; only the newest one keeps its "Calling tools" block.
        assert prompt_text(model.prompts[2]).count("Calling tools:") == blocks
