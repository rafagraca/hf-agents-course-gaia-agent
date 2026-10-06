"""Tests for the memory limits of the step callback: errors, parsing errors and the size of the next prompt."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from smolagents.memory import ActionStep, SystemPromptStep, TaskStep, ToolCall
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel, Timing
from smolagents.utils import AgentExecutionError, AgentParsingError

from gaia_agent.memory import TRIM_MARKER, make_trim_callback, trim_middle
from gaia_agent.tokens import estimate_tokens

LOGGER = AgentLogger(level=LogLevel.OFF)


def action_step(
    number: int, *, observations: str | None = None, error: Exception | None = None, reply: str | None = None
) -> ActionStep:
    """A realistic action step; ``reply`` is the model output (the assistant turn) when it differs from the default."""
    code = "print('hi')"
    text = reply if reply is not None else f"Thought: step {number}\n<code>\n{code}\n</code>"
    return ActionStep(
        step_number=number,
        timing=Timing(start_time=0.0, end_time=1.0),
        model_output=text,
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
        code_action=code,
        tool_calls=[ToolCall(name="python_interpreter", arguments=code, id=f"call_{number}")],
        observations=observations,
        error=error,
    )


def agent_with(*steps: Any, system: str | None = None, task: str | None = None) -> SimpleNamespace:
    """A stand-in for the agent: its memory holds an optional system prompt and task, and the given steps."""
    remembered = ([TaskStep(task=task)] if task is not None else []) + list(steps)
    system_step = SystemPromptStep(system_prompt=system) if system is not None else None
    return SimpleNamespace(memory=SimpleNamespace(steps=remembered, system_prompt=system_step))


def next_prompt(agent: SimpleNamespace, current: ActionStep) -> list[ChatMessage]:
    """What the next model call would be sent: system prompt, remembered steps and the step that just ran."""
    messages = list(agent.memory.system_prompt.to_messages()) if agent.memory.system_prompt else []
    for step in (*agent.memory.steps, current):
        messages.extend(step.to_messages())
    return messages


def execution_error(text: str) -> AgentExecutionError:
    return AgentExecutionError(text, LOGGER)


def parsing_error(reply: str) -> AgentParsingError:
    text = (
        "Error in code parsing:\nYour code snippet is invalid, because the regex pattern <code>(.*?)</code> was "
        f"not found in it.\nHere is your code snippet:\n{reply}\nMake sure to provide correct code blobs."
    )
    return AgentParsingError(text, LOGGER)


# --- trim_middle ----------------------------------------------------------------------------------------------------


class TestTrimMiddle:
    def test_short_text_is_returned_unchanged(self) -> None:
        assert trim_middle("short", 100) == "short"

    def test_a_long_text_keeps_its_start_and_its_end_around_one_marker(self) -> None:
        text = "START" + "x" * 3000 + "END"

        trimmed = trim_middle(text, 200)

        assert len(trimmed) <= 200
        assert trimmed.startswith("START")
        assert trimmed.endswith("END")
        assert trimmed.count(TRIM_MARKER) == 1

    def test_the_end_gets_the_larger_share(self) -> None:
        trimmed = trim_middle("a" * 1000 + "b" * 1000, 100)

        head, _, tail = trimmed.partition(TRIM_MARKER)
        assert len(tail) > len(head)

    def test_trimming_again_with_a_smaller_limit_keeps_a_single_marker(self) -> None:
        text = "START" + "x" * 3000 + "END"

        trimmed = trim_middle(trim_middle(text, 400), 120)

        assert len(trimmed) <= 120
        assert trimmed.count(TRIM_MARKER) == 1
        assert trimmed.startswith("START")
        assert trimmed.endswith("END")

    def test_trimming_twice_with_the_same_limit_changes_nothing(self) -> None:
        once = trim_middle("y" * 1000, 150)

        assert trim_middle(once, 150) == once

    def test_a_limit_too_small_for_the_marker_cuts_the_text_hard(self) -> None:
        assert trim_middle("abcdefghijklmnopqrstuvwxyz", 5) == "abcde"

    def test_a_negative_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            trim_middle("abc", -1)


# --- errors ---------------------------------------------------------------------------------------------------------


class TestErrors:
    def test_a_long_error_is_cut_in_place_and_keeps_its_type(self) -> None:
        error = execution_error("Code execution failed at line 'x' due to: KeyError: '" + "K" * 30_000 + "'")
        step = action_step(1, error=error)

        make_trim_callback()(step, agent=agent_with())

        assert step.error is error  # the same object, so the logger is not told about it a second time
        assert isinstance(step.error, AgentExecutionError)
        assert len(str(step.error)) <= 1500
        assert step.error.message == str(step.error)
        assert step.error.dict()["message"] == str(step.error)

    def test_both_ends_of_a_long_error_survive(self) -> None:
        step = action_step(1, error=execution_error("FAILED-LINE " + "m" * 20_000 + " WHY-IT-FAILED"))

        make_trim_callback()(step, agent=agent_with())

        assert str(step.error).startswith("FAILED-LINE")
        assert str(step.error).endswith("WHY-IT-FAILED")

    def test_the_next_prompt_shows_the_cut_error_only(self) -> None:
        step = action_step(1, error=execution_error("KeyError: '" + "K" * 30_000 + "'"))
        agent = agent_with(system="S")

        make_trim_callback()(step, agent=agent)

        assert estimate_tokens(next_prompt(agent, step)) < 1000

    def test_short_errors_are_left_alone(self) -> None:
        step = action_step(1, error=execution_error("NameError: name 'x' is not defined"))

        make_trim_callback()(step, agent=agent_with())

        assert str(step.error) == "NameError: name 'x' is not defined"

    def test_the_error_of_an_older_step_is_cut_too(self) -> None:
        old = action_step(1, error=execution_error("E" * 20_000))
        current = action_step(2, observations="fine")

        make_trim_callback()(current, agent=agent_with(old))

        assert len(str(old.error)) <= 1500

    def test_the_limit_is_an_option(self) -> None:
        step = action_step(1, error=execution_error("E" * 5000))

        make_trim_callback(max_error_chars=300)(step, agent=agent_with())

        assert len(str(step.error)) <= 300

    def test_the_generated_code_of_a_failed_step_is_kept(self) -> None:
        step = action_step(1, error=execution_error("E" * 20_000))
        before = (step.model_output, step.code_action, step.tool_calls)

        make_trim_callback()(step, agent=agent_with())

        assert (step.model_output, step.code_action, step.tool_calls) == before

    def test_a_step_without_an_error_is_not_touched(self) -> None:
        step = action_step(1, observations="fine")

        make_trim_callback()(step, agent=agent_with())

        assert step.error is None

    def test_the_error_limit_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_error_chars"):
            make_trim_callback(max_error_chars=0)


# --- parsing errors -------------------------------------------------------------------------------------------------


class TestParsingErrors:
    def test_the_reply_quoted_in_a_parsing_error_is_cut_down_with_the_error(self) -> None:
        reply = "I think the answer is probably " + "blah " * 1500
        step = action_step(1, error=parsing_error(reply), reply=reply)

        make_trim_callback()(step, agent=agent_with())

        assert len(str(step.error)) <= 1500
        assert len(step.model_output) <= 1500
        assert len(step.model_output_message.content) <= 1500
        assert step.model_output.startswith("I think the answer is probably")

    def test_the_instruction_at_the_end_of_a_parsing_error_survives(self) -> None:
        step = action_step(1, error=parsing_error("w " * 4000), reply="w " * 4000)

        make_trim_callback()(step, agent=agent_with())

        assert str(step.error).startswith("Error in code parsing")
        assert str(step.error).endswith("Make sure to provide correct code blobs.")

    def test_two_long_replies_without_code_do_not_blow_up_the_next_prompt(self) -> None:
        """Two replies of about 6,100 characters took the third request over the per-minute budget."""
        reply = ("A long thought without any code. " * 190)[:6100]
        first = action_step(1, error=parsing_error(reply), reply=reply)
        second = action_step(2, error=parsing_error(reply), reply=reply)
        agent = agent_with(system="s" * 4000, task="t" * 400)
        callback = make_trim_callback(max_prompt_tokens=5000)

        callback(first, agent=agent)
        agent.memory.steps.append(first)
        callback(second, agent=agent)

        assert estimate_tokens(next_prompt(agent, second)) <= 5000

    def test_an_execution_error_does_not_touch_the_reply(self) -> None:
        reply = "Thought: long\n<code>\n" + "x = 1\n" * 1000 + "</code>"
        step = action_step(1, error=execution_error("E" * 5000), reply=reply)

        make_trim_callback()(step, agent=agent_with())

        assert step.model_output == reply
        assert step.model_output_message.content == reply


# --- the size of the next prompt ------------------------------------------------------------------------------------


def long_history(count: int, observation_chars: int = 6000) -> list[ActionStep]:
    return [action_step(number, observations="o" * observation_chars) for number in range(1, count + 1)]


class TestPromptLimit:
    def test_without_a_limit_only_the_standard_trimming_applies(self) -> None:
        old = long_history(1)
        current = action_step(9, observations="c" * 9000)

        make_trim_callback()(current, agent=agent_with(*old, system="s" * 4000))

        assert len(old[0].observations) == 700
        assert len(current.observations) == 6000
        assert old[0].tool_calls is not None

    def test_a_prompt_within_the_limit_keeps_the_standard_trimming(self) -> None:
        old = long_history(2)
        current = action_step(3, observations="c" * 3000)

        make_trim_callback(max_prompt_tokens=50_000)(current, agent=agent_with(*old, system="s" * 1000))

        assert [len(step.observations) for step in old] == [700, 700]
        assert old[0].tool_calls is not None

    # For six old steps with 6,000-character observations, a 200-character system prompt and a 100-character task,
    # the next prompt is about 3,400 tokens with the standard limits, 2,300 after the first tighter level and 750
    # after the second.

    def test_a_prompt_over_the_limit_is_compressed_harder_but_the_newest_observation_is_spared(self) -> None:
        old = long_history(6)
        current = action_step(7, observations="c" * 6000)
        agent = agent_with(*old, system="s" * 200, task="t" * 100)

        make_trim_callback(max_prompt_tokens=3000)(current, agent=agent)

        assert estimate_tokens(next_prompt(agent, current)) <= 3000
        assert all(len(step.observations) == 200 for step in old)
        assert all(step.tool_calls is None for step in old)
        assert len(current.observations) == 6000

    def test_the_newest_observation_is_the_last_thing_to_be_cut(self) -> None:
        old = long_history(6)
        current = action_step(7, observations="c" * 6000)
        agent = agent_with(*old, system="s" * 200, task="t" * 100)

        make_trim_callback(max_prompt_tokens=1500)(current, agent=agent)

        assert estimate_tokens(next_prompt(agent, current)) <= 1500
        assert len(current.observations) == 1200
        assert all(len(step.observations) == 100 for step in old)

    def test_the_repeated_tool_calls_go_when_the_observations_are_not_enough(self) -> None:
        old = long_history(8, observation_chars=150)
        current = action_step(9, observations="c" * 100)
        agent = agent_with(*old, system="s" * 500)
        with_calls = estimate_tokens(next_prompt(agent, current))

        make_trim_callback(max_prompt_tokens=with_calls - 100)(current, agent=agent)

        assert all(step.tool_calls is None for step in old)
        assert current.tool_calls is not None
        assert estimate_tokens(next_prompt(agent, current)) <= with_calls - 100

    def test_a_prompt_that_cannot_fit_is_made_as_short_as_possible_without_failing(self) -> None:
        old = long_history(3)
        current = action_step(4, observations="c" * 6000)
        agent = agent_with(*old, system="s" * 40_000)

        make_trim_callback(max_prompt_tokens=100)(current, agent=agent)

        assert len(current.observations) == 1200
        assert all(len(step.observations) == 100 for step in old)

    def test_the_tighter_levels_never_exceed_the_configured_limits(self) -> None:
        old = long_history(3, observation_chars=1000)
        current = action_step(4, observations="c" * 1000)
        agent = agent_with(*old, system="s" * 40_000)

        make_trim_callback(max_last_obs_chars=500, max_old_obs_chars=80, max_prompt_tokens=100)(current, agent=agent)

        assert len(current.observations) == 500
        assert all(len(step.observations) == 80 for step in old)

    def test_an_agent_without_a_system_prompt_is_fine(self) -> None:
        current = action_step(1, observations="c" * 6000)

        make_trim_callback(max_prompt_tokens=100)(current, agent=SimpleNamespace(memory=SimpleNamespace(steps=[])))

        assert len(current.observations) == 1200

    @pytest.mark.parametrize("limit", [0, -5])
    def test_the_prompt_limit_must_be_positive(self, limit: int) -> None:
        with pytest.raises(ValueError, match="max_prompt_tokens"):
            make_trim_callback(max_prompt_tokens=limit)
