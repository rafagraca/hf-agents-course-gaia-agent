"""Tests for the check the agent runs on every ``final_answer`` before accepting it.

smolagents 1.26 calls ``check_function(final_answer, self.memory, agent=self)`` inside an ``assert``: a falsy result
(or any exception) makes it report an error to the model, which then tries again.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from agent_fakes import PLAIN, ScriptedModel, code_reply, final_reply
from smolagents import CodeAgent
from smolagents.memory import AgentMemory

from gaia_agent import agent as agent_module
from gaia_agent.agent import answer_question, build_agent, check, check_final_answer
from gaia_agent.config import Settings


def call(final_answer: object) -> bool:
    """Call the check the way smolagents does."""
    return check(final_answer, AgentMemory(system_prompt=""), agent=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "answer",
    [
        "42",
        "Lisbon",
        "3.14",
        "right, left, up",
        "St. Petersburg",
        "a b c d e f g h i j k l",  # exactly twelve words
        "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen",
        "ünïcode",
        42,
        3.5,
        ["a", "b"],
        "  padded  ",
        "Final",
        "The answer is long enough to need words but it has a comma, so it is a list",
    ],
)
def test_a_short_bare_answer_is_accepted(answer: object) -> None:
    assert call(answer) is True


@pytest.mark.parametrize("answer", [None, "", "   ", "\n", "\t \n"])
def test_an_empty_answer_is_refused(answer: object) -> None:
    assert call(answer) is False


@pytest.mark.parametrize(
    "answer", ["FINAL ANSWER: 42", "final answer 42", "42 (FINAL ANSWER)", "Final   Answer: x", "The Final\nAnswer"]
)
def test_the_final_answer_marker_is_refused(answer: str) -> None:
    assert call(answer) is False


@pytest.mark.parametrize("answer", ["42\n43", "a\r\nb", "line one\nline two", "x\ry"])
def test_an_answer_with_line_breaks_is_refused(answer: str) -> None:
    assert call(answer) is False


def test_a_trailing_line_break_is_only_whitespace() -> None:
    assert call("42\n") is True


@pytest.mark.parametrize(
    "answer",
    [
        "I cannot determine this",
        "I can't find it",
        "Unable to answer",
        "unable to find the file",
        "I don't know",
        "I don’t know",
        "i do not know",
        "It is not possible to determine",
        "NOT POSSIBLE TO DETERMINE",
        "cannot be determined",
    ],
)
def test_a_refusal_is_refused(answer: str) -> None:
    assert call(answer) is False


@pytest.mark.parametrize("answer", ["Cannon", "Incannot", "Unabletoread"])
def test_words_that_only_look_like_a_refusal_are_fine(answer: str) -> None:
    assert call(answer) is True


def test_a_long_sentence_without_commas_is_refused() -> None:
    assert call("The answer to the question is that the place was founded in the nineteenth century") is False


def test_thirteen_words_without_commas_are_refused() -> None:
    assert call(" ".join("w" * n for n in range(1, 14))) is False


def test_the_check_has_the_signature_smolagents_uses() -> None:
    parameters = list(inspect.signature(check_final_answer).parameters.values())

    assert [p.name for p in parameters] == ["final_answer", "memory", "agent"]
    assert parameters[2].default is None
    assert check is check_final_answer


def test_the_name_of_the_check_says_what_it_wants_because_the_model_reads_it() -> None:
    """smolagents reports "Check <name> failed with error: ..." to the model, with no more detail than that."""
    assert "short" in check_final_answer.__name__ and "bare" in check_final_answer.__name__


def test_smolagents_still_calls_final_answer_checks_the_way_the_check_expects() -> None:
    """Pins the shape of ``CodeAgent._validate_final_answer`` that the check depends on."""
    source = inspect.getsource(CodeAgent._validate_final_answer)

    assert "check_function(final_answer, self.memory, agent=self)" in source
    assert "assert" in source


# --- inside a real agent -------------------------------------------------------------------------------------------


def run_agent(settings: Settings, replies: list[object]) -> Any:
    model = ScriptedModel(replies)
    agent = build_agent(settings, tools=[], model=model)
    return answer_question(agent, PLAIN, None), model


def test_the_agent_is_built_with_the_check(settings: Settings) -> None:
    agent = build_agent(settings, tools=[], model=ScriptedModel())

    assert agent.final_answer_checks == [check_final_answer]


def test_a_refused_answer_is_tried_again(settings: Settings) -> None:
    result, model = run_agent(settings, [final_reply("I cannot determine this"), final_reply("42")])

    assert (result.answer, result.steps, result.error) == ("42", 2, None)
    assert len(model.requests) == 2


def test_the_model_is_told_which_check_refused_its_answer(settings: Settings) -> None:
    result, model = run_agent(settings, [final_reply("FINAL ANSWER: 42"), final_reply("42")])

    second_request = "\n".join(str(message.content) for message in model.requests[1])
    assert check_final_answer.__name__ in second_request
    assert result.answer == "42"


def test_a_good_answer_passes_at_once(settings: Settings) -> None:
    result, model = run_agent(settings, [final_reply("Lisbon")])

    assert (result.answer, result.steps) == ("Lisbon", 1)
    assert len(model.requests) == 1


def test_an_agent_that_never_gives_a_bare_answer_ends_with_the_forced_answer(settings: Settings) -> None:
    bad = final_reply("I don't know")
    forced = code_reply("final_answer('best guess')")
    replies = [bad] * settings.max_steps + [forced]

    result, _ = run_agent(settings, replies)

    assert (result.answer, result.error) == ("best guess", None)


def test_the_module_exports_the_check() -> None:
    assert "check_final_answer" in agent_module.__all__
