"""Tests for ``gaia_agent.reply_format``: repairing model replies that lost their code tags."""

from __future__ import annotations

from typing import Any

import pytest
from agent_fakes import ScriptedModel
from smolagents.models import ChatMessage, MessageRole

from gaia_agent.reply_format import (
    STEP_STOP_SEQUENCE,
    FormatGuardModel,
    final_answer_literal,
    repair_code_reply,
)

STEP_STOPS = [STEP_STOP_SEQUENCE, "Calling tools:", "</code>"]


# --- repair_code_reply -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("final_answer(3)", "<code>\nfinal_answer(3)\n</code>"),
        ("  final_answer('a, b')\n", "<code>\nfinal_answer('a, b')\n</code>"),
        (
            "Thought: count them.\nresult = web_search('x')\nprint(result)",
            "Thought: count them.\n<code>\nresult = web_search('x')\nprint(result)\n</code>",
        ),
        (
            "Thought: almost there\nthe page lists them all\nfinal_answer(len(rows))",
            "Thought: almost there\nthe page lists them all\n<code>\nfinal_answer(len(rows))\n</code>",
        ),
        ("Thought: search\nweb_search('x')", "Thought: search\n<code>\nweb_search('x')\n</code>"),
    ],
)
def test_code_that_lost_its_tags_gets_them_back(reply: str, expected: str) -> None:
    assert repair_code_reply(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "3",  # a bare answer: no call to run, so the parser's feedback is the better outcome
        "Paris",
        "Thought: I will search the web.",
        "x = 1",
        "Thought: x\n<code>\nprint(1)\n</code>",
        "Thought: x\n<code>\nprint(1)\n",  # smolagents closes an open block itself
        "```python\nprint(1)\n```",  # smolagents reads markdown fences too
        "Thought: search\nweb_search('x'",  # broken code stays broken
    ],
)
def test_other_replies_are_left_alone(reply: str) -> None:
    assert repair_code_reply(reply) == reply


def test_a_thought_line_is_never_run_as_code() -> None:
    # "Thought: search" is valid Python (an annotation), so it must not be swallowed into the code block.
    assert repair_code_reply("Thought: search\nprint(1)").startswith("Thought: search\n<code>\n")


def test_custom_tags_are_used() -> None:
    assert repair_code_reply("print(1)", ("```python", "```")) == "```python\nprint(1)\n```"


# --- final_answer_literal ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("final_answer(3)", 3),
        ("final_answer('Paris')", "Paris"),
        ("final_answer(answer='Paris')", "Paris"),
        ("<code>\nfinal_answer([1, 2])\n</code>", [1, 2]),
        ("Thought: done.\n<code>\nfinal_answer('x')\n</code>", "x"),
        ("```python\nfinal_answer(2.5)\n```", 2.5),
    ],
)
def test_a_literal_final_answer_call_is_read(text: str, expected: object) -> None:
    assert final_answer_literal(text) == expected


@pytest.mark.parametrize(
    "text",
    ["3", "Paris", "final_answer(x)", "print(1)", "final_answer(1, 2)", "final_answer(", "final_answer()", "f(3)"],
)
def test_anything_else_is_not_a_literal_final_answer(text: str) -> None:
    assert final_answer_literal(text) is None


# --- FormatGuardModel --------------------------------------------------------------------------------------------


class ChangingIdModel(ScriptedModel):
    """A scripted model that reports a different id after each call, like a model chain moving on."""

    def generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> ChatMessage:
        message = super().generate(messages, *args, **kwargs)
        self.model_id = f"fake/model-{len(self.requests)}"
        return message


class StopRecordingModel(ScriptedModel):
    """A scripted model that records the stop sequences it is given."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)
        self.stops: list[list[str] | None] = []

    def generate(self, messages: list[Any], stop_sequences: list[str] | None = None, **kwargs: Any) -> ChatMessage:
        self.stops.append(stop_sequences)
        return super().generate(messages, stop_sequences=stop_sequences, **kwargs)


def ask(model: FormatGuardModel, stop_sequences: list[str] | None = STEP_STOPS) -> ChatMessage:
    return model.generate([ChatMessage(role=MessageRole.USER, content="task")], stop_sequences=stop_sequences)


def test_action_step_replies_are_repaired() -> None:
    guard = FormatGuardModel(ScriptedModel(["final_answer(3)"]))

    reply = ask(guard)

    assert reply.content == "<code>\nfinal_answer(3)\n</code>"
    assert reply.token_usage is not None


def test_replies_outside_action_steps_are_untouched() -> None:
    guard = FormatGuardModel(ScriptedModel(["final_answer(3)"]))

    assert ask(guard, stop_sequences=None).content == "final_answer(3)"


def test_action_step_stop_sequences_are_applied_to_the_reply_instead_of_sent_to_the_provider() -> None:
    # gpt-oss drafts its code, closing tag included, in its hidden reasoning: a provider-side stop
    # sequence fires there and leaves an empty or cut reply.
    inner = StopRecordingModel(["\n".join(["Thought: x", "<code>", "print(1)", "</code>", "Observation: made up"])])

    reply = ask(FormatGuardModel(inner))

    assert inner.stops == [None]
    assert reply.content == "\n".join(["Thought: x", "<code>", "print(1)", ""])


@pytest.mark.parametrize("first", ["right", "Thought: I will search the web.", "30"])
def test_a_step_reply_without_code_is_asked_again_once(first: str) -> None:
    inner = ScriptedModel([first, "final_answer('right')"])

    assert ask(FormatGuardModel(inner)).content == "<code>\nfinal_answer('right')\n</code>"
    assert len(inner.requests) == 2


def test_a_second_reply_without_code_is_left_to_the_parser() -> None:
    inner = ScriptedModel(["right", "Thought: hmm.", "never asked"])

    assert ask(FormatGuardModel(inner)).content == "Thought: hmm."
    assert len(inner.requests) == 2


def test_a_markdown_python_block_counts_as_code() -> None:
    inner = ScriptedModel(["Thought: x\n```python\nprint(1)\n```", "never asked"])

    assert ask(FormatGuardModel(inner)).content == "Thought: x\n```python\nprint(1)\n```"
    assert len(inner.requests) == 1


def test_an_empty_forced_answer_is_asked_again_once() -> None:
    inner = ScriptedModel(["", "Lisbon"])

    assert ask(FormatGuardModel(inner), stop_sequences=None).content == "Lisbon"
    assert len(inner.requests) == 2


class RawContentModel(ScriptedModel):
    """Replies with the given contents as they are: message parts, None..."""

    def __init__(self, contents: list[object]) -> None:
        super().__init__([])
        self.contents = list(contents)

    def generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> ChatMessage:
        self.requests.append(list(messages))
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.contents.pop(0))


def test_replies_given_as_message_parts_are_read_as_text() -> None:
    inner = RawContentModel([None, [{"type": "text", "text": "print("}, {"type": "text", "text": "1)"}]])

    reply = ask(FormatGuardModel(inner))

    assert reply.content == "<code>\nprint(1)\n</code>"
    assert len(inner.requests) == 2


def test_other_stop_sequences_are_passed_on() -> None:
    inner = StopRecordingModel(["a plan"])

    reply = ask(FormatGuardModel(inner), stop_sequences=["<end_plan>"])

    assert inner.stops == [["<end_plan>"]]
    assert reply.content == "a plan"


def test_an_empty_reply_is_asked_again_once() -> None:
    inner = ScriptedModel(["", "print(1)"])
    guard = FormatGuardModel(inner)

    assert ask(guard).content == "<code>\nprint(1)\n</code>"
    assert len(inner.requests) == 2


def test_a_second_empty_reply_is_returned_as_it_is() -> None:
    inner = ScriptedModel(["", "  ", "never asked"])
    guard = FormatGuardModel(inner)

    assert ask(guard).content == "  "
    assert len(inner.requests) == 2


def test_the_guard_reports_the_model_that_answered_last() -> None:
    inner = ChangingIdModel(["print(1)", "print(2)"])
    guard = FormatGuardModel(inner)
    assert guard.model_id == inner.model_id

    ask(guard)
    ask(guard)

    assert guard.model_id == "fake/model-2"
    assert guard.inner is inner
