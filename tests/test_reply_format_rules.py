"""Tests for the format guard's handling of invalid escapes, ``<think>`` reasoning and a final answer
given before its evidence: ``strip_reasoning``, ``defer_final_answer`` and the guard that uses them.
"""

from __future__ import annotations

import warnings

import pytest
from agent_fakes import ScriptedModel
from smolagents.models import ChatMessage, MessageRole

from gaia_agent.reply_format import (
    DEFAULT_TAGS,
    STEP_STOP_SEQUENCE,
    FormatGuardModel,
    defer_final_answer,
    final_answer_literal,
    repair_code_reply,
    strip_reasoning,
)

STEP_STOPS = [STEP_STOP_SEQUENCE, "Calling tools:", "</code>"]


def ask(model: FormatGuardModel, stop_sequences: list[str] | None = STEP_STOPS) -> ChatMessage:
    return model.generate([ChatMessage(role=MessageRole.USER, content="task")], stop_sequences=stop_sequences)


# --- an invalid escape sequence in the model's code ----------------------------------------------------------------

BACKSLASH = chr(92)
# A backslash-d inside a normal string literal is an invalid escape sequence: ast.parse warns about it.
BAD_ESCAPE_CODE = f"re.findall('{BACKSLASH}d+', text)"


def test_code_with_an_invalid_escape_sequence_is_still_repaired_when_warnings_are_errors() -> None:
    """Under ``-W error`` the warning of ``ast.parse`` used to turn into a failed repair."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        repaired = repair_code_reply(f"Thought: find the digits.\n{BAD_ESCAPE_CODE}")

    assert repaired == f"Thought: find the digits.\n<code>\n{BAD_ESCAPE_CODE}\n</code>"


def test_the_invalid_escape_warning_does_not_reach_the_console() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repair_code_reply(BAD_ESCAPE_CODE)
        final_answer_literal(f"final_answer('{BACKSLASH}d')")

    assert [str(item.message) for item in caught if issubclass(item.category, SyntaxWarning)] == []


# --- a final answer that is not a literal ------------------------------------------------------------------------

NOT_A_LITERAL = object()


def test_a_final_answer_of_none_is_told_apart_from_a_text_that_is_no_literal() -> None:
    assert final_answer_literal("final_answer(None)", NOT_A_LITERAL) is None
    assert final_answer_literal("final_answer(x)", NOT_A_LITERAL) is NOT_A_LITERAL
    assert final_answer_literal("Lisbon", NOT_A_LITERAL) is NOT_A_LITERAL


def test_the_default_for_a_text_that_is_no_literal_is_none() -> None:
    assert final_answer_literal("final_answer(x)") is None


# --- the reasoning that some models write before their reply ------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("<think>draft</think>Thought: real", "Thought: real"),
        ("<think>draft\nover two lines</think>\n\nThought: real", "Thought: real"),
        ("<THINK>draft</THINK>Thought: real", "Thought: real"),
        ("<think>first</think>A<think>second</think>B", "AB"),
        ("<think>reasoning that never ends", ""),
        ("reasoning with its opening tag stripped by the server</think>Thought: real", "Thought: real"),
        ("  Thought: no reasoning, spaces kept  ", "  Thought: no reasoning, spaces kept  "),
        ("", ""),
    ],
)
def test_reasoning_blocks_are_removed_from_a_reply(reply: str, expected: str) -> None:
    assert strip_reasoning(reply) == expected


def test_the_drafts_inside_the_reasoning_are_not_taken_for_the_reply() -> None:
    draft = "<think>I will print. Observation: nothing yet. <code>print(0)</code></think>"
    reply = draft + "\nThought: go\n<code>\nprint(1)\n</code>"

    content = ask(FormatGuardModel(ScriptedModel([reply]))).content

    assert content == "Thought: go\n<code>\nprint(1)\n"  # the stop sequence cut the closing tag, as usual


def test_a_reply_that_is_only_reasoning_counts_as_empty_and_is_asked_again() -> None:
    inner = ScriptedModel(["<think>endless", "Thought: now\n<code>\nprint(1)\n</code>"])

    content = ask(FormatGuardModel(inner)).content

    assert content == "Thought: now\n<code>\nprint(1)\n"
    assert len(inner.requests) == 2


def test_reasoning_is_removed_from_a_forced_answer_too() -> None:
    inner = ScriptedModel(["<think>maybe Paris</think>Lisbon"])

    assert ask(FormatGuardModel(inner), stop_sequences=None).content == "Lisbon"


# --- a final answer given before the evidence has been seen -------------------------------------------------------

TOOLS = ("web_search", "read_webpage", "read_file")


def deferring(code: str) -> str:
    """What the guard makes of the code of a reply, for a model that has the tools above."""
    return defer_final_answer(f"Thought: go\n<code>\n{code}\n</code>", DEFAULT_TAGS, TOOLS)


def test_a_final_answer_after_a_tool_call_in_the_same_block_is_turned_into_a_print() -> None:
    reply = deferring("results = web_search('x')\nfinal_answer(len(results))")

    assert reply == "Thought: go\n<code>\nresults = web_search('x')\nprint(len(results))\n</code>"


def test_a_final_answer_that_is_alone_in_its_block_is_left_alone() -> None:
    assert deferring("final_answer(42)") == "Thought: go\n<code>\nfinal_answer(42)\n</code>"


def test_a_final_answer_after_calculations_that_use_no_tool_is_left_alone() -> None:
    code = "total = sum([1, 2, 3])\nprint(total)\nfinal_answer(total)"

    assert deferring(code) == f"Thought: go\n<code>\n{code}\n</code>"


def test_a_final_answer_before_a_tool_call_is_left_alone() -> None:
    code = "final_answer(42)\nweb_search('never reached')"

    assert deferring(code) == f"Thought: go\n<code>\n{code}\n</code>"


def test_a_tool_called_inside_the_answer_counts_as_unseen_evidence() -> None:
    reply = deferring("final_answer(read_webpage('https://example.org', 'price'))")

    assert reply == "Thought: go\n<code>\nprint(read_webpage('https://example.org', 'price'))\n</code>"


def test_the_keyword_form_of_the_call_is_handled() -> None:
    reply = deferring("n = len(read_file('a.txt'))\nfinal_answer(answer=n)")

    assert reply == "Thought: go\n<code>\nn = len(read_file('a.txt'))\nprint(n)\n</code>"


def test_the_rest_of_the_code_is_kept_exactly_as_written() -> None:
    lines = ["# find it", "text = read_file('a.txt')   # keep this comment", "answer = text.split()[0]"]
    code = "\n".join([*lines, "final_answer(  answer  )", ""])

    reply = deferring(code)

    assert reply == "Thought: go\n<code>\n" + "\n".join([*lines, "print(answer)", "", "</code>"])


def test_a_final_answer_in_a_branch_after_a_tool_call_is_deferred_too() -> None:
    reply = deferring("hits = web_search('x')\nif hits:\n    final_answer('found')\nelse:\n    print('none')")

    assert "print('found')" in reply
    assert "final_answer" not in reply


def test_every_final_answer_call_after_a_tool_call_is_deferred() -> None:
    reply = deferring("hits = web_search('x')\nif hits:\n    final_answer('a')\nelse:\n    final_answer('b')")

    assert "final_answer" not in reply
    assert "print('a')" in reply
    assert "print('b')" in reply


def test_the_rewrite_survives_non_ascii_text_in_the_code() -> None:
    reply = deferring("x = read_file('café.txt')  # éè 日本\nfinal_answer('été')")

    assert reply.endswith("print('été')\n</code>")
    assert "café.txt" in reply


def test_only_the_blocks_that_break_the_rule_are_changed() -> None:
    reply = (
        "Thought: two blocks\n<code>\nfinal_answer(1)\n</code>\n"
        "More thinking.\n<code>\nread_file('a')\nfinal_answer(2)\n</code>"
    )

    assert defer_final_answer(reply, DEFAULT_TAGS, TOOLS) == (
        "Thought: two blocks\n<code>\nfinal_answer(1)\n</code>\n"
        "More thinking.\n<code>\nread_file('a')\nprint(2)\n</code>"
    )


def test_a_block_whose_closing_tag_was_cut_by_the_stop_sequence_is_handled() -> None:
    reply = "Thought: go\n<code>\nread_file('a')\nfinal_answer(2)\n"

    assert defer_final_answer(reply, DEFAULT_TAGS, TOOLS) == "Thought: go\n<code>\nread_file('a')\nprint(2)\n"


def test_markdown_python_blocks_are_handled_too() -> None:
    reply = "Thought: go\n```python\nread_file('a')\nfinal_answer(2)\n```"

    assert defer_final_answer(reply, DEFAULT_TAGS, TOOLS) == "Thought: go\n```python\nread_file('a')\nprint(2)\n```"


@pytest.mark.parametrize(
    "code",
    [
        "read_file('a'\nfinal_answer(2)",  # broken code stays broken
        "x = read_file\nfinal_answer(2)",  # a tool named but not called
        "read_file('a')\nfinal_answer()",  # nothing to print
        "read_file('a')\nfinal_answer(1, 2)",
        "read_file('a')\nlater = final_answer",  # not a call
    ],
)
def test_code_that_is_not_what_the_rule_is_about_is_left_alone(code: str) -> None:
    assert deferring(code) == f"Thought: go\n<code>\n{code}\n</code>"


def test_code_with_windows_line_ends_is_left_alone_rather_than_risk_a_wrong_offset() -> None:
    reply = "Thought: go\n<code>\nread_file('a')\r\nfinal_answer(2)\r\n</code>"

    assert defer_final_answer(reply, DEFAULT_TAGS, TOOLS) == reply


def test_characters_that_only_splitlines_treats_as_line_ends_do_not_shift_the_rewrite() -> None:
    form_feed, separator = chr(0x0C), chr(0x2028)
    code = f"note = 'a{form_feed}b{separator}c'\nread_file('a')\nfinal_answer(2)"

    reply = deferring(code)

    assert reply == f"Thought: go\n<code>\nnote = 'a{form_feed}b{separator}c'\nread_file('a')\nprint(2)\n</code>"


def test_without_tool_names_nothing_is_deferred() -> None:
    reply = "Thought: go\n<code>\nread_file('a')\nfinal_answer(2)\n</code>"

    assert defer_final_answer(reply, DEFAULT_TAGS, ()) == reply


def test_the_guard_defers_a_final_answer_that_comes_before_its_evidence() -> None:
    inner = ScriptedModel(["Thought: go\n<code>\nrows = web_search('x')\nfinal_answer(len(rows))\n</code>"])
    guard = FormatGuardModel(inner, tool_names=TOOLS)

    content = ask(guard).content

    assert content == "Thought: go\n<code>\nrows = web_search('x')\nprint(len(rows))\n"


def test_the_guard_leaves_a_forced_answer_alone() -> None:
    inner = ScriptedModel(["rows = web_search('x')\nfinal_answer(len(rows))"])
    guard = FormatGuardModel(inner, tool_names=TOOLS)

    assert ask(guard, stop_sequences=None).content == "rows = web_search('x')\nfinal_answer(len(rows))"


def test_the_guard_does_nothing_about_final_answers_unless_it_is_given_tool_names() -> None:
    inner = ScriptedModel(["Thought: go\n<code>\nrows = web_search('x')\nfinal_answer(len(rows))\n</code>"])

    content = ask(FormatGuardModel(inner)).content

    assert content == "Thought: go\n<code>\nrows = web_search('x')\nfinal_answer(len(rows))\n"
