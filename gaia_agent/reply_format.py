"""Repairs for the reply format of the code agent.

smolagents expects each step to be a Thought followed by one code block between the agent's tags
(``<code>`` ... ``</code>``). gpt-oss often breaks that format: it replies ``final_answer(3)`` or a
few lines of bare code, a Thought without code, a bare answer such as ``3``, or nothing at all.
smolagents then appends the closing tag to the reply, which also defeats its own fallback for bare
code, and the step is lost to a parsing error.

``FormatGuardModel`` wraps the agent's model. In action steps it removes the reasoning some models
write before their reply (``<think>...</think>``), applies the stop sequences to the reply itself rather
than sending them to the provider (on Groq they also fire inside the hidden reasoning of gpt-oss and
leave an empty reply), puts the tags back around a reply that ends in valid Python calling something,
and asks once more when the reply still has no code block: a fresh sample costs a call, a parsing error
costs a call and a step. The repaired reply is what the agent stores, so later prompts show the model the
expected format. A reply that still has no code goes to the parser, whose feedback asks the model for
code; the guard never guesses that text is an answer.

When it is given the names of the tools, the guard also holds a model to the rule of the prompt that
``final_answer`` is called alone, after the evidence has been seen: a block that calls a tool and then
``final_answer`` in the same breath answers blind, before the output of that tool (or its own prints) has
been read, so the call is turned into a ``print`` and the next step confirms the answer.

``final_answer_literal`` reads a ``final_answer(<literal>)`` call written as text, which a model may
reply when smolagents forces an answer at the step limit.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import re
import warnings
from collections.abc import Collection
from typing import Any

from smolagents import Model, Tool
from smolagents.models import ChatMessage

__all__ = [
    "DEFAULT_TAGS",
    "STEP_STOP_SEQUENCE",
    "FormatGuardModel",
    "defer_final_answer",
    "final_answer_literal",
    "repair_code_reply",
    "strip_reasoning",
]

logger = logging.getLogger(__name__)

DEFAULT_TAGS = ("<code>", "</code>")
STEP_STOP_SEQUENCE = "Observation:"  # smolagents passes this stop sequence only when it asks for an action step
REPLY_RETRIES = 1  # a fresh sample after a reply without code (action step) or an empty one
FINAL_ANSWER_TOOL = "final_answer"

_THOUGHT_LINE = re.compile(r"\s*thoughts?\s*:", re.IGNORECASE)
_CODE_BLOCK = re.compile(r"(?:<code>|```(?:python|py)?)\s*(.*?)\s*(?:</code>|```|$)", re.DOTALL)
_PARSE_FAILURES = (SyntaxError, ValueError, RecursionError)
_MARKDOWN_FENCE = re.compile(r"```(?:python|py)")
_MARKDOWN_BLOCK = r"(```(?:python|py)\n)(.*?)(\n```|\Z)"
_REASONING = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)
_REASONING_END = "</think>"


def _parse(code: str) -> ast.Module | None:
    """Parse ``code``, or None when it is not valid Python. Warnings (``'\\d'`` escapes) are no concern here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ast.parse(code)
        except _PARSE_FAILURES:
            return None


def _calls_something(code: str) -> bool:
    """True when ``code`` is valid Python that calls at least one function."""
    tree = _parse(code)
    return tree is not None and any(isinstance(node, ast.Call) for node in ast.walk(tree))


def _cut_at_stop_sequences(content: str, stop_sequences: list[str] | None) -> str:
    """``content`` up to the first occurrence of any stop sequence."""
    for stop in stop_sequences or []:
        content = content.split(stop)[0]
    return content


def strip_reasoning(text: str) -> str:
    """``text`` without the ``<think>...</think>`` reasoning some models write before the reply.

    A block that never closes (the reasoning ate the whole output) is removed to the end, and so is the
    text before a lone ``</think>`` (some servers drop the opening tag). A text without any reasoning is
    returned exactly as it is; what remains after removing some is stripped at the start.
    """
    if "<think>" not in text.lower() and _REASONING_END not in text.lower():
        return text
    cleaned = _REASONING.sub("", text)
    if _REASONING_END in cleaned.lower():
        cleaned = cleaned[cleaned.lower().rindex(_REASONING_END) + len(_REASONING_END) :]
    return cleaned.lstrip()


def repair_code_reply(text: str, tags: tuple[str, str] = DEFAULT_TAGS) -> str:
    """Put the code tags back into ``text`` when it ends in bare code; otherwise return ``text`` unchanged.

    The code is the longest tail of lines that is valid Python and calls something; the lines
    before it stay outside the block, as the Thought. A code block never starts with a
    ``Thought:`` line, even though such a line happens to be valid Python (an annotation).
    Replies holding a tag or a markdown fence are left to smolagents, which parses them.
    """
    opening, closing = tags
    if not text.strip() or opening in text or "```" in text:
        return text
    lines = text.strip().splitlines()
    for start, line in enumerate(lines):
        if _THOUGHT_LINE.match(line):
            continue
        code = "\n".join(lines[start:]).strip()
        if _calls_something(code):
            thought = "\n".join(lines[:start]).strip()
            block = f"{opening}\n{code}\n{closing}"
            return f"{thought}\n{block}" if thought else block
    return text


def final_answer_literal(text: str, default: Any = None) -> Any:
    """The value of a text reply that is just ``final_answer(<literal>)`` (tags allowed), else ``default``.

    Pass a ``default`` of your own to tell ``final_answer(None)`` (the value None) from a text that is
    no such call at all.
    """
    match = _CODE_BLOCK.search(text)
    tree = _parse((match.group(1) if match else text).strip())
    if tree is None or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return default
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != FINAL_ANSWER_TOOL:
        return default
    arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
    if len(arguments) != 1 or any(keyword.arg != "answer" for keyword in call.keywords):
        return default
    try:
        return ast.literal_eval(arguments[0])
    except (*_PARSE_FAILURES, TypeError):
        return default


# --- a final answer given before the evidence has been seen ---------------------------------------------------------


def _calls_of(tree: ast.AST, names: Collection[str]) -> list[ast.Call]:
    """The calls ``name(...)`` to one of ``names`` in ``tree``."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
    ]


def _blind_final_answers(tree: ast.AST, tool_names: Collection[str]) -> list[ast.Call]:
    """The ``final_answer(x)`` calls with exactly one argument that come after (or around) a call to a tool."""
    tool_starts = [(call.lineno, call.col_offset) for call in _calls_of(tree, tool_names)]
    blind = []
    for call in _calls_of(tree, {FINAL_ANSWER_TOOL}):
        argument_count = len(call.args) + len(call.keywords)
        ends = (call.end_lineno or call.lineno, call.end_col_offset or 0)
        if argument_count == 1 and any(start < ends for start in tool_starts):
            blind.append(call)
    return blind


def _char_offset(lines: list[str], line: int, byte_column: int) -> int:
    """The offset in the code of an ``ast`` position (a 1-based line and a column counted in UTF-8 bytes).

    ``lines`` are the lines of the code, each with its line end (see :func:`_defer_in_code`).
    """
    before = sum(len(text) for text in lines[: line - 1])
    return before + len(lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8"))


def _defer_in_code(code: str, tool_names: Collection[str]) -> str:
    """``code`` with every blind ``final_answer(x)`` turned into ``print(x)``; the rest of the text is untouched."""
    tree = _parse(code)
    if tree is None or "\r" in code:  # Windows line ends: not worth the risk of a wrong offset
        return code
    blind = _blind_final_answers(tree, tool_names)
    if not blind:
        return code
    # Only "\n" ends a line for ast; str.splitlines knows more kinds of line end and would shift every offset.
    lines = [f"{line}\n" for line in code.split("\n")]
    result = code
    for call in sorted(blind, key=lambda node: (node.lineno, node.col_offset), reverse=True):
        argument = call.args[0] if call.args else call.keywords[0].value
        start = _char_offset(lines, call.lineno, call.col_offset)
        end = _char_offset(lines, call.end_lineno or call.lineno, call.end_col_offset or 0)
        result = f"{result[:start]}print({ast.unparse(argument)}){result[end:]}"
    return result


def defer_final_answer(text: str, tags: tuple[str, str], tool_names: Collection[str]) -> str:
    """Turn a ``final_answer(x)`` that follows a tool call in the same code block into ``print(x)``.

    The prompt asks the model to call ``final_answer`` alone, after it has seen the evidence it relies
    on. A model that searches and answers in one block answers blind: with a regular expression over a
    page it never read, for instance, and ``final_answer(0)`` when the expression matched nothing. The
    printed value is what the next step sees, and one step later the model confirms it (or corrects it).
    Only blocks that break the rule change, and only the ``final_answer`` call: the code around it stays
    exactly as written. Blocks are the ones between ``tags`` (a closing tag cut by a stop sequence is fine)
    and markdown python fences. ``tool_names`` are the names of the agent's tools; with none, nothing changes.
    """
    if not tool_names:
        return text
    tagged = rf"({re.escape(tags[0])})(.*?)({re.escape(tags[1])}|\Z)"
    pattern = re.compile(f"{tagged}|{_MARKDOWN_BLOCK}", re.DOTALL)

    def rewrite(match: re.Match[str]) -> str:
        opening, code, closing = match.group(1, 2, 3) if match.group(1) is not None else match.group(4, 5, 6)
        deferred = _defer_in_code(code, tool_names)
        if deferred != code:
            logger.info("Held a final_answer call back until its evidence has been seen.")
        return f"{opening}{deferred}{closing}"

    return pattern.sub(rewrite, text)


# --- the guard ------------------------------------------------------------------------------------------------------


def _has_code_block(text: str, tags: tuple[str, str]) -> bool:
    """True when smolagents can find a code block in ``text``: the opening tag or a python markdown fence."""
    return tags[0] in text or _MARKDOWN_FENCE.search(text) is not None


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


class FormatGuardModel(Model):
    """Wraps the agent's model and repairs the format of its action-step replies (see the module docstring).

    ``model_id`` follows the wrapped model, so it names the model that answered last. ``tool_names`` are
    the names of the agent's tools: with them, a ``final_answer`` call that comes after a tool call in the
    same block is turned into a ``print`` (see :func:`defer_final_answer`).
    """

    def __init__(self, inner: Model, tags: tuple[str, str] = DEFAULT_TAGS, tool_names: Collection[str] = ()) -> None:
        super().__init__(model_id=getattr(inner, "model_id", None))
        self._inner = inner
        self._tags = tags
        self._tool_names = frozenset(name for name in tool_names if name != FINAL_ANSWER_TOOL)

    @property
    def inner(self) -> Model:
        """The wrapped model."""
        return self._inner

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Ask the wrapped model; for an action step, apply the stop sequences and repair the reply.

        An action-step reply that still has no code block, and any other reply that is empty, is
        asked for once more. The stop sequences of an action step are applied here because the
        provider would also apply them inside the hidden reasoning of gpt-oss, which drafts its code
        there, closing tag included, and the reply would come back empty or cut after the Thought.
        A forced answer comes back empty when the reasoning eats all the output tokens.
        """
        if stop_sequences is None or STEP_STOP_SEQUENCE not in stop_sequences:
            return self._other_reply(messages, stop_sequences, response_format, tools_to_call_from, kwargs)
        return self._step_reply(messages, stop_sequences, response_format, tools_to_call_from, kwargs)

    def _step_reply(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str],
        response_format: dict[str, str] | None,
        tools_to_call_from: list[Tool] | None,
        kwargs: dict[str, Any],
    ) -> ChatMessage:
        for attempt in range(1 + REPLY_RETRIES):
            message = self._ask(messages, None, response_format, tools_to_call_from, kwargs)
            text = _cut_at_stop_sequences(strip_reasoning(_text(message.content)), stop_sequences)
            repaired = repair_code_reply(text, self._tags)
            if _has_code_block(repaired, self._tags):
                break
            if attempt < REPLY_RETRIES:
                logger.info("The reply has no code block; asking again.")
        if repaired != text:
            logger.info("Put the missing code tags back into a reply.")
        repaired = defer_final_answer(repaired, self._tags, self._tool_names)
        return message if repaired == message.content else dataclasses.replace(message, content=repaired)

    def _other_reply(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None,
        response_format: dict[str, str] | None,
        tools_to_call_from: list[Tool] | None,
        kwargs: dict[str, Any],
    ) -> ChatMessage:
        for attempt in range(1 + REPLY_RETRIES):
            message = self._ask(messages, stop_sequences, response_format, tools_to_call_from, kwargs)
            text = strip_reasoning(_text(message.content))
            if text.strip():
                break
            if attempt < REPLY_RETRIES:
                logger.info("The model sent an empty reply; asking again.")
        return message if text == message.content else dataclasses.replace(message, content=text)

    def _ask(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None,
        response_format: dict[str, str] | None,
        tools_to_call_from: list[Tool] | None,
        kwargs: dict[str, Any],
    ) -> ChatMessage:
        message = self._inner.generate(
            messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        self.model_id = getattr(self._inner, "model_id", self.model_id)
        return message
