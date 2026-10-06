"""Keep the agent's context small by shortening what its memory holds about past steps.

smolagents rebuilds the whole prompt from ``agent.memory.steps`` before every model
call, so everything a step leaves in memory is paid for again at every later step. The step
callback made here caps that cost:

* the newest observation may stay fairly long (the model is about to read it), older ones are
  cut down to a short reminder;
* errors are cut too. smolagents sends ``step.error`` in full with every later prompt, and the
  text of a parsing error quotes the whole reply of the model, which is already in the prompt
  as the assistant turn: a 30,000-character ``KeyError`` or two long replies without code
  would otherwise bring a request over the per-minute token budget and end the question;
* when ``max_prompt_tokens`` is given and the next prompt would still be larger than that, the
  limits tighten step by step (shorter old observations and errors, then no repeated tool
  calls, then a shorter newest observation) instead of letting the request fail.

How the installed smolagents calls a step callback: after a step has run, and before
it is appended to ``agent.memory.steps``, it calls ``callback(step, agent=agent)`` for
callbacks that take more than one parameter. The step that just finished is therefore
not in ``agent.memory.steps`` yet; everything already in there is older.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from smolagents.memory import ActionStep
from smolagents.utils import AgentParsingError

from gaia_agent.tokens import estimate_tokens

TRIM_MARKER = "[...trimmed]"
HEAD_SHARE_OF_AN_ERROR = 0.4  # an error keeps this share from its start (what failed) and the rest from its end

# The tighter compression levels, tried in turn while the next prompt is over its limit, as the caps for
# (newest observation, old observation, error) and whether the repeated tool calls go. None leaves a cap alone:
# the newest observation, which the model is about to read, is the last thing to be cut.
_TIGHT_LEVELS = ((None, 200, 400, True), (1200, 100, 200, True))


def trim_text(text: str, limit: int) -> str:
    """Return ``text`` unchanged if it fits in ``limit`` characters, else its head ending in ``TRIM_MARKER``.

    The result is never longer than ``limit``. A limit too small to hold the marker
    cuts the text hard, without a marker. Trimming is idempotent.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")
    if len(text) <= limit:
        return text
    keep = limit - len(TRIM_MARKER)
    if keep < 0:
        return text[:limit]
    return text[:keep] + TRIM_MARKER


def trim_middle(text: str, limit: int) -> str:
    """Return ``text`` unchanged if it fits in ``limit`` characters, else its two ends around ``TRIM_MARKER``.

    For error messages, where the start says what failed and the end says why. The result is never longer than
    ``limit`` and holds a single marker, even for a text that was trimmed before.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")
    if len(text) <= limit:
        return text
    room = limit - len(TRIM_MARKER)
    if room < 0:
        return text[:limit]
    head_length = int(room * HEAD_SHARE_OF_AN_ERROR)
    tail_length = room - head_length
    head, _, tail = text.partition(TRIM_MARKER) if TRIM_MARKER in text else (text, "", text)
    return head[:head_length] + TRIM_MARKER + (tail[len(tail) - tail_length :] if tail_length else "")


@dataclass(frozen=True)
class _Limits:
    """How far one compression level cuts the memory."""

    last_obs: int
    old_obs: int
    error: int
    drop_old_tool_calls: bool


def _shorten_observations(step: ActionStep, limit: int) -> None:
    """Cut the observations of ``step`` in place; the code and tool calls are left alone."""
    observations = step.observations
    if not isinstance(observations, str):
        return
    trimmed = trim_text(observations, limit)
    if trimmed is not observations:
        step.observations = trimmed


def _shorten_error(step: ActionStep, limit: int) -> None:
    """Cut the error of ``step`` in place, keeping the exception object (rebuilding it would log it again).

    smolagents sends ``str(error)`` with every later prompt. A parsing error quotes the whole reply of the
    model, which the prompt already holds as the assistant turn, so that reply is cut down as well.
    """
    error = step.error
    if error is None:
        return
    shortened = trim_middle(str(error), limit)
    if shortened != str(error):
        error.args = (shortened,)
        if hasattr(error, "message"):
            error.message = shortened
    if isinstance(error, AgentParsingError):
        _shorten_reply(step, limit)


def _shorten_reply(step: ActionStep, limit: int) -> None:
    """Cut the reply of the model that failed to parse (its text is quoted in the error)."""
    if isinstance(step.model_output, str):
        step.model_output = trim_text(step.model_output, limit)
    message = step.model_output_message
    if message is not None and isinstance(message.content, str):
        message.content = trim_text(message.content, limit)


def _remembered_steps(agent: Any) -> list[Any]:
    """The steps in the agent's memory, or none when ``agent`` is missing or has no memory."""
    memory = getattr(agent, "memory", None)
    return list(getattr(memory, "steps", None) or [])


def _next_prompt_tokens(agent: Any, current: ActionStep) -> int:
    """The estimated size of the prompt of the next model call: the system prompt, the memory and ``current``."""
    system = getattr(getattr(agent, "memory", None), "system_prompt", None)
    messages = list(system.to_messages()) if system is not None else []
    for step in (*_remembered_steps(agent), current):
        messages.extend(step.to_messages())
    return estimate_tokens(messages)


def _apply(limits: _Limits, current: ActionStep, past: list[ActionStep]) -> None:
    """Shorten the newest step and the older ones to ``limits``."""
    _shorten_observations(current, limits.last_obs)
    _shorten_error(current, limits.error)
    for step in past:
        _shorten_observations(step, limits.old_obs)
        _shorten_error(step, limits.error)
        if limits.drop_old_tool_calls:
            step.tool_calls = None


def _levels(
    last_obs: int, old_obs: int, error: int, drop_old_tool_calls: bool, max_prompt_tokens: int | None
) -> list[_Limits]:
    """The compression levels for the configured limits, gentlest first; raises ValueError for a limit < 1."""
    for name, value in (("max_last_obs_chars", last_obs), ("max_old_obs_chars", old_obs), ("max_error_chars", error)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if max_prompt_tokens is not None and max_prompt_tokens <= 0:
        raise ValueError(f"max_prompt_tokens must be positive, got {max_prompt_tokens}")
    levels = [_Limits(last_obs, old_obs, error, drop_old_tool_calls)]
    for tight_last, tight_old, tight_error, drop in _TIGHT_LEVELS:
        levels.append(
            _Limits(
                last_obs=last_obs if tight_last is None else min(tight_last, last_obs),
                old_obs=min(tight_old, old_obs),
                error=min(tight_error, error),
                drop_old_tool_calls=drop or drop_old_tool_calls,
            )
        )
    return levels


def make_trim_callback(
    max_last_obs_chars: int = 6000,
    max_old_obs_chars: int = 700,
    *,
    drop_old_tool_calls: bool = False,
    max_error_chars: int = 1500,
    max_prompt_tokens: int | None = None,
) -> Callable[..., None]:
    """Return a smolagents step callback that shortens the memory of past steps.

    The latest observation may keep up to ``max_last_obs_chars`` characters; the observations of older
    steps are cut to ``max_old_obs_chars``, and every error to ``max_error_chars`` (both ends are kept, and
    a parsing error also loses the copy of the reply it quotes). Only observations, errors and, for parsing
    errors, the failed reply are touched, never the generated code of a step, the task or the planning
    steps. The smolagents memory is mutable by design and the prompt is rebuilt from it at every step, so
    the callback edits the step objects in place.

    ``max_prompt_tokens`` (off by default) is the size the next prompt may have, estimated as
    ``gaia_agent.tokens`` does. While the prompt is larger, the callback cuts harder: first old
    observations to 200 characters, errors to 400 and the repeated tool calls dropped, then the newest
    observation to 1200, the old ones to 100 and errors to 200 (a cap is never raised above the one
    configured). A prompt that cannot be made to fit this way is left as short as it gets.

    ``drop_old_tool_calls`` (off by default) removes ``tool_calls`` from every step older than the newest
    one. The CodeAgent prompt shows a step's code twice: in the model output and again in a ``Calling
    tools:`` block built from ``tool_calls``. The code stays, once, in the model output. The price is that
    saved traces of old steps no longer list their tool calls. Not validated on a real benchmark run:
    check the answers do not get worse before relying on it.

    Pass it as ``CodeAgent(..., step_callbacks=[callback])``. All limits must be positive.
    """
    levels = _levels(max_last_obs_chars, max_old_obs_chars, max_error_chars, drop_old_tool_calls, max_prompt_tokens)

    def trim_memory(memory_step: Any, agent: Any = None) -> None:
        if not isinstance(memory_step, ActionStep):
            return
        past = [step for step in _remembered_steps(agent) if step is not memory_step and isinstance(step, ActionStep)]
        for level in levels:
            _apply(level, memory_step, past)
            if max_prompt_tokens is None or _next_prompt_tokens(agent, memory_step) <= max_prompt_tokens:
                return

    return trim_memory
