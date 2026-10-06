"""Build the smolagents CodeAgent and answer one question with it.

``build_agent`` wires the compact prompt, the project tools, the budgeted model and the
observation-trimming callback into a ``CodeAgent``. ``answer_question`` runs an agent on one
exam question from a clean memory, normalises the answer for the exact-match scorer and turns
every failure into an ``AnswerResult`` whose ``error`` says what went wrong, so that one bad
question never stops a whole run. Error texts are single-line, bounded and free of secrets.

Use a new agent for every question (``build_agent`` is cheap) but share one model across a run:
the code executor keeps the variables of earlier tasks, while the ``BudgetedModel`` must see
every call of the run to keep its per-minute budget right.
"""

from __future__ import annotations

import importlib.resources
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from smolagents import CodeAgent, Model, Tool
from smolagents.memory import ActionStep
from smolagents.monitoring import LogLevel
from smolagents.utils import AgentGenerationError

from gaia_agent._answer_check import check, check_final_answer
from gaia_agent._secrets import mask_secrets
from gaia_agent.answer import normalize_answer
from gaia_agent.api import Question
from gaia_agent.budget import BudgetedModel, NoModelAvailableError, PromptTooLargeError
from gaia_agent.config import Settings
from gaia_agent.memory import make_trim_callback
from gaia_agent.prompts import build_prompt_templates
from gaia_agent.reply_format import FormatGuardModel, final_answer_literal
from gaia_agent.tools import default_tools

__all__ = [
    "AUTHORIZED_IMPORTS",
    "EXECUTOR_TIMEOUT_SECONDS",
    "AnswerResult",
    "answer_question",
    "build_agent",
    "build_task",
    "check",
    "check_final_answer",
    "is_fatal",
]

# The code the model writes runs in smolagents' LocalPythonExecutor, which is NOT a sandbox: the safety of the agent is
# what this list lets through. Only modules that compute (no files, no network, no processes, no unpickling) are here.
# pandas and numpy were once on the list and are gone on purpose: ``DataFrame.to_csv`` and ``ndarray.tofile`` write
# arbitrary files, ``read_csv`` reads any file or URL and ``read_pickle`` runs code, so a prompt injected into a web
# page could have used them. Spreadsheets are read by the ``read_file`` tool, which renders them as CSV text.
AUTHORIZED_IMPORTS = (
    "math",
    "statistics",
    "re",
    "json",
    "datetime",
    "collections",
    "itertools",
    "fractions",
    "decimal",
    "unicodedata",
    "csv",
)
# Not preemptive: smolagents runs the code in a thread and only raises ExecutionTimeoutError once that thread returns,
# so a call that never returns (``time.sleep(10**9)``) still blocks the run. The value bounds well-behaved code only.
EXECUTOR_TIMEOUT_SECONDS = 120  # smolagents allows 30 s per code block; transcriptions and scripts need more
MAX_ERROR_CHARS = 500
FATAL_ERRORS: tuple[type[Exception], ...] = (NoModelAvailableError,)
FINAL_ANSWER_FAILURE = "FinalAnswerError"
_NOT_A_LITERAL = object()  # what final_answer_literal returns for a text that is no final_answer(<literal>) call
CODE_TAGS = ("<code>", "</code>")  # given to the CodeAgent and to the format guard, so that they always agree

# smolagents returns this text (inside a list of message parts) when the answer forced at the step limit fails.
_FORCED_ANSWER_ERROR_PREFIX = "Error in generating final LLM output:"
# Replaces the smolagents request made at the step limit, which invites a free-form explanation.
FORCED_ANSWER_PROMPT = (
    "Based on the above, give your best answer to the task below. Reply with ONLY the answer, as short as possible "
    "and in the exact format the task asks for: no explanation and no label.\nTask: {{task}}"
)


@dataclass(frozen=True)
class AnswerResult:
    """Outcome of answering one question; ``error`` is None on success."""

    task_id: str
    answer: str
    raw_answer: str
    steps: int
    seconds: float
    error: str | None = None
    model: str = ""


def build_agent(
    settings: Settings,
    tools: list[Tool] | None = None,
    model: Model | None = None,
    usage_log: Path | None = None,
    *,
    verbosity: LogLevel = LogLevel.OFF,
) -> CodeAgent:
    """Return a CodeAgent wired with the compact prompts, the trim callback and a budgeted model.

    ``tools`` defaults to ``default_tools(settings)``; ``model`` defaults to a ``BudgetedModel``
    over ``settings.model_id`` and the fallbacks, which appends its consumption to ``usage_log``
    when a path is given (a ``model`` that is passed in keeps its own log, so passing both is an
    error). Either way the model is wrapped in a ``FormatGuardModel``, which repairs replies that
    lost their code tags and holds the model to calling ``final_answer`` alone. The memory callback
    keeps the next prompt within ``settings.max_prompt_tokens`` by cutting old observations and
    errors. There is no planning step, the step limit is ``settings.max_steps`` and the console
    shows nothing (not even errors, whose text quotes the model's replies: the traces keep them)
    unless ``verbosity`` asks for more.
    """
    if model is not None and usage_log is not None:
        raise ValueError(
            "usage_log only goes with the default model: a model that is passed in keeps its own usage log"
        )
    tool_list = list(tools) if tools is not None else default_tools(settings)
    chain = (settings.model_id, *settings.fallback_model_ids)
    inner = model if model is not None else BudgetedModel(settings, usage_log, model_ids=chain)
    return CodeAgent(
        tools=tool_list,
        model=FormatGuardModel(inner, CODE_TAGS, tool_names=[tool.name for tool in tool_list]),
        prompt_templates=_prompt_templates(),
        instructions=_instructions(settings.max_steps),
        additional_authorized_imports=list(AUTHORIZED_IMPORTS),
        code_block_tags=CODE_TAGS,
        max_steps=settings.max_steps,
        planning_interval=None,
        final_answer_checks=[check_final_answer],
        step_callbacks=[make_trim_callback(max_prompt_tokens=settings.max_prompt_tokens)],
        verbosity_level=verbosity,
        executor_kwargs={"timeout_seconds": EXECUTOR_TIMEOUT_SECONDS},
    )


def build_task(q: Question, file_path: Path | None) -> str:
    """The task text given to the agent: the question, plus the name of its attachment (or that it is missing).

    Only the file name is given, never the path: the file tools look attachments up by name in the
    attachments folder, and a path would show the model (and the language model provider) where the
    benchmark data lives on this machine.
    """
    if file_path is not None:
        return f"{q.question}\n\nAttached file: {file_path.name}"
    if q.file_name:
        kind = Path(q.file_name).suffix or "an"
        return (
            f"{q.question}\n\nNote: this task comes with an attached {kind} file, but it could not be "
            "downloaded. Answer without it as best you can."
        )
    return q.question


def answer_question(agent: CodeAgent, q: Question, file_path: Path | None) -> AnswerResult:
    """Run ``agent`` on ``q`` (mentioning the attachment ``file_path`` if any) and normalise the answer.

    The agent starts from a clean memory. Any exception is caught and reported in ``error``
    (``"<ExceptionType>: <message>"``, the provider error rather than the smolagents wrapper);
    ``is_fatal`` tells whether it means that no model is left for the rest of the run. A failed
    answer at the step limit is reported as ``FinalAnswerError``; a forced answer written as a
    ``final_answer(<literal>)`` call is read as that literal. ``KeyboardInterrupt`` is not caught.

    The tools that have a ``set_attachment`` method (``run_python_file``) are told which file is the
    attachment of this question while it is answered, and are told there is none afterwards.
    """
    _announce_attachment(agent, file_path)
    try:
        return _answer(agent, q, file_path)
    finally:
        _announce_attachment(agent, None)


def _announce_attachment(agent: CodeAgent, file_path: Path | None) -> None:
    """Tell every tool that wants to know (it has ``set_attachment``) which attachment the question has."""
    for tool in agent.tools.values():
        announce = getattr(tool, "set_attachment", None)
        if callable(announce):
            announce(file_path)


def _answer(agent: CodeAgent, q: Question, file_path: Path | None) -> AnswerResult:
    started = time.monotonic()
    try:
        output = _unwrap_parts(agent.run(build_task(q, file_path), reset=True))
    except Exception as exc:  # noqa: BLE001 - one failed question must not stop the run; the error is recorded
        if _is_prompt_too_large(exc):
            return _best_answer_now(agent, q, started, exc)
        return _result(agent, q, started, error=_describe(exc))
    return _finish(agent, q, started, output)


def _finish(agent: CodeAgent, q: Question, started: float, output: object) -> AnswerResult:
    """The result for what the agent returned: a failed forced answer, or the normalised answer."""
    if isinstance(output, str) and output.startswith(_FORCED_ANSWER_ERROR_PREFIX):
        failure = output.removeprefix(_FORCED_ANSWER_ERROR_PREFIX).strip()
        return _result(agent, q, started, error=_clean_error(f"{FINAL_ANSWER_FAILURE}: {failure}"))
    raw = "" if output is None else str(output)
    # A forced answer may be code. ``final_answer(None)`` is a literal too, and means no answer.
    literal = final_answer_literal(output, _NOT_A_LITERAL) if isinstance(output, str) else _NOT_A_LITERAL
    value = output if literal is _NOT_A_LITERAL else literal
    return _result(agent, q, started, answer=normalize_answer(q.question, value), raw=raw)


def _is_prompt_too_large(exc: Exception) -> bool:
    """True when ``exc`` (or the provider error smolagents wrapped) says the prompt cannot fit the budget."""
    return isinstance(exc, PromptTooLargeError) or isinstance(exc.__cause__, PromptTooLargeError)


def _best_answer_now(agent: CodeAgent, q: Question, started: float, failure: Exception) -> AnswerResult:
    """The memory no longer fits a request: ask for the best answer to the task now, as at the step limit.

    The memory has been squeezed as far as it goes, so this seldom works; when it does not, the result
    is the original error.
    """
    reply = agent.provide_final_answer(getattr(agent, "task", None) or q.question)
    output = _unwrap_parts(reply.content)
    failed = isinstance(output, str) and output.startswith(_FORCED_ANSWER_ERROR_PREFIX)
    if failed or not isinstance(output, str) or not output.strip():
        return _result(agent, q, started, error=_describe(failure))
    return _finish(agent, q, started, output)


def is_fatal(result: AnswerResult) -> bool:
    """True when ``result`` failed because no model can answer anymore, so the run should stop."""
    if result.error is None:
        return False
    return result.error.startswith(tuple(f"{error.__name__}:" for error in FATAL_ERRORS))


def _prompt_templates() -> dict[str, Any]:
    """The CodeAgent templates of the installed smolagents, with the compact system prompt and forced answer."""
    text = importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text(encoding="utf-8")
    templates = build_prompt_templates(yaml.safe_load(text))
    return {**templates, "final_answer": {**templates["final_answer"], "post_messages": FORCED_ANSWER_PROMPT}}


def _instructions(max_steps: int) -> str:
    """Extra system prompt rules: gpt-oss tends to answer with a JSON tool call, which the provider refuses."""
    opening, closing = CODE_TAGS
    return (
        'There is no function-calling API here: never emit a tool call or JSON like {"name": ...}. '
        f"Reply in plain text: a Thought, then one {opening}...{closing} block that calls the tools as Python "
        f"functions. You have at most {max_steps} steps."
    )


def _result(
    agent: CodeAgent, q: Question, started: float, *, answer: str = "", raw: str = "", error: str | None = None
) -> AnswerResult:
    """Assemble the result of a run that started at ``started`` (a ``time.monotonic`` reading)."""
    return AnswerResult(
        task_id=q.task_id,
        answer=mask_secrets(answer),
        raw_answer=mask_secrets(raw),
        steps=sum(isinstance(step, ActionStep) for step in agent.memory.steps),
        seconds=round(time.monotonic() - started, 2),
        error=error,
        model=str(getattr(agent.model, "model_id", "") or ""),
    )


def _unwrap_parts(output: object) -> object:
    """The text of ``output`` when it is a list of message parts (``[{"type": "text", "text": ...}]``), else ``output``.

    smolagents returns the content of a model message as the answer forced at the step limit, and
    that content can come as parts; its error report for a failed forced answer has this shape too.
    """
    if isinstance(output, str) or not isinstance(output, Sequence) or not output:
        return output
    if not all(isinstance(part, Mapping) and part.get("type") == "text" for part in output):
        return output
    return "".join(str(part.get("text", "")) for part in output)


def _describe(exc: Exception) -> str:
    """``"<Type>: <message>"`` for ``exc``, looking through the wrapper smolagents puts around model errors."""
    root = exc.__cause__ if isinstance(exc, AgentGenerationError) and exc.__cause__ is not None else exc
    return _clean_error(f"{type(root).__name__}: {root}")


def _clean_error(text: str) -> str:
    """One line, at most ``MAX_ERROR_CHARS`` characters, with every known secret replaced by ``***``."""
    text = " ".join(mask_secrets(text).split())
    return text if len(text) <= MAX_ERROR_CHARS else text[: MAX_ERROR_CHARS - 3] + "..."
