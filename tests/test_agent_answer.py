"""Tests for ``answer_question``, ``build_task`` and ``is_fatal``: a real CodeAgent driven by a scripted model."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from agent_fakes import (
    AUDIO,
    PLAIN,
    SCRIPTED_MODEL_ID,
    SHEET,
    AttachmentAwareTool,
    EchoTool,
    ScriptedModel,
    code_reply,
    final_reply,
    message_text,
    request_text,
)

from gaia_agent.agent import AnswerResult, answer_question, build_agent, build_task, is_fatal
from gaia_agent.budget import NoModelAvailableError, PromptTooLargeError
from gaia_agent.config import Settings
from gaia_agent.tools.files import RunPythonFileTool

GROQ_SECRET = "groq-secret-for-tests"


def run(settings: Settings, replies: list[object], question=PLAIN, file_path: Path | None = None, **changes):
    """Answer ``question`` with an agent whose model replays ``replies``; return the result and the model."""
    model = ScriptedModel(replies)
    agent = build_agent(dataclasses.replace(settings, **changes), tools=[EchoTool()], model=model)
    return answer_question(agent, question, file_path), model


def task_sent(model: ScriptedModel) -> str:
    """The task message of the first request (the message right after the system prompt)."""
    return message_text(model.requests[0][1])


# --- build_task --------------------------------------------------------------------------------------------------


def test_without_an_attachment_the_task_is_the_question() -> None:
    assert build_task(PLAIN, None) == PLAIN.question


def test_an_attachment_is_named_by_its_file_name_and_never_by_its_path(tmp_path: Path) -> None:
    """The tools find the file by name; a path would show the model (and the provider) where DATA lives."""
    attachment = tmp_path / "files" / "task-sheet.xlsx"

    task = build_task(SHEET, attachment)

    assert task.startswith(SHEET.question)
    assert task.endswith("Attached file: task-sheet.xlsx")
    assert str(tmp_path) not in task
    assert tmp_path.as_posix() not in task
    assert "\\" not in task


def test_a_missing_attachment_is_announced_by_its_type_only() -> None:
    task = build_task(AUDIO, None)

    assert task.startswith(AUDIO.question)
    assert "attached .mp3 file" in task
    assert "could not be downloaded" in task
    assert AUDIO.file_name not in task


# --- successful runs ---------------------------------------------------------------------------------------------


def test_a_run_returns_the_normalised_answer_and_its_details(settings: Settings) -> None:
    result, _ = run(settings, [final_reply("1,234.")])

    assert result.task_id == PLAIN.task_id
    assert result.answer == "1234"
    assert result.raw_answer == "1,234."
    assert result.steps == 1
    assert result.seconds >= 0
    assert result.error is None
    assert result.model == SCRIPTED_MODEL_ID


def test_a_non_text_answer_is_turned_into_text(settings: Settings) -> None:
    result, _ = run(settings, [code_reply("final_answer(6 * 7)")])

    assert (result.answer, result.raw_answer) == ("42", "42")


def test_a_list_answer_becomes_a_comma_separated_text(settings: Settings) -> None:
    result, _ = run(settings, [code_reply("final_answer(['pear', 'fig'])")])

    assert (result.answer, result.raw_answer, result.error) == ("pear, fig", "['pear', 'fig']", None)


def test_an_answer_given_as_message_parts_is_read_as_its_text(settings: Settings) -> None:
    parts = "[{'type': 'text', 'text': 'Porto'}, {'type': 'text', 'text': '.'}]"

    result, _ = run(settings, [code_reply(f"final_answer({parts})")])

    assert (result.answer, result.raw_answer, result.error) == ("Porto", "Porto.", None)


def test_the_question_decides_how_a_list_answer_is_written(settings: Settings) -> None:
    listing = dataclasses.replace(PLAIN, question="Name the two rivers, comma-separated.")

    result, _ = run(settings, [code_reply("final_answer('Elbe;Vlt')")], question=listing)

    assert (result.answer, result.raw_answer) == ("Elbe, Vlt", "Elbe;Vlt")


def test_the_agent_receives_the_task_text(settings: Settings, tmp_path: Path) -> None:
    attachment = tmp_path / "task-sheet.xlsx"

    _, model = run(settings, [final_reply("10")], question=SHEET, file_path=attachment)

    assert build_task(SHEET, attachment) in task_sent(model)


def test_the_attachment_is_registered_with_the_tools_that_want_it_and_withdrawn_afterwards(
    settings: Settings, tmp_path: Path
) -> None:
    tool = AttachmentAwareTool()
    agent = build_agent(
        settings,
        tools=[tool, EchoTool()],
        model=ScriptedModel([code_reply("print(attachment_tool())"), final_reply("x")]),
    )
    attachment = tmp_path / "task-sheet.xlsx"

    answer_question(agent, SHEET, attachment)

    assert tool.registered == [attachment, None]
    assert any(
        "attachment: task-sheet.xlsx" in step.observations
        for step in agent.memory.steps
        if getattr(step, "observations", None)
    )


def test_the_attachment_is_withdrawn_even_when_the_run_fails(settings: Settings, tmp_path: Path) -> None:
    tool = AttachmentAwareTool()
    agent = build_agent(settings, tools=[tool], model=ScriptedModel([RuntimeError("the provider is down")]))
    attachment = tmp_path / "task-sheet.xlsx"

    result = answer_question(agent, SHEET, attachment)

    assert result.error is not None
    assert tool.registered == [attachment, None]


def test_a_question_without_an_attachment_registers_none(settings: Settings) -> None:
    tool = AttachmentAwareTool()
    agent = build_agent(settings, tools=[tool], model=ScriptedModel([final_reply("x")]))

    answer_question(agent, PLAIN, None)

    assert tool.registered == [None, None]


def test_the_real_run_python_tool_runs_the_attachment_and_nothing_else(settings: Settings) -> None:
    (settings.files_dir / "task.py").write_text("print('output of the attachment')", encoding="utf-8")
    (settings.files_dir / "planted.py").write_text("print('output of a planted script')", encoding="utf-8")
    model = ScriptedModel(
        [
            code_reply("print(run_python_file('task.py'))\nprint(run_python_file('planted.py'))"),
            final_reply("done"),
        ]
    )
    agent = build_agent(settings, tools=[RunPythonFileTool(settings)], model=model)
    question = dataclasses.replace(PLAIN, file_name="task.py")

    answer_question(agent, question, settings.files_dir / "task.py")

    seen = request_text(model.requests[1])
    assert "output of the attachment" in seen
    assert "output of a planted script" not in seen
    assert "only the .py attachment of the current question can be run" in seen


def test_tool_calls_count_as_steps_and_their_output_reaches_the_model(settings: Settings) -> None:
    result, model = run(settings, [code_reply("print(echo_tool(text='abc'))"), final_reply("abc")])

    assert (result.steps, result.answer) == (2, "abc")
    assert "echo: abc" in request_text(model.requests[1])


def test_code_that_lost_its_tags_still_runs(settings: Settings) -> None:
    result, model = run(settings, ["\n".join(["Thought: easy.", "final_answer('Lisbon')"])])

    assert (result.answer, result.steps, result.error) == ("Lisbon", 1, None)
    assert len(model.requests) == 1


def test_an_empty_reply_is_asked_again_without_losing_a_step(settings: Settings) -> None:
    result, model = run(settings, ["", final_reply("Lisbon")])

    assert (result.answer, result.steps) == ("Lisbon", 1)
    assert len(model.requests) == 2


def test_each_question_starts_from_a_clean_memory(settings: Settings) -> None:
    model = ScriptedModel([final_reply("first"), final_reply("second")])
    agent = build_agent(settings, tools=[], model=model)

    answer_question(agent, PLAIN, None)
    second = answer_question(agent, dataclasses.replace(PLAIN, task_id="task-other", question="Another one?"), None)

    assert second.answer == "second"
    assert PLAIN.question not in request_text(model.requests[1])


def test_a_forced_answer_written_as_a_final_answer_call_is_read(settings: Settings) -> None:
    forced = "\n".join(["<code>", "final_answer('Lisbon')", "</code>"])

    result, _ = run(settings, [code_reply("x = 1"), forced], max_steps=1)

    assert (result.answer, result.raw_answer, result.error) == ("Lisbon", forced, None)


def test_at_the_step_limit_the_forced_answer_is_normalised(settings: Settings) -> None:
    result, model = run(settings, [code_reply("x = 1"), "FINAL ANSWER: Lisbon."], max_steps=1)

    assert (result.answer, result.raw_answer, result.error) == ("Lisbon", "FINAL ANSWER: Lisbon.", None)
    assert "ONLY the answer" in request_text(model.requests[-1])


def test_a_forced_answer_of_none_is_an_empty_answer_not_the_text_of_the_call(settings: Settings) -> None:
    forced = "\n".join(["<code>", "final_answer(None)", "</code>"])

    result, _ = run(settings, [code_reply("x = 1"), forced], max_steps=1)

    assert (result.answer, result.raw_answer, result.error) == ("", forced, None)


def test_a_tool_call_and_a_final_answer_in_one_block_are_not_answered_blind(settings: Settings) -> None:
    """The prompt asks for ``final_answer`` alone, after the evidence has been seen; the guard holds the model to it."""
    blind = code_reply("echoed = echo_tool(text='abc')\nfinal_answer(echoed)")
    model = ScriptedModel([blind, final_reply("abc")])
    agent = build_agent(settings, tools=[EchoTool()], model=model)

    result = answer_question(agent, PLAIN, None)

    assert (result.answer, result.steps, result.error) == ("abc", 2, None)
    assert "echo: abc" in request_text(model.requests[1])  # what the model saw before it answered


# --- failures ----------------------------------------------------------------------------------------------------


def test_a_model_failure_becomes_an_error_result(settings: Settings) -> None:
    result, _ = run(settings, [RuntimeError("the provider is down")])

    assert result.error == "RuntimeError: the provider is down"
    assert (result.answer, result.raw_answer) == ("", "")
    assert result.steps == 1
    assert not is_fatal(result)


def test_a_prompt_too_large_ends_the_question_only(settings: Settings) -> None:
    result, _ = run(
        settings,
        [code_reply("x = 1"), PromptTooLargeError("about 9000 tokens cannot fit"), RuntimeError("still too big")],
    )

    assert result.error == "PromptTooLargeError: about 9000 tokens cannot fit"
    assert (result.answer, result.raw_answer) == ("", "")
    assert result.steps == 2
    assert not is_fatal(result)


def test_a_prompt_too_large_gets_one_last_try_at_the_best_answer(settings: Settings) -> None:
    result, model = run(
        settings,
        [code_reply("x = 1"), PromptTooLargeError("about 9000 tokens cannot fit"), "FINAL ANSWER: Lisbon."],
    )

    assert (result.answer, result.raw_answer, result.error) == ("Lisbon", "FINAL ANSWER: Lisbon.", None)
    assert "ONLY the answer" in request_text(model.requests[-1])


def test_a_prompt_too_large_whose_last_try_gives_no_answer_is_still_the_original_error(settings: Settings) -> None:
    result, _ = run(settings, [code_reply("x = 1"), PromptTooLargeError("about 9000 tokens cannot fit"), ""])

    assert result.error == "PromptTooLargeError: about 9000 tokens cannot fit"
    assert result.answer == ""


def test_running_out_of_models_is_fatal(settings: Settings) -> None:
    result, _ = run(settings, [NoModelAvailableError("No model can answer: daily quota exhausted")])

    assert result.error.startswith("NoModelAvailableError: No model can answer")
    assert is_fatal(result)


def test_a_failed_forced_answer_is_an_error_not_an_answer(settings: Settings) -> None:
    result, _ = run(settings, [code_reply("x = 1"), RuntimeError("quota gone")], max_steps=1)

    assert result.answer == ""
    assert result.error.startswith("FinalAnswerError: ")
    assert "quota gone" in result.error


def test_secrets_never_reach_the_error_text(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", GROQ_SECRET)

    result, _ = run(settings, [RuntimeError(f"invalid key {GROQ_SECRET}")])

    assert GROQ_SECRET not in result.error
    assert "invalid key ***" in result.error


def test_long_error_texts_are_cut_and_kept_on_one_line(settings: Settings) -> None:
    result, _ = run(settings, [RuntimeError("line one\nline two " + "x" * 5000)])

    assert "\n" not in result.error
    assert len(result.error) <= 600
    assert result.error.startswith("RuntimeError: line one line two")


def test_an_interrupt_is_never_swallowed(settings: Settings) -> None:
    with pytest.raises(KeyboardInterrupt):
        run(settings, [KeyboardInterrupt()])


# --- is_fatal ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "fatal"),
    [
        (None, False),
        ("RuntimeError: boom", False),
        ("PromptTooLargeError: too big", False),
        ("NoModelAvailableError: No model can answer", True),
    ],
)
def test_is_fatal_flags_only_the_end_of_every_model(error: str | None, fatal: bool) -> None:
    result = AnswerResult(task_id="t", answer="", raw_answer="", steps=0, seconds=0.0, error=error, model="m")

    assert is_fatal(result) is fatal


def test_answer_result_is_immutable() -> None:
    result = AnswerResult(task_id="t", answer="a", raw_answer="a", steps=1, seconds=1.0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer = "b"  # type: ignore[misc]
    assert (result.error, result.model) == (None, "")
