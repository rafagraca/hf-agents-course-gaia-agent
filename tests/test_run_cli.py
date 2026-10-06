"""Tests for ``python -m gaia_agent.run``: a real CodeAgent driven by a scripted model, with a fake scoring client.

The scoring client, the model and the tools are swapped for doubles, so nothing leaves the machine.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from agent_fakes import AUDIO, PLAIN, SHEET, EchoTool, FakeClient, ScriptedModel, code_reply, final_reply, request_text

from gaia_agent import run
from gaia_agent.api import ApiError
from gaia_agent.budget import NoModelAvailableError
from gaia_agent.config import REPO_ROOT, Settings
from gaia_agent.runs import append_record, new_run_dir
from gaia_agent.usage_log import UsageLog

DISTINCT_ANSWER = "Zanzibar-answer"  # never printed: answers stay in the run folder


class UsageWritingModel(ScriptedModel):
    """A scripted model that books one usage line per call, like ``BudgetedModel`` does."""

    def __init__(self, replies: Sequence[object], usage_log: Path | None) -> None:
        super().__init__(replies)
        self.usage = UsageLog(usage_log)

    def generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        message = super().generate(messages, *args, **kwargs)
        self.usage.append(
            self.model_id, tokens=(1000 * len(self.requests), 50), seconds=0.1, estimate=900, max_tokens=2000,
            attempts=1, waited=0.0,
        )  # fmt: skip
        return message


@dataclasses.dataclass
class Harness:
    """What a test controls: the questions and files served, and the replies of the model."""

    client: FakeClient
    replies: list[object]
    models: list[UsageWritingModel] = dataclasses.field(default_factory=list)

    @property
    def requests(self) -> list[list[Any]]:
        return [request for model in self.models for request in model.requests]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    state = Harness(client=FakeClient([PLAIN, SHEET, AUDIO]), replies=[])

    def make_model(settings: Settings, usage_log: Path | None = None, **_: Any) -> UsageWritingModel:
        model = UsageWritingModel(state.replies, usage_log)
        state.models.append(model)
        return model

    monkeypatch.setattr(run, "ScoringClient", lambda settings: state.client)
    monkeypatch.setattr(run, "BudgetedModel", make_model)
    monkeypatch.setattr(run, "default_tools", lambda settings: [EchoTool()])
    return state


def only_run_dir(settings: Settings) -> Path:
    folders = list(settings.runs_dir.iterdir())
    assert len(folders) == 1, folders
    return folders[0]


def records(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_a_run_answers_the_questions_and_saves_answers_and_traces(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.client.questions = [PLAIN]
    harness.replies[:] = [code_reply("print(echo_tool(text='hi'))"), final_reply(DISTINCT_ANSWER)]

    assert run.main([]) == 0

    run_dir = only_run_dir(settings)
    [saved] = records(run_dir)
    assert (saved["task_id"], saved["answer"], saved["steps"], saved["error"]) == (
        PLAIN.task_id, DISTINCT_ANSWER, 2, None,
    )  # fmt: skip
    assert set(saved) == {"task_id", "answer", "raw_answer", "steps", "seconds", "error", "model"}
    trace = (run_dir / "traces" / f"{PLAIN.task_id}.md").read_text(encoding="utf-8")
    assert "echo: hi" in trace
    out = capsys.readouterr().out
    assert str(run_dir) in out
    assert "[1/1] task-pla answered: 2 steps" in out
    assert "2 model calls, input tokens max 2000 / mean 1500" in out
    assert DISTINCT_ANSWER not in out


def test_a_run_refuses_to_start_while_the_answer_key_is_on_disk(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    key = settings.gaia_dir / "2023" / "validation" / "metadata.jsonl"
    key.parent.mkdir(parents=True)
    key.write_text("{}", encoding="utf-8")

    assert run.main([]) == 2

    err = capsys.readouterr().err
    assert "answer key" in err
    assert str(settings.gaia_dir) in err
    assert list(settings.runs_dir.iterdir()) == []
    assert harness.models == [] and harness.client.downloads == []


def test_leftover_download_bookkeeping_counts_as_the_answer_key_too(harness: Harness, settings: Settings) -> None:
    bookkeeping = settings.gaia_dir / ".cache" / "huggingface" / "download" / "metadata.jsonl.metadata"
    bookkeeping.parent.mkdir(parents=True)
    bookkeeping.write_text("etag", encoding="utf-8")

    assert run.main([]) == 2


def test_an_empty_gaia_folder_does_not_stop_a_run(harness: Harness, settings: Settings) -> None:
    (settings.gaia_dir / "notes.txt").write_text("not an answer key", encoding="utf-8")
    harness.replies[:] = [final_reply("a"), final_reply("b"), final_reply("c")]

    assert run.main([]) == 0


def test_the_console_never_shows_what_the_model_wrote_or_the_errors_that_quote_it(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.client.questions = [PLAIN]
    harness.replies[:] = [code_reply("{}['a key the model made up']"), final_reply(DISTINCT_ANSWER)]

    assert run.main([]) == 0

    captured = capsys.readouterr()
    assert "a key the model made up" not in captured.out + captured.err
    assert DISTINCT_ANSWER not in captured.out + captured.err


def test_verbose_shows_the_steps_of_the_agent(harness: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    harness.client.questions = [PLAIN]
    harness.replies[:] = [final_reply(DISTINCT_ANSWER)]

    assert run.main(["--verbose"]) == 0

    assert "Step 1" in capsys.readouterr().out


def test_only_the_chosen_tasks_are_answered_in_the_order_given(harness: Harness, settings: Settings) -> None:
    harness.replies[:] = [final_reply("10"), final_reply("6")]

    assert run.main(["--task", SHEET.task_id, "--task", PLAIN.task_id, "--task", SHEET.task_id]) == 0

    assert [item["task_id"] for item in records(only_run_dir(settings))] == [SHEET.task_id, PLAIN.task_id]


def test_unknown_tasks_are_refused_before_anything_runs(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run.main(["--task", "task-nope", "--task", PLAIN.task_id]) == 2

    assert "task-nope" in capsys.readouterr().err
    assert list(settings.runs_dir.iterdir()) == []
    assert harness.models == []


def test_limit_caps_the_number_of_questions_attempted(harness: Harness, settings: Settings) -> None:
    harness.replies[:] = [final_reply("a"), final_reply("b")]

    assert run.main(["--limit", "2"]) == 0

    assert [item["task_id"] for item in records(only_run_dir(settings))] == [PLAIN.task_id, SHEET.task_id]


@pytest.mark.parametrize("bad", ["0", "-1", "two"])
def test_limit_must_be_a_positive_number(harness: Harness, bad: str) -> None:
    with pytest.raises(SystemExit) as stopped:
        run.main(["--limit", bad])

    assert stopped.value.code == 2


def test_resuming_skips_answered_questions_and_retries_failed_ones(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = new_run_dir(settings.runs_dir, datetime(2026, 1, 1))
    answers = run_dir / "answers.jsonl"
    append_record(answers, {"task_id": PLAIN.task_id, "answer": "42", "error": None})
    append_record(answers, {"task_id": SHEET.task_id, "answer": "", "error": "RuntimeError: boom"})
    harness.replies[:] = [final_reply("10"), final_reply("echo")]

    assert run.main(["--run-dir", run_dir.name]) == 0

    assert [item["task_id"] for item in records(run_dir)][2:] == [SHEET.task_id, AUDIO.task_id]
    assert "1 already answered" in capsys.readouterr().out


def test_attachments_are_downloaded_and_their_name_is_given_to_the_agent(harness: Harness, settings: Settings) -> None:
    attachment = settings.files_dir / SHEET.file_name
    harness.client.files[SHEET.task_id] = attachment
    harness.replies[:] = [final_reply("a"), final_reply("b"), final_reply("c")]

    assert run.main([]) == 0

    assert harness.client.downloads == [SHEET.task_id, AUDIO.task_id]
    tasks = [request_text(request) for request in harness.requests]
    assert f"Attached file: {attachment.name}" in tasks[1]
    assert attachment.as_posix() not in tasks[1]  # never where DATA lives
    assert str(settings.data_dir) not in tasks[1]
    assert "attached .mp3 file, but it could not be downloaded" in tasks[2]


def test_a_failed_question_is_recorded_and_the_run_goes_on(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.client.questions = [PLAIN, SHEET]
    harness.replies[:] = [RuntimeError("provider hiccup"), final_reply("b")]

    assert run.main([]) == 1

    saved = records(only_run_dir(settings))
    assert [item["error"] for item in saved] == ["RuntimeError: provider hiccup", None]
    out = capsys.readouterr().out
    assert "[1/2] task-pla FAILED (RuntimeError: provider hiccup)" in out
    assert "1 answered, 1 failed" in out


def test_an_empty_answer_counts_as_a_failure(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.client.questions = [PLAIN]
    # The check on final answers refuses an empty one every time; the forced answer at the step limit is empty too.
    harness.replies[:] = [final_reply("")] * 11

    assert run.main([]) == 1

    assert "[1/1] task-pla FAILED (empty answer)" in capsys.readouterr().out


def test_running_out_of_models_stops_the_run(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.replies[:] = [NoModelAvailableError("No model can answer: daily quota exhausted")]

    assert run.main([]) == 1

    run_dir = only_run_dir(settings)
    assert [item["task_id"] for item in records(run_dir)] == [PLAIN.task_id]
    assert f"--run-dir {run_dir.name}" in capsys.readouterr().out


def test_the_question_list_can_be_refreshed(harness: Harness) -> None:
    harness.client.questions = [PLAIN]
    harness.replies[:] = [final_reply("x")]

    run.main(["--refresh"])

    assert harness.client.refreshes == [True]


def test_a_run_folder_outside_the_runs_folder_is_refused(
    harness: Harness, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "elsewhere").mkdir()

    assert run.main(["--run-dir", str(tmp_path / "elsewhere")]) == 2
    assert "must be inside" in capsys.readouterr().err


def test_a_failure_to_fetch_the_questions_is_reported(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(refresh: bool = False) -> None:
        raise ApiError("GET /questions returned HTTP 503")

    monkeypatch.setattr(harness.client, "get_questions", broken)

    assert run.main([]) == 2


def test_a_data_folder_inside_the_repository_is_refused(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GAIA_DATA_DIR", str(REPO_ROOT / "data"))

    assert run.main([]) == 2
    assert "outside the repository" in capsys.readouterr().err


def test_an_interrupt_stops_the_run_and_keeps_what_was_saved(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.replies[:] = [final_reply("a"), KeyboardInterrupt()]

    assert run.main([]) == 130

    assert [item["task_id"] for item in records(only_run_dir(settings))] == [PLAIN.task_id]
    assert "Interrupted" in capsys.readouterr().err


def test_a_write_failure_is_reported(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def full_disk(path: Path, record: dict[str, Any]) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(run, "append_record", full_disk)
    harness.replies[:] = [final_reply("a")]

    assert run.main([]) == 1
    assert "No space left on device" in capsys.readouterr().err


def test_nothing_to_do_is_said_plainly(
    harness: Harness, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = new_run_dir(settings.runs_dir, datetime(2026, 1, 1))
    for question in (PLAIN, SHEET, AUDIO):
        append_record(run_dir / "answers.jsonl", {"task_id": question.task_id, "answer": "x", "error": None})

    assert run.main(["--run-dir", run_dir.name]) == 0

    assert "Nothing to do" in capsys.readouterr().out
    assert harness.models == []
