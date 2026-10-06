"""Tests for ``run --wait-for-quota``, ``--defer-missing-attachments`` and the reset of the tools per question."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from agent_fakes import AUDIO, PLAIN, SHEET, EchoTool, FakeClient, final_reply
from budget_fakes import FakeClock
from test_run_cli import UsageWritingModel, only_run_dir, records

from gaia_agent import run
from gaia_agent._run_args import parse_args
from gaia_agent.budget import NoModelAvailableError
from gaia_agent.config import Settings

QUOTA = NoModelAvailableError("No model can answer: groq/a: daily quota exhausted; groq/b: daily quota exhausted")
NO_KEY = NoModelAvailableError("No model can answer: groq/a: the provider rejected the API key (check it)")


@dataclasses.dataclass
class Rig:
    """One script per model the run builds: a quota wait makes the run build a new one."""

    client: FakeClient
    scripts: list[list[object]]
    models: list[UsageWritingModel] = dataclasses.field(default_factory=list)
    reset_calls: list[str] = dataclasses.field(default_factory=list)
    usage_logs: list[Path | None] = dataclasses.field(default_factory=list)


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Rig:
    state = Rig(client=FakeClient([PLAIN, SHEET]), scripts=[])

    def make_model(settings: Settings, usage_log: Path | None = None, **_: Any) -> UsageWritingModel:
        model = UsageWritingModel(state.scripts[len(state.models)], usage_log)  # type: ignore[arg-type]
        state.models.append(model)
        return model

    class ResettingTool(EchoTool):
        def reset_question(self) -> None:
            state.reset_calls.append("reset")

        def set_usage_log(self, path: Path | None) -> None:
            state.usage_logs.append(path)

    monkeypatch.setattr(run, "ScoringClient", lambda settings: state.client)
    monkeypatch.setattr(run, "BudgetedModel", make_model)
    monkeypatch.setattr(run, "default_tools", lambda settings: [ResettingTool()])
    return state


def main_with(clock: FakeClock, *args: str) -> int:
    return run.main(list(args), clock=clock, sleep=clock.sleep)


# --- --wait-for-quota -------------------------------------------------------------------------------------------


def test_without_the_flag_a_quota_error_still_stops_the_run(
    rig: Rig, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    rig.scripts[:] = [[QUOTA]]
    clock = FakeClock()

    assert main_with(clock) == 1

    assert clock.sleeps == []
    assert [item["task_id"] for item in records(only_run_dir(settings))] == [PLAIN.task_id]
    assert "Stopping" in capsys.readouterr().out


def test_with_the_flag_the_run_waits_and_asks_the_same_question_again(
    rig: Rig, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    rig.client.questions = [PLAIN]
    rig.scripts[:] = [[QUOTA], [final_reply("a")]]
    clock = FakeClock()

    assert main_with(clock, "--wait-for-quota") == 0

    assert clock.sleeps == [900.0]
    assert len(rig.models) == 2  # a new model: the old one remembers that its quota is gone
    [saved] = records(only_run_dir(settings))  # the attempt that hit the quota is not an answer and is not saved
    assert (saved["task_id"], saved["answer"], saved["error"]) == (PLAIN.task_id, "a", None)
    assert "waiting 15 min" in capsys.readouterr().out


def test_the_run_keeps_going_after_the_wait_with_the_next_questions(rig: Rig, settings: Settings) -> None:
    rig.scripts[:] = [[final_reply("a"), QUOTA], [final_reply("b")]]
    clock = FakeClock()

    assert main_with(clock, "--wait-for-quota") == 0

    assert [item["task_id"] for item in records(only_run_dir(settings))] == [PLAIN.task_id, SHEET.task_id]


def test_it_waits_again_while_the_quota_stays_gone(rig: Rig, settings: Settings) -> None:
    rig.client.questions = [PLAIN]
    rig.scripts[:] = [[QUOTA], [QUOTA], [QUOTA], [final_reply("a")]]
    clock = FakeClock()

    assert main_with(clock, "--wait-for-quota") == 0

    assert clock.sleeps == [900.0] * 3


def test_after_the_maximum_wait_the_run_stops_and_saves_the_failure(
    rig: Rig, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    rig.client.questions = [PLAIN]
    rig.scripts[:] = [[QUOTA]] * 8
    clock = FakeClock()

    assert main_with(clock, "--wait-for-quota", "--max-wait-hours", "1") == 1

    assert clock.slept == 3600.0
    [saved] = records(only_run_dir(settings))
    assert saved["error"].startswith("NoModelAvailableError:")
    out = capsys.readouterr().out
    assert "Giving up" in out and "--run-dir" in out


def test_a_rejected_key_is_not_waited_for(rig: Rig, settings: Settings) -> None:
    rig.scripts[:] = [[NO_KEY]]
    clock = FakeClock()

    assert main_with(clock, "--wait-for-quota") == 1

    assert clock.sleeps == []
    assert len(records(only_run_dir(settings))) == 1


def test_a_failure_that_is_not_about_the_quota_does_not_wait(rig: Rig, settings: Settings) -> None:
    rig.client.questions = [PLAIN, SHEET]
    rig.scripts[:] = [[RuntimeError("provider hiccup"), final_reply("b")]]
    clock = FakeClock()

    assert main_with(clock, "--wait-for-quota") == 1

    assert clock.sleeps == []


def test_the_wait_is_in_the_summary(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    rig.client.questions = [PLAIN]
    rig.scripts[:] = [[QUOTA], [final_reply("a")]]

    main_with(FakeClock(), "--wait-for-quota")

    assert "waited 15 min for the quota" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["0", "-2", "many"])
def test_the_maximum_wait_must_be_a_positive_number(rig: Rig, bad: str) -> None:
    with pytest.raises(SystemExit) as stopped:
        run.main(["--wait-for-quota", "--max-wait-hours", bad])

    assert stopped.value.code == 2


def test_the_default_maximum_wait_is_thirty_six_hours() -> None:
    assert parse_args([]).max_wait_hours == 36.0


# --- --defer-missing-attachments -------------------------------------------------------------------------------


def test_without_the_flag_a_missing_attachment_is_answered_without_it(rig: Rig, settings: Settings) -> None:
    rig.client.questions = [SHEET]
    rig.scripts[:] = [[final_reply("a")]]

    assert main_with(FakeClock()) == 0

    assert len(records(only_run_dir(settings))) == 1


def test_a_question_whose_attachment_is_missing_is_left_for_later(
    rig: Rig, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    rig.client.questions = [PLAIN, SHEET, AUDIO]
    rig.scripts[:] = [[final_reply("a")]]

    assert main_with(FakeClock(), "--defer-missing-attachments") == 1

    run_dir = only_run_dir(settings)
    assert [item["task_id"] for item in records(run_dir)] == [PLAIN.task_id]
    assert not (run_dir / "traces" / f"{SHEET.task_id}.md").exists()
    out = capsys.readouterr().out
    assert "[2/3] task-she DEFERRED" in out
    assert "2 deferred" in out
    assert len(rig.models) == 1 and len(rig.models[0].requests) == 1  # the model never saw them


def test_a_later_run_answers_the_deferred_questions_once_the_attachment_exists(rig: Rig, settings: Settings) -> None:
    rig.client.questions = [PLAIN, SHEET]
    rig.scripts[:] = [[final_reply("a")], [final_reply("b")]]
    assert main_with(FakeClock(), "--defer-missing-attachments") == 1
    run_dir = only_run_dir(settings)
    rig.client.files[SHEET.task_id] = settings.files_dir / SHEET.file_name

    assert main_with(FakeClock(), "--defer-missing-attachments", "--run-dir", run_dir.name) == 0

    assert [item["task_id"] for item in records(run_dir)] == [PLAIN.task_id, SHEET.task_id]


def test_a_question_with_its_attachment_is_not_deferred(rig: Rig, settings: Settings) -> None:
    rig.client.questions = [SHEET]
    rig.client.files[SHEET.task_id] = settings.files_dir / SHEET.file_name
    rig.scripts[:] = [[final_reply("a")]]

    assert main_with(FakeClock(), "--defer-missing-attachments") == 0


# --- reset_question ---------------------------------------------------------------------------------------------


def test_the_tools_that_can_reset_are_reset_before_each_question(rig: Rig) -> None:
    rig.client.questions = [PLAIN, SHEET]
    rig.scripts[:] = [[final_reply("a"), final_reply("b")]]

    assert main_with(FakeClock()) == 0

    assert rig.reset_calls == ["reset", "reset"]


def test_a_retry_after_a_quota_wait_resets_again(rig: Rig) -> None:
    rig.client.questions = [PLAIN]
    rig.scripts[:] = [[QUOTA], [final_reply("a")]]

    assert main_with(FakeClock(), "--wait-for-quota") == 0

    assert rig.reset_calls == ["reset", "reset"]


def test_tools_without_a_reset_are_left_alone() -> None:
    run.reset_tools([EchoTool()])
    run.reset_tools([])


def test_a_tool_whose_reset_is_not_callable_is_left_alone() -> None:
    tool = EchoTool()
    tool.reset_question = "not a function"  # type: ignore[attr-defined]

    run.reset_tools([tool])  # must not raise


def test_the_tools_that_log_their_own_use_are_pointed_at_the_usage_log_of_the_run(rig: Rig, settings: Settings) -> None:
    rig.client.questions = [PLAIN, SHEET]
    rig.scripts[:] = [[final_reply("a"), final_reply("b")]]

    assert main_with(FakeClock()) == 0

    assert rig.usage_logs == [only_run_dir(settings) / "usage.jsonl"]  # once, not per question


def test_a_tool_whose_usage_log_hook_is_not_callable_is_left_alone(tmp_path: Path) -> None:
    tool = EchoTool()
    tool.set_usage_log = None  # type: ignore[attr-defined]

    run.attach_usage_log([tool, EchoTool()], tmp_path / "usage.jsonl")  # must not raise
