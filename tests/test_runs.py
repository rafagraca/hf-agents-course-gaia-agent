"""Tests for ``gaia_agent.runs``: run folders, the answers file and the usage log summary. Everything is local."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from gaia_agent import runs
from gaia_agent.runs import (
    RunDirError,
    UsageStats,
    append_record,
    is_answered,
    latest_run_dir,
    load_answers,
    new_run_dir,
    pick_run_dir,
    read_usage,
    resolve_run_dir,
    summarize_usage,
    trace_path,
)

WHEN = datetime(2026, 1, 2, 3, 4, 5)


def record(task_id: str, answer: str = "x", error: str | None = None) -> dict[str, object]:
    return {
        "task_id": task_id,
        "answer": answer,
        "raw_answer": answer,
        "steps": 1,
        "seconds": 1.5,
        "error": error,
        "model": "fake/model",
    }


def usage_line(input_tokens: int | None, output_tokens: int | None = 10, waited: float = 0.0) -> dict[str, object]:
    return {
        "model": "fake/model",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_input_tokens": 100,
        "seconds": 1.0,
        "waited_seconds": waited,
    }


# --- run folders -------------------------------------------------------------------------------------------------


def test_a_new_run_folder_is_named_after_the_time(tmp_path: Path) -> None:
    folder = new_run_dir(tmp_path, WHEN)

    assert folder == tmp_path / "20260102-030405"
    assert folder.is_dir()


def test_a_new_run_folder_never_reuses_an_existing_one(tmp_path: Path) -> None:
    first, second, third = (new_run_dir(tmp_path, WHEN) for _ in range(3))

    assert [first.name, second.name, third.name] == ["20260102-030405", "20260102-030405-2", "20260102-030405-3"]


def test_too_many_runs_in_the_same_second_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runs, "MAX_SAME_SECOND_RUNS", 2)
    new_run_dir(tmp_path, WHEN)
    new_run_dir(tmp_path, WHEN)

    with pytest.raises(RunDirError, match="too many runs"):
        new_run_dir(tmp_path, WHEN)


def test_a_run_folder_can_be_given_by_name_or_by_path(tmp_path: Path) -> None:
    folder = new_run_dir(tmp_path, WHEN)

    assert resolve_run_dir(tmp_path, folder.name) == folder.resolve()
    assert resolve_run_dir(tmp_path, str(folder)) == folder.resolve()


@pytest.mark.parametrize("value", ["..", "../elsewhere", "missing-run", ""])
def test_a_run_folder_must_exist_inside_the_runs_folder(tmp_path: Path, value: str) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (tmp_path / "elsewhere").mkdir()

    with pytest.raises(RunDirError):
        resolve_run_dir(runs, value)


def test_the_runs_folder_itself_is_not_a_run(tmp_path: Path) -> None:
    with pytest.raises(RunDirError, match="inside"):
        resolve_run_dir(tmp_path, str(tmp_path))


def test_a_file_is_not_a_run_folder(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not a folder", encoding="utf-8")

    with pytest.raises(RunDirError, match="no run folder"):
        resolve_run_dir(tmp_path, "notes.txt")


def test_the_latest_run_is_the_newest_folder_holding_answers(tmp_path: Path) -> None:
    older = new_run_dir(tmp_path, datetime(2026, 1, 1))
    newer = new_run_dir(tmp_path, datetime(2026, 1, 3))
    new_run_dir(tmp_path, datetime(2026, 1, 5))  # newest, but nothing was answered in it
    for folder in (older, newer):
        append_record(folder / "answers.jsonl", record("t1"))

    assert latest_run_dir(tmp_path) == newer


def test_runs_started_in_the_same_second_are_ordered_by_number_not_by_text(tmp_path: Path) -> None:
    """``...-10`` comes after ``...-9`` even though the text ``-10`` sorts before ``-9``."""
    folders = [new_run_dir(tmp_path, datetime(2026, 1, 1, 12, 0, 0)) for _ in range(10)]
    for folder in folders:
        append_record(folder / "answers.jsonl", record("t1"))

    assert folders[-1].name == "20260101-120000-10"
    assert latest_run_dir(tmp_path) == folders[-1]


def test_a_later_second_beats_any_suffix_of_an_earlier_one(tmp_path: Path) -> None:
    crowded = [new_run_dir(tmp_path, datetime(2026, 1, 1, 12, 0, 0)) for _ in range(3)]
    later = new_run_dir(tmp_path, datetime(2026, 1, 1, 12, 0, 1))
    for folder in (*crowded, later):
        append_record(folder / "answers.jsonl", record("t1"))

    assert latest_run_dir(tmp_path) == later


def test_a_folder_with_an_unexpected_name_does_not_break_the_ordering(tmp_path: Path) -> None:
    odd = tmp_path / "my-own-run"
    odd.mkdir()
    normal = new_run_dir(tmp_path, datetime(2026, 1, 1))
    for folder in (odd, normal):
        append_record(folder / "answers.jsonl", record("t1"))

    assert latest_run_dir(tmp_path) in (odd, normal)


def test_there_is_no_latest_run_without_answers(tmp_path: Path) -> None:
    assert latest_run_dir(tmp_path) is None
    assert latest_run_dir(tmp_path / "missing") is None


def test_picking_a_run_uses_the_given_folder_or_else_the_latest_one(tmp_path: Path) -> None:
    first = new_run_dir(tmp_path, datetime(2026, 1, 1))
    second = new_run_dir(tmp_path, datetime(2026, 1, 2))
    for folder in (first, second):
        append_record(folder / "answers.jsonl", record("t1"))

    assert pick_run_dir(tmp_path, first.name) == first.resolve()
    assert pick_run_dir(tmp_path, None) == second


def test_picking_a_run_fails_clearly_when_there_is_none(tmp_path: Path) -> None:
    with pytest.raises(RunDirError, match="no run with answers"):
        pick_run_dir(tmp_path, None)


def test_traces_are_named_after_a_safe_version_of_the_task_id(tmp_path: Path) -> None:
    assert trace_path(tmp_path, "task-001") == tmp_path / "traces" / "task-001.md"
    assert trace_path(tmp_path, "../odd id") == tmp_path / "traces" / "___odd_id.md"


# --- answers file ------------------------------------------------------------------------------------------------


def test_records_are_appended_one_json_line_each(tmp_path: Path) -> None:
    path = tmp_path / "run" / "answers.jsonl"

    append_record(path, record("t1", "Lisbon"))
    append_record(path, record("t2", "Ação"))

    lines = path.read_bytes().split(b"\n")
    assert lines[-1] == b""
    assert [json.loads(line)["answer"] for line in lines[:-1]] == ["Lisbon", "Ação"]


def test_loading_keeps_the_latest_record_of_each_task(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    for item in (record("t1", "", error="RuntimeError: boom"), record("t2", "b"), record("t1", "a")):
        append_record(path, item)

    loaded = load_answers(path)

    assert list(loaded) == ["t1", "t2"]
    assert loaded["t1"]["answer"] == "a"


def test_a_missing_answers_file_means_no_answers(tmp_path: Path) -> None:
    assert load_answers(tmp_path / "answers.jsonl") == {}


def test_unreadable_lines_are_skipped_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text('not json\n["a list"]\n{"answer": "no task id"}\n\n' + json.dumps(record("t1")) + "\n", "utf-8")

    with caplog.at_level(logging.WARNING):
        loaded = load_answers(path)

    assert list(loaded) == ["t1"]
    assert caplog.text.count("Skipping line") == 3


def test_an_answer_with_unusual_line_separators_survives_the_round_trip(tmp_path: Path) -> None:
    """``json.dumps`` leaves U+2028 and U+0085 alone, and ``str.splitlines`` would cut the record there."""
    path = tmp_path / "answers.jsonl"
    answer = f"first{chr(0x2028)}second{chr(0x85)}third{chr(0x2029)}fourth{chr(0x0C)}fifth"

    append_record(path, record("t1", answer))
    append_record(path, record("t2", "plain"))

    loaded = load_answers(path)
    assert loaded["t1"]["answer"] == answer
    assert list(loaded) == ["t1", "t2"]


def test_a_line_that_is_not_valid_utf8_is_skipped_instead_of_breaking_the_whole_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "answers.jsonl"
    good = json.dumps(record("t1", "ok")).encode("utf-8")
    other = json.dumps(record("t2", "ok too")).encode("utf-8")
    path.write_bytes(good + b"\n" + bytes([0xFF, 0xFE]) + b" broken line\n" + other + b"\n")

    with caplog.at_level(logging.WARNING):
        loaded = load_answers(path)

    assert list(loaded) == ["t1", "t2"]
    assert caplog.text.count("Skipping line") == 1


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (None, False),
        (record("t1", "Lisbon"), True),
        (record("t1", ""), False),
        (record("t1", "x", error="RuntimeError: boom"), False),
        ({"task_id": "t1"}, False),
        ({"task_id": "t1", "answer": 42, "error": None}, False),
    ],
)
def test_only_an_answer_without_an_error_counts_as_answered(item: dict[str, object] | None, expected: bool) -> None:
    assert is_answered(item) is expected


# --- usage log ---------------------------------------------------------------------------------------------------


def test_usage_lines_are_read_in_order_skipping_bad_ones(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "usage.jsonl"
    path.write_text(json.dumps(usage_line(10)) + "\n{broken\n" + json.dumps(usage_line(20)) + "\n", "utf-8")

    with caplog.at_level(logging.WARNING):
        lines = read_usage(path)

    assert [line["input_tokens"] for line in lines] == [10, 20]
    assert "Skipping line" in caplog.text


def test_a_missing_usage_log_is_empty(tmp_path: Path) -> None:
    assert read_usage(tmp_path / "usage.jsonl") == []


def test_usage_is_summarised_over_the_calls_that_report_tokens() -> None:
    stats = summarize_usage([usage_line(1000, 50, 2.5), usage_line(3000, 150), usage_line(None, None, 1.0)])

    assert stats == UsageStats(
        calls=3, max_input=3000, mean_input=2000, total_input=4000, total_output=200, waited_seconds=3.5
    )


def test_the_mean_input_is_rounded_not_cut() -> None:
    stats = summarize_usage([usage_line(1, 0), usage_line(2, 0), usage_line(2, 0)])  # 5 / 3 = 1.67

    assert stats.mean_input == 2


def test_an_empty_usage_log_gives_zeros() -> None:
    assert summarize_usage([]) == UsageStats(
        calls=0, max_input=0, mean_input=0, total_input=0, total_output=0, waited_seconds=0.0
    )


def test_usage_stats_render_as_one_short_line() -> None:
    stats = UsageStats(
        calls=4, max_input=4200, mean_input=3000, total_input=12000, total_output=900, waited_seconds=61.4
    )

    assert stats.describe() == "4 model calls, input tokens max 4200 / mean 3000, output tokens 900, waited 61 s"


def tool_line(input_tokens: int, output_tokens: int) -> dict[str, object]:
    return {**usage_line(input_tokens, output_tokens), "model": "openai/gpt-oss-20b (deep_search)"}


def test_the_calls_a_tool_makes_to_its_own_model_are_counted_apart() -> None:
    stats = summarize_usage([usage_line(1000, 50, 2.5), tool_line(31000, 130), usage_line(3000, 150)])

    assert (stats.calls, stats.max_input, stats.mean_input, stats.total_input) == (2, 3000, 2000, 4000)
    assert (stats.tool_calls, stats.tool_input) == (1, 31000)
    assert stats.total_output == 200


def test_the_tool_calls_appear_in_the_line_only_when_there_are_some() -> None:
    stats = summarize_usage([usage_line(3000, 100, 4.0), tool_line(31000, 130), tool_line(29000, 70)])

    assert stats.describe() == (
        "1 model calls, input tokens max 3000 / mean 3000, output tokens 100, waited 4 s; "
        "2 tool model calls, input tokens 60000"
    )


def test_the_lines_deep_search_writes_count_as_tool_calls() -> None:
    from gaia_agent.tools.deep_search import USAGE_LABEL

    stats = summarize_usage([{**usage_line(20000, 100), "model": USAGE_LABEL}])

    assert (stats.calls, stats.tool_calls, stats.tool_input) == (0, 1, 20000)
