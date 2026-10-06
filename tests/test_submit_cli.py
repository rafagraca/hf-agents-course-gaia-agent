"""Tests for ``python -m gaia_agent.submit``: only the answers of a final run, a dry run by default.

``responses`` stands in for the scoring API, so nothing is ever sent. Every task id and answer is invented.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
import responses
from api_fakes import QUESTIONS_JSON, seed_cache
from final_fakes import COMMIT, OTHER_COMMIT, TAG, FakeGit

from gaia_agent import submit
from gaia_agent.config import DEFAULT_API_URL, Settings
from gaia_agent.runs import append_record, new_run_dir

SUBMIT_URL = f"{DEFAULT_API_URL}/submit"
USER = "some-user"
CODE = f"https://github.com/some-user/some-repo/tree/{TAG}"
IDENTITY = ["--username", USER, "--agent-code", CODE]
ANSWERS = {"task-001": "Answer-Four", "task-002": "Answer-Spider", "task-003": "Answer-Lisbon"}
REPLY = {
    "username": USER,
    "score": 66.7,
    "correct_count": 2,
    "total_attempted": 3,
    "message": "Submission Successful! (invented reply)",
}


@pytest.fixture
def rsps() -> Iterator[responses.RequestsMock]:
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


@pytest.fixture(autouse=True)
def tag_names_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The repository of the tests is not the one that was tagged: git answers as if the tag still named COMMIT."""
    monkeypatch.setattr(submit, "git_runner", lambda cwd: FakeGit())


def seal(folder: Path) -> None:
    """Record in ``run.json`` the hash of the answers file as it is now (what the end of a final run does)."""
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    info["answers_sha256"] = hashlib.sha256((folder / "answers.jsonl").read_bytes()).hexdigest()
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")


def make_run(settings: Settings, answers: dict[str, str], *, info: object = "final", day: int = 1) -> Path:
    """A run folder with the given answers and a ``run.json`` (a final run unless ``info`` says otherwise)."""
    seed_cache(settings)
    folder = new_run_dir(settings.runs_dir, datetime(2026, 1, day))
    (folder / "answers.jsonl").touch()
    for task_id, answer in answers.items():
        append_record(folder / "answers.jsonl", {"task_id": task_id, "answer": answer, "error": None})
    if info == "final":
        info = {"final": True, "commit": COMMIT, "tag": TAG}
    if info is not None:
        (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
        if isinstance(info, dict) and info.get("final") is True:
            seal(folder)
    return folder


@pytest.fixture
def run_dir(settings: Settings) -> Path:
    return make_run(settings, ANSWERS)


def saved(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))


def test_by_default_nothing_is_sent(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    assert submit.main(IDENTITY) == 0

    assert len(rsps.calls) == 0
    out = capsys.readouterr().out
    assert "Dry run: 3 answers" in out
    assert "--yes" in out
    record = saved(run_dir)
    assert record["dry_run"] is True
    assert record["reply"] is None
    assert record["tag"] == TAG
    assert record["answers"] == [
        {"task_id": task_id, "submitted_answer": answer} for task_id, answer in ANSWERS.items()
    ]


def test_yes_sends_the_answers_and_shows_the_score(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    rsps.add(responses.POST, SUBMIT_URL, json=REPLY)

    assert submit.main([*IDENTITY, "--yes"]) == 0

    [call] = rsps.calls
    body = json.loads(call.request.body)
    assert body == {"username": USER, "agent_code": CODE, "answers": saved(run_dir)["answers"]}
    assert call.request.req_kwargs["timeout"] == 180
    out = capsys.readouterr().out
    assert "score 66.7% (2/3 correct)" in out
    assert REPLY["message"] in out
    record = saved(run_dir)
    assert (record["dry_run"], record["reply"]) == (False, REPLY)


def test_only_the_four_figures_of_the_reply_are_printed(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    rsps.add(responses.POST, SUBMIT_URL, json={**REPLY, "details": "per-question verdicts", "extra": "leak"})

    submit.main([*IDENTITY, "--yes"])

    printed = capsys.readouterr()
    assert "per-question verdicts" not in printed.out + printed.err
    assert "leak" not in printed.out + printed.err


def test_the_answers_are_never_printed(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    rsps.add(responses.POST, SUBMIT_URL, json=REPLY)

    submit.main(IDENTITY)
    submit.main([*IDENTITY, "--yes"])

    printed = capsys.readouterr()
    for answer in ANSWERS.values():
        assert answer not in printed.out + printed.err


def test_a_redirect_is_not_followed(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    rsps.add(responses.POST, SUBMIT_URL, status=307, headers={"Location": "https://elsewhere.example.test/submit"})
    rsps.add(responses.POST, "https://elsewhere.example.test/submit", json=REPLY)

    assert submit.main([*IDENTITY, "--yes"]) == 1

    assert [call.request.url for call in rsps.calls] == [SUBMIT_URL]
    assert "HTTP 307" in capsys.readouterr().err


# --- only a final run, only a code link that names its tag ---------------------------------------------------------


@pytest.mark.parametrize("info", [None, {"final": False}, {"commit": COMMIT, "tag": TAG}, {"final": "true"}])
def test_a_run_that_is_not_final_is_refused(
    settings: Settings, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str], info: object
) -> None:
    folder = make_run(settings, ANSWERS, info=info)

    assert submit.main([*IDENTITY, "--run-dir", folder.name, "--yes"]) == 2

    assert "not a final run" in capsys.readouterr().err
    assert len(rsps.calls) == 0
    assert not (folder / "submission.json").exists()


def test_a_spoiled_run_json_is_refused(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    folder = make_run(settings, ANSWERS)
    (folder / "run.json").write_text("{not json", encoding="utf-8")

    assert submit.main([*IDENTITY, "--run-dir", folder.name]) == 2

    assert "run.json" in capsys.readouterr().err


def test_the_latest_run_is_used_by_default_and_must_be_final(
    settings: Settings, run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(settings, ANSWERS, info=None, day=2)

    assert submit.main(IDENTITY) == 2

    assert "not a final run" in capsys.readouterr().err


def test_the_run_folder_can_be_chosen(settings: Settings, run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_run(settings, {"task-001": "x"}, info=None, day=2)

    assert submit.main([*IDENTITY, "--run-dir", run_dir.name]) == 0

    assert "Dry run: 3 answers" in capsys.readouterr().out


@pytest.mark.parametrize(
    "code",
    [
        f"https://github.com/some-user/some-repo/tree/{COMMIT}",  # the commit is as good as the tag
        f"https://github.com/Some-User/some_repo.v2/tree/{TAG}",
    ],
)
def test_a_link_to_the_tag_or_the_commit_is_accepted(run_dir: Path, code: str) -> None:
    assert submit.main(["--username", USER, "--agent-code", code]) == 0


@pytest.mark.parametrize(
    "code",
    [
        "https://github.com/some-user/some-repo/tree/main",  # a branch moves
        f"https://github.com/some-user/some-repo/tree/{OTHER_COMMIT}",
        "https://github.com/some-user/some-repo/tree/v1.0.1",
        f"https://github.com/some-user/some-repo/tree/{TAG}/",
        f"https://github.com/some-user/some-repo/tree/{TAG}?plain=1",
        f"https://github.com/some-user/some-repo/tree/{TAG}#readme",
        f"https://github.com/some-user/some-repo/blob/{TAG}/README.md",
        f"https://github.com/some-user/some-repo/{TAG}",
        "https://github.com/some-user/some-repo",
        f"http://github.com/some-user/some-repo/tree/{TAG}",
        f"https://gitlab.com/some-user/some-repo/tree/{TAG}",
        f"https://github.com.evil.example/some-user/some-repo/tree/{TAG}",
        f"https://github.com/some-user/../tree/{TAG}",
        f"https://github.com/some-user/some-repo/tree/{TAG} ",
        f"https://github.com/some-user/some repo/tree/{TAG}",
        f"https://github.com/-bad/some-repo/tree/{TAG}",
        "https://code.example.test/tree/main",
    ],
)
def test_a_code_link_that_does_not_name_the_tag_of_the_run_is_refused(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str], code: str
) -> None:
    assert submit.main(["--username", USER, "--agent-code", code, "--yes"]) == 2

    assert "--agent-code" in capsys.readouterr().err
    assert len(rsps.calls) == 0
    assert not (run_dir / "submission.json").exists()


def test_the_error_says_which_tag_the_link_must_name(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    submit.main(["--username", USER, "--agent-code", "https://github.com/some-user/some-repo/tree/main"])

    assert f"/tree/{TAG}" in capsys.readouterr().err


# --- every question needs an answer --------------------------------------------------------------------------------


def test_a_question_without_an_answer_blocks_the_submission(
    settings: Settings, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_run(settings, {"task-001": "a", "task-003": "c"})

    assert submit.main([*IDENTITY, "--yes"]) == 1

    err = capsys.readouterr().err
    assert "1 of 3 questions have no answer" in err
    assert "task-002" in err
    assert len(rsps.calls) == 0
    assert not (folder / "submission.json").exists()


def test_a_failed_attempt_is_no_answer(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    folder = make_run(settings, {"task-001": "a", "task-003": "c"})
    append_record(folder / "answers.jsonl", {"task_id": "task-002", "answer": "", "error": "RuntimeError: boom"})
    seal(folder)

    assert submit.main(IDENTITY) == 1

    assert "no answer" in capsys.readouterr().err


def test_a_run_without_answers_has_nothing_to_submit(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    make_run(settings, {})

    assert submit.main(IDENTITY) == 1

    assert "3 of 3 questions have no answer" in capsys.readouterr().err


def test_invalid_answers_are_refused_before_anything_is_sent(
    settings: Settings, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_run(settings, {**ANSWERS, "task-001": "FINAL ANSWER: 4"})

    assert submit.main([*IDENTITY, "--yes"]) == 1

    assert len(rsps.calls) == 0
    assert "FINAL ANSWER" in capsys.readouterr().err
    assert not (folder / "submission.json").exists()


def test_an_answer_for_a_task_that_is_not_in_the_list_is_refused(
    settings: Settings, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(settings, {**ANSWERS, "task-999": "x"})

    assert submit.main([*IDENTITY, "--yes"]) == 1

    assert "unknown task_id 'task-999'" in capsys.readouterr().err
    assert len(rsps.calls) == 0


def test_the_question_list_that_cannot_be_fetched_is_reported(
    settings: Settings, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(settings, ANSWERS)
    (settings.cache_dir / "questions.json").unlink()
    rsps.add(responses.GET, f"{DEFAULT_API_URL}/questions", status=404, json={"detail": "gone"})

    assert submit.main(IDENTITY) == 1

    assert "HTTP 404" in capsys.readouterr().err


# --- the rest -------------------------------------------------------------------------------------------------------


def test_a_refused_submission_is_reported(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    rsps.add(responses.POST, SUBMIT_URL, status=422, json={"detail": "invalid payload"})

    assert submit.main([*IDENTITY, "--yes"]) == 1

    assert "HTTP 422: invalid payload" in capsys.readouterr().err
    assert saved(run_dir)["reply"] is None


def test_a_failure_to_save_the_record_is_only_a_warning(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def full_disk(path: Path, text: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(submit, "write_text", full_disk)

    assert submit.main(IDENTITY) == 0
    assert "could not save submission.json" in capsys.readouterr().err


def test_a_reply_without_a_message_is_fine(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    rsps.add(responses.POST, SUBMIT_URL, json={**REPLY, "message": ""})

    assert submit.main([*IDENTITY, "--yes"]) == 0
    assert capsys.readouterr().out.strip().endswith("(2/3 correct).")


def test_a_username_with_capital_letters_gets_a_warning(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert submit.main(["--username", "Some-User", "--agent-code", CODE]) == 0

    err = capsys.readouterr().err
    assert "warning" in err and "capital letters" in err


def test_a_lower_case_username_gets_no_warning(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert submit.main(IDENTITY) == 0

    assert capsys.readouterr().err == ""


def test_without_any_run_there_is_nothing_to_submit(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    assert submit.main(IDENTITY) == 2

    assert "no run with answers" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["--username", "--agent-code"])
def test_username_and_agent_code_are_required(missing: str) -> None:
    args = [arg for pair in zip(IDENTITY[::2], IDENTITY[1::2], strict=True) if pair[0] != missing for arg in pair]

    with pytest.raises(SystemExit) as stopped:
        submit.main(args)

    assert stopped.value.code == 2


def test_the_questions_fixture_is_the_invented_one() -> None:
    assert [item["task_id"] for item in QUESTIONS_JSON] == ["task-001", "task-002", "task-003"]
