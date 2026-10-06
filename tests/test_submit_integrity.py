"""``submit`` checks the proofs of a final run before it sends anything: the tag, the seal and the single use.

Same doubles as ``test_submit_cli``: ``responses`` stands in for the scoring API and ``git`` is scripted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import responses
from final_fakes import OTHER_COMMIT, TAG, FakeGit
from test_submit_cli import (  # noqa: F401  (fixtures too)
    IDENTITY,
    REPLY,
    SUBMIT_URL,
    make_run,
    rsps,
    run_dir,
    saved,
    tag_names_the_commit,
)

from gaia_agent import submit
from gaia_agent.config import REPO_ROOT, Settings

ANSWERS = {"task-001": "Answer-Four", "task-002": "Answer-Spider", "task-003": "Answer-Lisbon"}


# --- the code and the answers are the ones of the run ----------------------------------------------------------------


def test_a_tag_that_was_moved_after_the_run_is_refused(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    assert submit.main([*IDENTITY, "--yes"], git=FakeGit(tag_commit=OTHER_COMMIT)) == 2

    err = capsys.readouterr().err
    assert TAG in err
    assert "now names" in err
    assert len(rsps.calls) == 0
    assert not (run_dir / "submission.json").exists()


def test_a_tag_that_does_not_exist_here_is_refused(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    assert submit.main(IDENTITY, git=FakeGit(tag_commit="")) == 2

    assert "does not exist" in capsys.readouterr().err
    assert len(rsps.calls) == 0


def test_the_tag_is_checked_in_the_repository_by_default(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[Path] = []

    def runner(cwd: Path) -> FakeGit:
        asked.append(cwd)
        return FakeGit()

    monkeypatch.setattr(submit, "git_runner", runner)

    assert submit.main(IDENTITY) == 0

    assert asked == [REPO_ROOT]


def test_answers_edited_after_the_run_are_refused(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = (run_dir / "answers.jsonl").read_text(encoding="utf-8").replace("Answer-Four", "Answer-4")
    (run_dir / "answers.jsonl").write_text(lines, encoding="utf-8")

    assert submit.main([*IDENTITY, "--yes"]) == 2

    assert "changed since" in capsys.readouterr().err
    assert len(rsps.calls) == 0


def test_a_run_whose_answers_were_never_sealed_is_refused(
    settings: Settings, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_run(settings, ANSWERS)
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    del info["answers_sha256"]
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")

    assert submit.main([*IDENTITY, "--yes"]) == 2

    assert "not sealed" in capsys.readouterr().err
    assert len(rsps.calls) == 0


def test_the_saved_record_carries_the_hash_of_the_answers(run_dir: Path) -> None:
    submit.main(IDENTITY)

    expected = hashlib.sha256((run_dir / "answers.jsonl").read_bytes()).hexdigest()
    assert saved(run_dir)["answers_sha256"] == expected


# --- a final run is submitted once --------------------------------------------------------------------------------


def test_a_run_that_was_already_submitted_is_not_submitted_again(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    rsps.add(responses.POST, SUBMIT_URL, json=REPLY)
    assert submit.main([*IDENTITY, "--yes"]) == 0

    assert submit.main([*IDENTITY, "--yes"]) == 2

    assert len(rsps.calls) == 1
    assert "already submitted" in capsys.readouterr().err
    assert saved(run_dir)["reply"] == REPLY  # the record of the real submission is kept


def test_a_dry_run_does_not_use_up_the_submission(run_dir: Path, rsps: responses.RequestsMock) -> None:
    rsps.add(responses.POST, SUBMIT_URL, json=REPLY)

    assert submit.main(IDENTITY) == 0
    assert submit.main([*IDENTITY, "--yes"]) == 0

    assert len(rsps.calls) == 1


def test_a_dry_run_after_a_real_submission_is_allowed_but_keeps_the_record_and_the_lock(
    run_dir: Path, rsps: responses.RequestsMock
) -> None:
    rsps.add(responses.POST, SUBMIT_URL, json=REPLY)
    submit.main([*IDENTITY, "--yes"])

    assert submit.main(IDENTITY) == 0

    assert saved(run_dir)["reply"] == REPLY
    assert submit.main([*IDENTITY, "--yes"]) == 2
    assert len(rsps.calls) == 1


def test_a_submission_that_failed_may_be_tried_again(run_dir: Path, rsps: responses.RequestsMock) -> None:
    rsps.add(responses.POST, SUBMIT_URL, status=422, json={"detail": "boom"})
    rsps.add(responses.POST, SUBMIT_URL, json=REPLY)

    assert submit.main([*IDENTITY, "--yes"]) == 1
    assert submit.main([*IDENTITY, "--yes"]) == 0

    assert len(rsps.calls) == 2


def test_a_record_that_cannot_be_read_stops_a_real_submission(
    run_dir: Path, rsps: responses.RequestsMock, capsys: pytest.CaptureFixture[str]
) -> None:
    (run_dir / "submission.json").write_text("{not json", encoding="utf-8")

    assert submit.main([*IDENTITY, "--yes"]) == 2

    assert "submission.json" in capsys.readouterr().err
    assert len(rsps.calls) == 0


def test_a_dry_run_does_not_overwrite_a_record_it_cannot_read(run_dir: Path) -> None:
    (run_dir / "submission.json").write_text("{not json", encoding="utf-8")

    assert submit.main(IDENTITY) == 0

    assert (run_dir / "submission.json").read_text(encoding="utf-8") == "{not json"
