"""Tests for the attachment sandbox of the file tools and for the errors ``read_file`` reports.

Every file is generated inside the test (invented content only) and lives in the
temporary DATA folder, so nothing touches the network or the real DATA folder.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gaia_agent.config import Settings
from gaia_agent.tools._local_files import (
    AttachmentSandbox,
    ToolError,
    scratch_folder,
)
from gaia_agent.tools.files import ReadFileTool, RunPythonFileTool
from gaia_agent.tools.media import DescribeImageTool, TranscribeAudioTool


@pytest.fixture
def attachments(settings: Settings) -> Path:
    """The folder where question attachments live (inside DATA)."""
    return settings.files_dir


@pytest.fixture
def read_file(settings: Settings) -> ReadFileTool:
    return ReadFileTool(settings)


# --------------------------------------------------------------------------- sandbox


def test_sandbox_accepts_a_file_inside_data(settings: Settings, attachments: Path) -> None:
    target = attachments / "note.txt"
    target.write_text("hello", encoding="utf-8")

    assert AttachmentSandbox.from_settings(settings).resolve_file(str(target)) == target.resolve()


def test_sandbox_resolves_a_bare_file_name_in_the_attachments_folder(settings: Settings, attachments: Path) -> None:
    target = attachments / "note.txt"
    target.write_text("hello", encoding="utf-8")

    assert AttachmentSandbox.from_settings(settings).resolve_file("note.txt") == target.resolve()


def test_sandbox_finds_a_bare_name_in_the_dataset_download_folder(settings: Settings, attachments: Path) -> None:
    """Files fetched from the Hugging Face dataset land in ``<files>/<split folder>/``."""
    target = attachments / settings.hf_split_dir / "from-dataset.txt"
    target.parent.mkdir(parents=True)
    target.write_text("hello", encoding="utf-8")

    assert AttachmentSandbox.from_settings(settings).resolve_file("from-dataset.txt") == target.resolve()


def test_sandbox_prefers_the_attachments_folder_over_the_dataset_folder(settings: Settings, attachments: Path) -> None:
    first = attachments / "same-name.txt"
    first.write_text("api copy", encoding="utf-8")
    second = attachments / settings.hf_split_dir / "same-name.txt"
    second.parent.mkdir(parents=True)
    second.write_text("dataset copy", encoding="utf-8")

    assert AttachmentSandbox.from_settings(settings).resolve_file("same-name.txt") == first.resolve()


def test_sandbox_accepts_forward_slashes_and_quotes(settings: Settings, attachments: Path) -> None:
    target = attachments / "note.txt"
    target.write_text("hello", encoding="utf-8")
    quoted = '"' + str(target).replace("\\", "/") + '"'

    assert AttachmentSandbox.from_settings(settings).resolve_file(quoted) == target.resolve()


@pytest.mark.parametrize("raw", ["", "   ", '""', "' '", None, 42, b"bytes.txt", ["a.txt"]])
def test_sandbox_rejects_an_empty_or_non_text_path(settings: Settings, raw: object) -> None:
    with pytest.raises(ToolError, match="file_path"):
        AttachmentSandbox.from_settings(settings).resolve_file(raw)


def test_sandbox_reports_a_path_the_operating_system_cannot_resolve(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = AttachmentSandbox.from_settings(settings)

    def unresolvable(self: Path, strict: bool = False) -> Path:
        raise OSError("path is too long")

    monkeypatch.setattr(Path, "resolve", unresolvable)

    with pytest.raises(ToolError, match="invalid file path"):
        sandbox.resolve_file("anything.txt")


def test_sandbox_rejects_a_path_with_a_null_byte(settings: Settings) -> None:
    with pytest.raises(ToolError, match="invalid"):
        AttachmentSandbox.from_settings(settings).resolve_file("note\x00.txt")


def test_sandbox_rejects_a_file_outside_data(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    with pytest.raises(ToolError, match="outside"):
        AttachmentSandbox.from_settings(settings).resolve_file(str(outside))


def test_sandbox_rejects_parent_directory_traversal(settings: Settings, attachments: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    with pytest.raises(ToolError, match="outside"):
        AttachmentSandbox.from_settings(settings).resolve_file("../../../outside.txt")


def link_directory(link: Path, target: Path) -> None:
    """Create a directory symlink, or a junction on Windows accounts that cannot make symlinks."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False)
        if made.returncode == 0:
            return
    pytest.skip("this account can create neither symlinks nor junctions")


def test_sandbox_rejects_a_link_that_leaves_data(settings: Settings, attachments: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside-folder"
    outside.mkdir()
    (outside / "secret.txt").write_text("private", encoding="utf-8")
    link_directory(attachments / "shortcut", outside)

    with pytest.raises(ToolError, match="outside"):
        AttachmentSandbox.from_settings(settings).resolve_file("shortcut/secret.txt")


@pytest.mark.parametrize("folder", ["gaia_dir", "runs_dir", "cache_dir", "data_dir", "tmp_dir"])
def test_sandbox_is_the_attachments_folder_and_nothing_else_in_data(settings: Settings, folder: str) -> None:
    """The answer key (``gaia``), earlier runs, the question list and the scratch folders are all off limits."""
    secret = getattr(settings, folder) / "metadata.jsonl"
    secret.write_text("{}", encoding="utf-8")

    with pytest.raises(ToolError, match="outside"):
        AttachmentSandbox.from_settings(settings).resolve_file(str(secret))


@pytest.mark.parametrize("name", ["../questions.json", "../../gaia/metadata.jsonl", "../../runs/answers.jsonl"])
def test_sandbox_does_not_let_a_relative_path_climb_out_of_the_attachments(settings: Settings, name: str) -> None:
    (settings.cache_dir / "questions.json").write_text("[]", encoding="utf-8")
    (settings.gaia_dir / "metadata.jsonl").write_text("{}", encoding="utf-8")
    (settings.runs_dir / "answers.jsonl").write_text("{}", encoding="utf-8")

    with pytest.raises(ToolError, match="outside"):
        AttachmentSandbox.from_settings(settings).resolve_file(name)


def test_sandbox_error_does_not_say_what_the_other_folders_hold(settings: Settings) -> None:
    (settings.gaia_dir / "metadata.jsonl").write_text("{}", encoding="utf-8")

    with pytest.raises(ToolError) as caught:
        AttachmentSandbox.from_settings(settings).resolve_file(str(settings.gaia_dir / "metadata.jsonl"))

    assert "benchmark" not in str(caught.value).lower()
    assert "metadata" not in str(caught.value).lower()


def test_sandbox_offers_the_scratch_folder_of_the_data_folder(settings: Settings) -> None:
    assert AttachmentSandbox.from_settings(settings).scratch_dir == settings.tmp_dir


def test_a_scratch_folder_lives_where_it_is_told_and_is_removed_afterwards(tmp_path: Path) -> None:
    with scratch_folder(tmp_path / "scratch-parent", "demo-") as folder:
        assert folder.parent == tmp_path / "scratch-parent"
        assert folder.name.startswith("demo-")
        (folder / "work.txt").write_text("temporary", encoding="utf-8")

    assert not folder.exists()


def test_a_scratch_folder_is_removed_even_when_the_work_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), scratch_folder(tmp_path, "demo-") as folder:
        raise RuntimeError("the work failed")

    assert not folder.exists()


def test_a_scratch_folder_that_cannot_be_removed_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(shutil, "rmtree", lambda *args, **kwargs: None)  # a folder Windows keeps locked

    with caplog.at_level(logging.WARNING, logger="gaia_agent.tools._local_files"), scratch_folder(tmp_path, "demo-"):
        pass

    assert "Could not remove the scratch folder" in caplog.text


def test_sandbox_reports_a_missing_file(settings: Settings) -> None:
    with pytest.raises(ToolError, match="not found"):
        AttachmentSandbox.from_settings(settings).resolve_file("nothing-here.xlsx")


def test_sandbox_rejects_a_directory(settings: Settings, attachments: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        AttachmentSandbox.from_settings(settings).resolve_file(str(attachments))


def test_sandbox_accepts_a_path_object(settings: Settings, attachments: Path) -> None:
    target = attachments / "note.txt"
    target.write_text("hello", encoding="utf-8")

    assert AttachmentSandbox.from_settings(settings).resolve_file(target) == target.resolve()


def test_sandbox_shortens_the_path_it_echoes_back(settings: Settings) -> None:
    with pytest.raises(ToolError) as caught:
        AttachmentSandbox.from_settings(settings).resolve_file("x" * 3000 + ".txt")

    assert len(str(caught.value)) < 300


def test_sandbox_treats_an_unreadable_candidate_as_missing(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = AttachmentSandbox.from_settings(settings)

    def locked(self: Path) -> bool:
        raise PermissionError("locked by another process")

    monkeypatch.setattr(Path, "is_file", locked)

    with pytest.raises(ToolError, match="not found"):
        sandbox.resolve_file("anything.txt")


def test_sandbox_without_settings_uses_the_environment_data_dir(settings: Settings, attachments: Path) -> None:
    target = attachments / "env.txt"
    target.write_text("hello", encoding="utf-8")

    assert AttachmentSandbox.from_settings().resolve_file("env.txt") == target.resolve()


# --------------------------------------------------------------------------- read_file: errors


def test_read_file_refuses_a_file_outside_data(read_file: ReadFileTool, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    result = read_file(str(outside))

    assert result.startswith("Error:")
    assert "outside" in result
    assert "private" not in result


def test_read_file_refuses_the_repository_sources(read_file: ReadFileTool) -> None:
    result = read_file(str(Path(__file__)))

    assert result.startswith("Error:")


def test_read_file_reports_a_missing_file(read_file: ReadFileTool) -> None:
    assert read_file("missing.csv").startswith("Error: file not found")


def test_read_file_works_without_explicit_settings(attachments: Path) -> None:
    (attachments / "plain.txt").write_text("from the environment", encoding="utf-8")

    assert ReadFileTool()("plain.txt") == "from the environment"


def test_building_a_tool_does_not_touch_the_disk(data_dir: Path) -> None:
    for tool_class in (ReadFileTool, RunPythonFileTool, TranscribeAudioTool, DescribeImageTool):
        tool_class()

    assert not data_dir.exists()


def test_a_tool_reports_a_data_folder_inside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

    result = ReadFileTool()("anything.txt")

    assert result.startswith("Error: the data folder is not usable")
    assert "outside the repository" in result
