"""Tests for the guards of ``gaia_agent.config``: the data folder must be a dedicated one outside the
repository, the answer key must not be on disk during a run, and ``run_python_file`` is opt-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gaia_agent import config
from gaia_agent.config import ConfigError, Settings, load_settings

REPO_ROOT = Path(config.__file__).resolve().parent.parent


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " 1 "])
def test_running_python_attachments_accepts_the_usual_yes_words(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GAIA_ALLOW_RUN_PYTHON", value)

    assert config.run_python_enabled() is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "2", "maybe"])
def test_running_python_attachments_is_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is not None:
        monkeypatch.setenv("GAIA_ALLOW_RUN_PYTHON", value)

    assert config.run_python_enabled() is False


def test_load_settings_rejects_a_data_dir_that_is_a_drive_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GAIA_DATA_DIR", tmp_path.anchor)

    with pytest.raises(ConfigError, match="dedicated folder"):
        load_settings()


def test_a_home_folder_that_cannot_be_found_does_not_stop_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_home(cls: type[Path]) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", classmethod(no_home))

    assert load_settings().data_dir.name == "gaia-data"


def test_the_answer_key_files_of_a_missing_gaia_folder_are_none(tmp_path: Path) -> None:
    assert config.answer_key_files(Settings(data_dir=tmp_path / "never-created")) == []


def test_load_settings_rejects_the_home_folder_as_the_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("GAIA_DATA_DIR", str(tmp_path))

    with pytest.raises(ConfigError, match="dedicated folder"):
        load_settings()


@pytest.mark.parametrize("ancestor", [REPO_ROOT.parent, REPO_ROOT.parent.parent])
def test_load_settings_rejects_a_parent_of_the_repository_as_the_data_dir(
    monkeypatch: pytest.MonkeyPatch, ancestor: Path
) -> None:
    monkeypatch.setenv("GAIA_DATA_DIR", str(ancestor))

    with pytest.raises(ConfigError, match="dedicated folder"):
        load_settings()
