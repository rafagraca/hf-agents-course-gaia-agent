"""Tests for gaia_agent.config: settings, data directory layout and secret lookup."""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gaia_agent import config
from gaia_agent.config import ConfigError, Settings, get_secret, load_settings

REPO_ROOT = Path(config.__file__).resolve().parent.parent
SECRET_NAME = "GAIA_TEST_SECRET"
REG_SZ = 1
REG_EXPAND_SZ = 2


# --------------------------------------------------------------------- Settings


def test_settings_defaults_match_the_spec(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path)

    assert s.api_url == "https://agents-course-unit4-scoring.hf.space"
    assert s.model_id == "groq/qwen/qwen3.8-27b"
    assert s.fallback_model_ids == ("groq/openai/gpt-oss-120b",)
    assert (s.tpm_limit, s.max_prompt_tokens, s.max_output_tokens) == (8000, 5000, 2000)
    assert s.max_steps == 10
    assert s.reasoning_effort == "low"
    assert (s.hf_dataset, s.hf_split_dir) == ("gaia-benchmark/GAIA", "2023/validation")


def test_settings_is_immutable(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.max_steps = 3  # type: ignore[misc]


def test_settings_requires_a_data_dir() -> None:
    with pytest.raises(TypeError):
        Settings()  # type: ignore[call-arg]


def test_settings_can_be_copied_with_changes(tmp_path: Path) -> None:
    s = dataclasses.replace(Settings(data_dir=tmp_path), max_steps=3)

    assert s.max_steps == 3
    assert s.data_dir == tmp_path


def test_settings_coerces_a_string_path_and_a_list_of_models(tmp_path: Path) -> None:
    s = Settings(data_dir=str(tmp_path), fallback_model_ids=["groq/a"])  # type: ignore[arg-type]

    assert isinstance(s.data_dir, Path)
    assert s.data_dir == tmp_path
    assert s.fallback_model_ids == ("groq/a",)


def test_settings_exposes_the_data_subfolders(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path)

    assert s.cache_dir == tmp_path / "cache"
    assert s.files_dir == tmp_path / "cache" / "files"
    assert s.runs_dir == tmp_path / "runs"
    assert s.gaia_dir == tmp_path / "gaia"
    assert s.tmp_dir == tmp_path / "tmp"


# --------------------------------------------------------------- load_settings


def test_load_settings_uses_gaia_data_dir_and_creates_the_subfolders(data_dir: Path) -> None:
    s = load_settings()

    assert s.data_dir == data_dir.resolve()
    for folder in (s.cache_dir, s.files_dir, s.runs_dir, s.gaia_dir, s.tmp_dir):
        assert folder.is_dir()


def test_load_settings_is_idempotent(data_dir: Path) -> None:
    first = load_settings()
    (first.cache_dir / "questions.json").write_text("[]", encoding="utf-8")

    second = load_settings()

    assert second == first
    assert (second.cache_dir / "questions.json").read_text(encoding="utf-8") == "[]"


def test_load_settings_defaults_to_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GAIA_DATA_DIR")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    s = load_settings()

    assert s.data_dir == (tmp_path / "local" / "hf-gaia-agent").resolve()
    assert s.cache_dir.is_dir()


def test_load_settings_falls_back_to_the_home_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GAIA_DATA_DIR")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    s = load_settings()

    assert s.data_dir == (tmp_path / ".local" / "share" / "hf-gaia-agent").resolve()


def test_load_settings_treats_a_blank_data_dir_as_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GAIA_DATA_DIR", "   ")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    assert load_settings().data_dir == (tmp_path / "local" / "hf-gaia-agent").resolve()


def test_load_settings_reads_the_model_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_MODEL_ID", " groq/some/model ")
    monkeypatch.setenv("GAIA_FALLBACK_MODELS", "groq/a, groq/b ,, groq/c")

    s = load_settings()

    assert s.model_id == "groq/some/model"
    assert s.fallback_model_ids == ("groq/a", "groq/b", "groq/c")


@pytest.mark.parametrize("blank", ["", "   "])
def test_load_settings_blank_model_id_keeps_the_default(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("GAIA_MODEL_ID", blank)

    assert load_settings().model_id == config.DEFAULT_MODEL_ID


@pytest.mark.parametrize("blank", ["", "   ", " , ,"])
def test_load_settings_blank_fallback_list_keeps_the_default(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("GAIA_FALLBACK_MODELS", blank)

    assert load_settings().fallback_model_ids == config.DEFAULT_FALLBACK_MODEL_IDS


@pytest.mark.parametrize("value", ["none", "NONE", " None ", "-"])
def test_load_settings_none_switches_the_fallback_models_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GAIA_FALLBACK_MODELS", value)

    assert load_settings().fallback_model_ids == ()


def test_load_settings_rejects_a_data_dir_inside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    target = REPO_ROOT / "gaia-data-should-not-exist"
    monkeypatch.setenv("GAIA_DATA_DIR", str(target))

    with pytest.raises(ConfigError, match="outside the repository"):
        load_settings()

    assert not target.exists()


def test_load_settings_rejects_a_relative_data_dir_inside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("GAIA_DATA_DIR", "gaia-data-should-not-exist")

    with pytest.raises(ConfigError, match="outside the repository"):
        load_settings()

    assert not (REPO_ROOT / "gaia-data-should-not-exist").exists()


def test_load_settings_reports_a_data_dir_that_cannot_be_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_home(self: Path) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "expanduser", no_home)
    monkeypatch.setenv("GAIA_DATA_DIR", "~nobody/data")

    with pytest.raises(ConfigError, match="invalid GAIA_DATA_DIR"):
        load_settings()


def test_load_settings_reports_a_data_dir_that_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, not a folder", encoding="utf-8")
    monkeypatch.setenv("GAIA_DATA_DIR", str(blocker / "data"))

    with pytest.raises(ConfigError, match="cannot create"):
        load_settings()


# ------------------------------------------------------------------ get_secret


def test_get_secret_prefers_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_NAME, "from-env")
    monkeypatch.setattr(config, "_persistent_env_reader", lambda name: "from-registry")

    assert get_secret(SECRET_NAME) == "from-env"


def test_get_secret_strips_surrounding_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_NAME, "  value\r\n")

    assert get_secret(SECRET_NAME) == "value"


def test_get_secret_falls_back_to_the_persistent_user_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def reader(name: str) -> str:
        asked.append(name)
        return "stored-value"

    monkeypatch.delenv(SECRET_NAME, raising=False)
    monkeypatch.setattr(config, "_persistent_env_reader", reader)

    assert get_secret(SECRET_NAME) == "stored-value"
    assert asked == [SECRET_NAME]


@pytest.mark.parametrize("env_value", [None, "", "   "])
def test_get_secret_returns_none_when_it_is_not_set_anywhere(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    if env_value is None:
        monkeypatch.delenv(SECRET_NAME, raising=False)
    else:
        monkeypatch.setenv(SECRET_NAME, env_value)

    assert get_secret(SECRET_NAME) is None


def test_get_secret_ignores_a_blank_persistent_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SECRET_NAME, raising=False)
    monkeypatch.setattr(config, "_persistent_env_reader", lambda name: "  ")

    assert get_secret(SECRET_NAME) is None


def test_get_secret_never_logs_the_value(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv(SECRET_NAME, "s3cr3t-value")

    get_secret(SECRET_NAME)

    assert "s3cr3t-value" not in caplog.text


# ------------------------------------------------------ Windows user environment


def _fake_winreg(
    values: dict[str, tuple[object, int]] | None = None,
    *,
    open_error: OSError | None = None,
) -> SimpleNamespace:
    """Build a stand-in for the ``winreg`` module with one ``HKCU\\Environment`` key."""
    stored = values or {}

    class Key:
        def __enter__(self) -> Key:
            return self

        def __exit__(self, *exc_info: object) -> bool:
            return False

    def open_key(hive: object, sub_key: str) -> Key:
        if open_error is not None:
            raise open_error
        assert sub_key == "Environment"
        return Key()

    def query_value_ex(key: Key, name: str) -> tuple[object, int]:
        if name not in stored:
            raise FileNotFoundError(2, "The system cannot find the file specified")
        return stored[name]

    return SimpleNamespace(
        HKEY_CURRENT_USER="HKCU",
        REG_SZ=REG_SZ,
        REG_EXPAND_SZ=REG_EXPAND_SZ,
        OpenKey=open_key,
        QueryValueEx=query_value_ex,
    )


def test_registry_reader_returns_a_stored_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg({SECRET_NAME: ("token-value", REG_SZ)}))

    assert config._read_windows_user_env(SECRET_NAME) == "token-value"


def test_registry_reader_expands_expandable_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_TEST_BASE", "expanded")
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg({SECRET_NAME: ("%GAIA_TEST_BASE%-tail", REG_EXPAND_SZ)}))

    assert config._read_windows_user_env(SECRET_NAME) == "expanded-tail"


def test_registry_reader_does_not_expand_plain_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_TEST_BASE", "expanded")
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg({SECRET_NAME: ("%GAIA_TEST_BASE%-tail", REG_SZ)}))

    assert config._read_windows_user_env(SECRET_NAME) == "%GAIA_TEST_BASE%-tail"


def test_registry_reader_returns_none_for_an_undefined_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg({}))

    assert config._read_windows_user_env(SECRET_NAME) is None


def test_registry_reader_ignores_values_that_are_not_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg({SECRET_NAME: (5, 4)}))

    assert config._read_windows_user_env(SECRET_NAME) is None


def test_registry_reader_warns_and_returns_none_on_os_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(open_error=PermissionError(5, "Access is denied")))
    caplog.set_level(logging.WARNING)

    assert config._read_windows_user_env(SECRET_NAME) is None
    assert SECRET_NAME in caplog.text


def test_registry_reader_is_a_no_op_where_winreg_does_not_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "winreg", None)

    assert config._read_windows_user_env(SECRET_NAME) is None


def test_registry_reader_returns_none_for_an_unknown_variable_in_the_real_registry() -> None:
    assert config._read_windows_user_env("GAIA_AGENT_TEST_SURELY_UNDEFINED_VARIABLE") is None


@pytest.mark.skipif(sys.platform != "win32", reason="needs the Windows registry")
def test_registry_reader_reads_and_expands_a_real_user_variable() -> None:
    value = config._read_windows_user_env("TEMP")  # present in HKCU\Environment on stock Windows
    if value is None:
        pytest.skip("TEMP is not defined in HKCU\\Environment on this machine")

    assert value
    assert "%" not in value
