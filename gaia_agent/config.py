"""Runtime configuration: settings, the DATA folder layout and secret lookup.

Everything the agent needs from the outside world is resolved here: the frozen
``Settings`` dataclass, the on-disk DATA folder (always kept outside the
repository, because it holds benchmark material that must never be committed)
and :func:`get_secret`, the only way the project reads API keys and tokens.
Secret values are never printed, logged or written to disk.
"""

from __future__ import annotations

import logging
import ntpath
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_MODEL_ID = "groq/qwen/qwen3.8-27b"
DEFAULT_FALLBACK_MODEL_IDS = ("groq/openai/gpt-oss-120b",)

ENV_DATA_DIR = "GAIA_DATA_DIR"
ENV_MODEL_ID = "GAIA_MODEL_ID"
ENV_FALLBACK_MODELS = "GAIA_FALLBACK_MODELS"
ENV_ALLOW_RUN_PYTHON = "GAIA_ALLOW_RUN_PYTHON"

ANSWER_KEY_PREFIX = "metadata"  # the files of the benchmark metadata are named metadata.jsonl, metadata.parquet...
ANSWER_KEY_SUFFIXES = (".parquet",)  # a table of the benchmark, whatever it is called
MAX_REPORTED_KEY_FILES = 5
NO_FALLBACKS = frozenset({"none", "-"})  # a GAIA_FALLBACK_MODELS value that switches the fallback models off
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})

APP_DIR_NAME = "hf-gaia-agent"
REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """The environment describes a configuration the agent cannot use."""


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Immutable agent settings.

    The dataclass is keyword-only: build it with ``Settings(data_dir=...)`` or
    with :func:`load_settings`, and derive variants with ``dataclasses.replace``.
    """

    api_url: str = DEFAULT_API_URL
    data_dir: Path
    model_id: str = DEFAULT_MODEL_ID
    fallback_model_ids: tuple[str, ...] = DEFAULT_FALLBACK_MODEL_IDS
    tpm_limit: int = 8000
    max_prompt_tokens: int = 5000
    max_output_tokens: int = 2000
    max_steps: int = 10
    reasoning_effort: str = "low"  # sent to gpt-oss models only (see BudgetedModel)
    hf_dataset: str = "gaia-benchmark/GAIA"
    hf_split_dir: str = "2023/validation"

    def __post_init__(self) -> None:
        # The dataclass is frozen, so normalise through object.__setattr__ to let
        # callers pass a plain ``str`` path or a ``list`` of model ids.
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "fallback_model_ids", tuple(self.fallback_model_ids))

    @property
    def cache_dir(self) -> Path:
        """Question list and downloaded attachments."""
        return self.data_dir / "cache"

    @property
    def files_dir(self) -> Path:
        """Downloaded question attachments."""
        return self.cache_dir / "files"

    @property
    def runs_dir(self) -> Path:
        """One ``<YYYYMMDD-HHMMSS>`` sub-folder per run (answers and traces)."""
        return self.data_dir / "runs"

    @property
    def gaia_dir(self) -> Path:
        """A folder nothing writes to any more (the answer key was once kept here); the guard looks everywhere."""
        return self.data_dir / "gaia"

    @property
    def tmp_dir(self) -> Path:
        """Throw-away folders (converted audio, script runs): benchmark material stays inside DATA, never in %TEMP%."""
        return self.data_dir / "tmp"


def _strip_or_none(value: str | None) -> str | None:
    """Strip ``value``; blank text counts as missing."""
    if value is None:
        return None
    return value.strip() or None


def _is_answer_key_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(ANSWER_KEY_PREFIX) or lowered.endswith(ANSWER_KEY_SUFFIXES)


def answer_key_files(settings: Settings) -> list[Path]:
    """The files anywhere in the data folder that look like the benchmark metadata, its answer key.

    Named ``metadata*`` or ending in ``.parquet``. The files of a run (``answers.jsonl``, ``usage.jsonl``)
    and the question list are not.
    """
    if not settings.data_dir.is_dir():
        return []
    return sorted(path for path in settings.data_dir.rglob("*") if path.is_file() and _is_answer_key_name(path.name))


def ensure_no_answer_key(settings: Settings) -> None:
    """Raise :class:`ConfigError` while a file that looks like the benchmark answer key is in the data folder.

    The agent can read the attachments folder and a run could be started with the key in reach of its tools,
    so a run refuses to start. Nothing in this repository downloads that file: if it is there, someone put it
    there, and the way out is to delete it.
    """
    found = answer_key_files(settings)
    if found:
        shown = ", ".join(str(path) for path in found[:MAX_REPORTED_KEY_FILES])
        more = f" (and {len(found) - MAX_REPORTED_KEY_FILES} more)" if len(found) > MAX_REPORTED_KEY_FILES else ""
        raise ConfigError(
            f"files that look like the benchmark metadata, which holds the answer key, are in the data folder: "
            f"{shown}{more}. Delete them before running the agent."
        )


def _env_text(name: str) -> str | None:
    """Return a stripped environment variable, or None when unset or blank."""
    return _strip_or_none(os.environ.get(name))


def _env_flag(name: str) -> bool:
    """True when the environment variable ``name`` says yes (``1``, ``true``, ``yes`` or ``on``, in any case)."""
    value = _env_text(name)
    return value is not None and value.lower() in TRUE_WORDS


def run_python_enabled() -> bool:
    """Whether ``GAIA_ALLOW_RUN_PYTHON`` switches on the ``run_python_file`` tool (off unless it says yes).

    The tool starts a program with the privileges of the user, so someone has to read the attachment
    and decide before it is offered to the agent.
    """
    return _env_flag(ENV_ALLOW_RUN_PYTHON)


def _parse_model_list(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated list of model ids, dropping blanks."""
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _fallback_models(raw: str | None) -> tuple[str, ...]:
    """The fallback chain of ``GAIA_FALLBACK_MODELS``: unset or blank means the defaults, ``none`` means no models."""
    if raw is not None and raw.strip().lower() in NO_FALLBACKS:
        return ()
    return _parse_model_list(raw) or DEFAULT_FALLBACK_MODEL_IDS


def _default_data_dir() -> Path:
    """``%LOCALAPPDATA%\\hf-gaia-agent``, or ``~/.local/share/hf-gaia-agent`` off Windows."""
    local_app_data = _env_text("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
    return base / APP_DIR_NAME


def _is_too_broad(path: Path) -> bool:
    """True for a folder that is no place for benchmark data: a drive root, the home folder or a parent of the repo."""
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None
    return path.parent == path or path == home or REPO_ROOT.is_relative_to(path)


def _resolve_data_dir() -> Path:
    """Work out the DATA folder; refuse any location inside the repository and any folder that is too broad."""
    configured = _env_text(ENV_DATA_DIR)
    try:
        path = (Path(configured).expanduser() if configured else _default_data_dir()).resolve()
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"invalid {ENV_DATA_DIR} {configured!r}: {exc}") from exc
    if path.is_relative_to(REPO_ROOT):
        raise ConfigError(
            f"{ENV_DATA_DIR} must point outside the repository (benchmark data must never be committed): {path}"
        )
    if _is_too_broad(path):
        raise ConfigError(
            f"{ENV_DATA_DIR} must be a dedicated folder, not a drive root, your home folder or a parent "
            f"of the repository: {path}"
        )
    return path


def _create_data_dirs(settings: Settings) -> None:
    """Create the DATA sub-folders, turning OS failures into a ``ConfigError``."""
    for folder in (settings.cache_dir, settings.files_dir, settings.runs_dir, settings.gaia_dir, settings.tmp_dir):
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            raise ConfigError(f"cannot create data folder {folder}: {reason}") from exc


def load_settings() -> Settings:
    """Build :class:`Settings` from the environment and create the DATA sub-folders.

    Reads ``GAIA_DATA_DIR`` (default ``%LOCALAPPDATA%\\hf-gaia-agent``),
    ``GAIA_MODEL_ID`` and ``GAIA_FALLBACK_MODELS`` (comma separated; ``none`` switches the
    fallbacks off). Blank values count as unset. Raises :class:`ConfigError` when the DATA folder lies
    inside the repository, is too broad (a drive root, the home folder, a parent of the
    repository) or cannot be created.
    """
    settings = Settings(
        data_dir=_resolve_data_dir(),
        model_id=_env_text(ENV_MODEL_ID) or DEFAULT_MODEL_ID,
        fallback_model_ids=_fallback_models(os.environ.get(ENV_FALLBACK_MODELS)),
    )
    _create_data_dirs(settings)
    return settings


def _read_windows_user_env(name: str) -> str | None:
    """Read ``name`` from the persistent user environment (``HKCU\\Environment``).

    Returns None off Windows, when the variable is not defined, and when the
    registry cannot be read (a warning names the variable, never its value).
    """
    try:
        import winreg  # Windows only; absent elsewhere
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, kind = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None  # the variable is simply not defined
    except OSError as exc:
        logger.warning("Could not read %s from the Windows user environment (%s).", name, type(exc).__name__)
        return None
    if not isinstance(value, str):
        return None
    return ntpath.expandvars(value) if kind == winreg.REG_EXPAND_SZ else value


# Seam for tests: how a variable missing from the process environment is looked up.
_persistent_env_reader: Callable[[str], str | None] = _read_windows_user_env


def get_secret(name: str) -> str | None:
    """Return the secret called ``name``, or None when it is not set anywhere.

    The process environment wins. Otherwise, on Windows, the user's persistent
    environment is consulted, so a token saved after this session started is
    still found. The value is stripped and is never printed or logged.
    """
    value = _env_text(name)
    if value is None:
        value = _strip_or_none(_persistent_env_reader(name))
    return value
