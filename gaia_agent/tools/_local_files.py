"""Shared helpers for the tools that touch local files.

The :class:`AttachmentSandbox` is what keeps ``read_file``, ``run_python_file``,
``transcribe_audio`` and ``describe_image`` away from the rest of the machine:
a tool may only open a file that lives inside the attachments folder
(``<data>/cache/files``). Everything else in DATA stays out of reach: the
question list, the benchmark metadata (which holds the answer key), earlier
runs and the scratch folders. The module also holds the small output helpers
every tool shares (truncation with a visible marker and secret redaction) and
:func:`scratch_folder`, the throw-away folder tools work in.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from gaia_agent.config import ConfigError, Settings, load_settings

logger = logging.getLogger(__name__)

ECHO_LIMIT = 200  # how much of a rejected path an error message repeats


class ToolError(Exception):
    """An expected tool failure. The message is written for the agent and never contains a secret."""


def truncate_text(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters, appending ``[truncated: N chars total]`` when it was longer."""
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n[truncated: {len(text)} chars total]"


def truncate_middle(text: str, limit: int) -> str:
    """Keep the start and the end of ``text`` (a third and two thirds of ``limit``) and mark the cut.

    Used for program output, where the end (the result or the traceback) matters most.
    """
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n[truncated: {len(text)} chars total]\n{text[len(text) - tail :]}"


def redact(text: str, *secrets: str | None) -> str:
    """Replace every secret value found in ``text`` with ``[redacted]`` (blank secrets are ignored)."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


@contextmanager
def scratch_folder(parent: Path | None, prefix: str) -> Iterator[Path]:
    """A throw-away folder inside ``parent`` (the system temp folder when None), removed when the block ends.

    Tools work on copies of benchmark files here, so ``parent`` is normally ``<data>/tmp``: nothing from the
    benchmark is left in %TEMP%. A folder that cannot be removed (Windows keeps files open for a moment) is
    reported in a warning instead of being left behind silently.
    """
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        if folder.exists():
            logger.warning("Could not remove the scratch folder %s; delete it by hand.", folder)


def _clean_path_text(file_path: object) -> str:
    """Normalise the path the agent typed: a non-empty string (or path object) without wrapping quotes."""
    if isinstance(file_path, os.PathLike):
        file_path = os.fspath(file_path)
    if not isinstance(file_path, str) or not file_path.strip():
        raise ToolError("file_path must be a non-empty string")
    text = file_path.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        raise ToolError("file_path must be a non-empty string")
    if "\x00" in text:
        raise ToolError("invalid file path (it contains a null character)")
    return text


def _is_file(path: Path) -> bool:
    """True for an existing regular file; a path the operating system will not even stat counts as missing."""
    try:
        return path.is_file()
    except OSError:
        return False


@dataclass(frozen=True)
class AttachmentSandbox:
    """The only place the file tools may read from: the attachments folder, nothing else.

    Attributes:
        root: The attachments folder (``<data>/cache/files``); nothing outside it is ever opened. The
            rest of DATA (question list, benchmark metadata, runs, scratch folders) is outside it.
        search_dirs: Where a relative path (a bare file name, for instance) is looked up, in order:
            the attachments folder, then the folder the Hugging Face dataset fallback downloads into.
        scratch_dir: Where a tool may make throw-away working folders (see :func:`scratch_folder`).
    """

    root: Path
    search_dirs: tuple[Path, ...]
    scratch_dir: Path | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> AttachmentSandbox:
        """Build the sandbox of ``settings`` (or of the environment when ``settings`` is None)."""
        try:
            active = settings if settings is not None else load_settings()
        except ConfigError as exc:
            raise ToolError(f"the data folder is not usable: {exc}") from exc
        files = active.files_dir.resolve()
        return cls(
            root=files,
            search_dirs=(files, (files / active.hf_split_dir).resolve()),
            scratch_dir=active.tmp_dir.resolve(),
        )

    def resolve_file(self, file_path: object) -> Path:
        """Return the real path of an existing file the tools may open, or raise :class:`ToolError`.

        ``file_path`` may be absolute or relative (a relative path is tried in each of
        ``search_dirs``). The path is fully resolved first, so ``..`` segments and
        symlinks cannot lead outside the attachments folder.
        """
        text = _clean_path_text(file_path)
        for candidate in self._candidates(Path(text)):
            resolved = self._resolve(candidate)
            self._check_allowed(resolved)
            if _is_file(resolved):
                return resolved
        raise ToolError(f"file not found: {text[:ECHO_LIMIT]}")

    def _candidates(self, path: Path) -> list[Path]:
        if path.is_absolute():
            return [path]
        return [folder / path for folder in self.search_dirs]

    @staticmethod
    def _resolve(candidate: Path) -> Path:
        try:
            return candidate.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ToolError(f"invalid file path ({type(exc).__name__})") from exc

    def _check_allowed(self, resolved: Path) -> None:
        if not resolved.is_relative_to(self.root):
            raise ToolError("that path is outside the attachments folder; only question attachments can be used")
