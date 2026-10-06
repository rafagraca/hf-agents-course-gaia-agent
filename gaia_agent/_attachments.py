"""Safe local storage for question attachments.

File names come from the scoring API, so they are untrusted input: they are cut down
to a plain file name, checked against path traversal and Windows quirks, and written
atomically directly inside one folder.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path, PureWindowsPath

_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_METADATA_NAME = re.compile(r"metadata(?:\..*)?", re.IGNORECASE)
_DATA_TABLE_SUFFIXES = (".parquet", ".jsonl")  # a table of the benchmark, whatever it is called
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}
)


class UnsafeFileNameError(ValueError):
    """An attachment name received from the API cannot be used as a local file name."""


def _shorten(text: str, limit: int = 60) -> str:
    """``repr`` of ``text`` cut to ``limit`` characters, for error messages."""
    shown = repr(text)
    return shown if len(shown) <= limit else shown[:limit] + "..."


def sanitize_file_name(raw: str) -> str:
    """Return the safe local name for an attachment name received from the API.

    Keeps only the last path component (what ``Path(name).name`` gives on Windows, with
    ``/`` and ``\\`` both treated as separators on every OS) and refuses names that are
    empty, contain ``..``, hold characters Windows forbids in file names (which also blocks
    NTFS alternate data streams) or are reserved device names such as ``NUL``.
    """
    if ".." in raw:
        raise UnsafeFileNameError(f"{_shorten(raw)} contains '..'")
    name = PureWindowsPath(raw).name
    if not name:
        raise UnsafeFileNameError(f"{_shorten(raw)} has no file name")
    if _ILLEGAL_NAME_CHARS.search(name):
        raise UnsafeFileNameError(f"{_shorten(raw)} contains characters not allowed in file names")
    if name.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_DEVICE_NAMES:
        raise UnsafeFileNameError(f"{_shorten(raw)} is a reserved device name")
    if name.endswith((".", " ")):  # Windows drops them: the file would land under another name than the one checked
        raise UnsafeFileNameError(f"{_shorten(raw)} ends with a dot or a space")
    if _METADATA_NAME.fullmatch(name) or name.lower().endswith(_DATA_TABLE_SUFFIXES):
        # the answer key of the benchmark is named like this, never an attachment
        raise UnsafeFileNameError(f"{_shorten(raw)} is named like the benchmark metadata")
    return name


def destination(root: Path, name: str) -> Path:
    """Where the attachment ``name`` is stored: directly inside ``root``, whatever ``name`` holds.

    A second line of defence behind :func:`sanitize_file_name`: the resolved target (which
    follows symbolic links) must have ``root`` as its parent.
    """
    target = (root / name).resolve()
    if target.parent != root:
        raise UnsafeFileNameError(f"{_shorten(name)} does not resolve inside the files folder")
    return target


def cached_copy(root: Path, split_dir: str, name: str) -> Path | None:
    """A non-empty copy of ``name`` downloaded earlier, from the API (``root``) or the dataset (``split_dir``)."""
    for candidate in (root / name, root / split_dir / name):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so that a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".download-", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
