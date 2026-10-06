"""Run a Python attachment in a throw-away subprocess, with good manners but no real isolation.

This exists for the ``.py`` attachments of the exam. It runs a *benign* program politely: isolated
interpreter (``-I``), empty stdin, a fresh working folder that doubles as HOME and TEMP, a minimal
environment without any credential, a wall-clock timeout and a cap on captured output.

It is NOT a security sandbox. The script runs with the privileges of the user: it can read every file
the user can, open network connections and read the persistent user environment of Windows
(``HKCU\\Environment``), where API tokens may be stored, because cleaning the *process* environment does
not touch the registry. Only run code you have read; that is why the tool is off unless
``GAIA_ALLOW_RUN_PYTHON=1`` and runs only the attachment of the question being answered.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from gaia_agent.tools._local_files import ToolError, scratch_folder, truncate_middle

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 20.0
MAX_CAPTURE_BYTES = 1_000_000
STREAM_LIMIT = 3000
POLL_INTERVAL_S = 0.05
STOP_WAIT_S = 5.0

FINISHED = "finished"
TIMED_OUT = "timed out"
FLOODED = "flooded"

# The only variables the child gets from its parent: what Python and the C runtime need on Windows and POSIX.
ENV_PASSTHROUGH = ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "LANG", "LC_ALL", "LC_CTYPE")
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH", re.IGNORECASE)


@dataclass(frozen=True)
class RunResult:
    """What happened to a script: how it ended, its exit code and its (truncated) output."""

    outcome: str
    exit_code: int | None
    stdout: str
    stderr: str


def build_child_env(
    parent: Mapping[str, str], home: Path, passthrough: Sequence[str] = ENV_PASSTHROUGH
) -> dict[str, str]:
    """Return the environment of the child: allow-listed variables only, never a secret-looking one.

    HOME, USERPROFILE, TEMP, TMP and TMPDIR all point at ``home``, the throw-away working folder.
    """
    allowed = {name.upper() for name in passthrough}
    env = {name: value for name, value in parent.items() if name.upper() in allowed and not _SECRET_NAME.search(name)}
    for name in ("HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR"):
        env[name] = str(home)
    return env


def _taskkill_path() -> str | None:
    """The full path of ``taskkill.exe`` (looked up in %SystemRoot%, never through PATH), None when unknown."""
    system_root = os.environ.get("SYSTEMROOT")  # os.environ upper-cases the names on Windows
    return os.path.join(system_root, "System32", "taskkill.exe") if system_root else None


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Kill the script and, on Windows, everything it started (elsewhere only the script itself is stopped)."""
    taskkill = _taskkill_path() if sys.platform == "win32" else None
    if taskkill is not None:
        # Best effort: process.kill() below still stops the direct child when taskkill is missing or fails.
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [taskkill, "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False, timeout=10
            )
    process.kill()
    try:
        process.wait(timeout=STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        logger.warning("The script (pid %d) was still running %.0f s after it was killed.", process.pid, STOP_WAIT_S)


def _wait_for(process: subprocess.Popen[bytes], timeout: float, streams: Sequence[Path]) -> str:
    """Wait until the script ends, runs out of time or floods its output; stop it in the last two cases."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            process.wait(timeout=POLL_INTERVAL_S)
            return FINISHED
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() >= deadline:
            outcome = TIMED_OUT
        elif any(stream.stat().st_size > MAX_CAPTURE_BYTES for stream in streams):
            outcome = FLOODED
        else:
            continue
        _stop(process)
        return outcome


def _read_stream(path: Path) -> str:
    with path.open("rb") as handle:
        data = handle.read(MAX_CAPTURE_BYTES)
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").rstrip()
    return truncate_middle(text, STREAM_LIMIT)


def _start(script: Path, workdir: Path, stdout: BinaryIO, stderr: BinaryIO) -> subprocess.Popen[bytes]:
    command = [sys.executable, "-I", "-X", "utf8", str(script)]
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            cwd=workdir,
            env=build_child_env(os.environ, workdir),
        )
    except OSError as exc:
        raise ToolError(f"could not start the script ({type(exc).__name__})") from exc


def run_python_script(
    script: Path, *, timeout: float = DEFAULT_TIMEOUT_S, scratch_dir: Path | None = None
) -> RunResult:
    """Run ``script`` with the current interpreter in isolated mode and collect its output.

    The working folder and the captured output live in a throw-away folder inside ``scratch_dir`` (the
    system temp folder when None).
    """
    with scratch_folder(scratch_dir, "gaia-run-") as root:
        workdir = root / "work"
        workdir.mkdir()
        stdout_path, stderr_path = root / "stdout.txt", root / "stderr.txt"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = _start(script, workdir, stdout, stderr)
            try:
                outcome = _wait_for(process, timeout, (stdout_path, stderr_path))
            finally:
                if process.poll() is None:  # never leave the script running if waiting was interrupted
                    _stop(process)
        exit_code = process.returncode if outcome == FINISHED else None
        return RunResult(outcome, exit_code, _read_stream(stdout_path), _read_stream(stderr_path))


def format_run_result(result: RunResult, timeout: float) -> str:
    """Render a :class:`RunResult` as the compact text the agent reads."""
    if result.outcome == TIMED_OUT:
        headline = f"Timed out: the script did not finish within {timeout:g} s and was stopped."
    elif result.outcome == FLOODED:
        headline = f"Output limit exceeded: the script wrote more than {MAX_CAPTURE_BYTES} bytes and was stopped."
    else:
        headline = f"exit code: {result.exit_code}"
    return "\n".join([headline, "stdout:", result.stdout or "(empty)", "stderr:", result.stderr or "(empty)"])
