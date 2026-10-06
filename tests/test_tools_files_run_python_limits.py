"""Tests for ``run_python_file``: the time and output limits, and which files may be run.

The scripts are tiny invented programs written into the temporary attachments folder and executed
by a real subprocess, so the tests check what the isolation really does.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from gaia_agent.config import Settings
from gaia_agent.tools import _pyrun
from gaia_agent.tools.files import RunPythonFileTool


@pytest.fixture
def attachments(settings: Settings) -> Path:
    return settings.files_dir


@pytest.fixture
def tool(settings: Settings) -> RunPythonFileTool:
    return RunPythonFileTool(settings, timeout=20)


@pytest.fixture
def run_python(tool: RunPythonFileTool, attachments: Path) -> Callable[[str], str]:
    """Run a script as the attachment of the question being answered, the only kind of file the tool runs."""

    def run(name: str) -> str:
        tool.set_attachment(attachments / name)
        return tool(name)

    return run


def run_attached(settings: Settings, attachments: Path, name: str, **options: float) -> str:
    """Run ``name`` with a new tool built with ``options``, after registering it as the attachment."""
    tool = RunPythonFileTool(settings, **options)
    tool.set_attachment(attachments / name)
    return tool(name)


def script(attachments: Path, body: str, name: str = "task.py") -> str:
    """Write a script into the attachments folder and return its bare file name."""
    (attachments / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return name


# --------------------------------------------------------------------------- limits


def test_run_python_file_kills_a_script_that_runs_too_long(settings: Settings, attachments: Path) -> None:
    name = script(attachments, "import time\nprint('started', flush=True)\ntime.sleep(60)")

    started = time.monotonic()
    result = run_attached(settings, attachments, name, timeout=1)

    assert time.monotonic() - started < 15
    assert "timed out" in result.lower()
    assert "started" in result


def process_is_alive(pid: int) -> bool:
    if sys.platform == "win32":
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in listing.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform != "win32", reason="the whole process tree is only stopped on Windows")
def test_run_python_file_kills_the_processes_the_script_started(settings: Settings, attachments: Path) -> None:
    name = script(
        attachments,
        """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        print('spawned', child.pid, flush=True)
        time.sleep(60)
        """,
    )

    started = time.monotonic()
    result = run_attached(settings, attachments, name, timeout=1.5)

    assert time.monotonic() - started < 15
    assert "timed out" in result.lower()
    grandchild = int(re.search(r"spawned (\d+)", result).group(1))
    deadline = time.monotonic() + 5
    while process_is_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not process_is_alive(grandchild)


def test_run_python_file_still_stops_the_script_when_taskkill_is_unavailable(
    settings: Settings, attachments: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = script(attachments, "import time\nprint('started', flush=True)\ntime.sleep(60)")

    def no_taskkill(command: list[str], **kwargs: object) -> None:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(_pyrun.subprocess, "run", no_taskkill)

    started = time.monotonic()
    result = run_attached(settings, attachments, name, timeout=0.5)

    assert time.monotonic() - started < 15
    assert "timed out" in result.lower()


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill is a Windows program")
def test_the_process_tree_is_stopped_with_the_taskkill_of_the_system_folder_not_one_found_on_the_path(
    settings: Settings, attachments: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = script(attachments, "import time\ntime.sleep(60)")
    commands: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(_pyrun.subprocess, "run", recording_run)

    assert "timed out" in run_attached(settings, attachments, name, timeout=0.5).lower()

    program = Path(commands[0][0])
    assert program.is_absolute()
    assert program.name.lower() == "taskkill.exe"
    assert program.parent.name.lower() == "system32"


def test_a_script_that_survives_being_killed_is_reported_instead_of_crashing_the_tool(
    settings: Settings, attachments: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    name = script(attachments, "import time\ntime.sleep(60)")
    monkeypatch.setattr(_pyrun, "STOP_WAIT_S", 0.0)
    real_wait = subprocess.Popen.wait
    calls = {"count": 0}

    def stubborn_wait(self: subprocess.Popen[bytes], timeout: float | None = None) -> int:
        calls["count"] += 1
        if timeout == 0.0:  # the wait that follows kill()
            raise subprocess.TimeoutExpired(cmd="python", timeout=0.0)
        return real_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", stubborn_wait)

    with caplog.at_level("WARNING", logger="gaia_agent.tools._pyrun"):
        result = run_attached(settings, attachments, name, timeout=0.5)

    assert "timed out" in result.lower()
    assert "still running" in caplog.text


def test_run_python_file_never_leaves_the_script_running_when_waiting_is_interrupted(
    attachments: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = script(attachments, "import time\ntime.sleep(60)")
    started: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    def interrupted(*args: object, **kwargs: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(_pyrun.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(_pyrun, "_wait_for", interrupted)

    with pytest.raises(KeyboardInterrupt):
        _pyrun.run_python_script(attachments / name)

    assert started[0].poll() is not None


def test_run_python_file_reports_an_interpreter_that_cannot_start(
    run_python: Callable[[str], str], attachments: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = script(attachments, "print('never runs')")

    def refuse(*args: object, **kwargs: object) -> None:
        raise PermissionError("blocked by the test")

    monkeypatch.setattr(_pyrun.subprocess, "Popen", refuse)

    result = run_python(name)

    assert result.startswith("Error:")
    assert "could not start" in result
    assert "PermissionError" in result


# --------------------------------------------------------------------------- what may be run


def test_run_python_file_only_runs_python_sources(run_python: Callable[[str], str], attachments: Path) -> None:
    (attachments / "data.txt").write_text("print('no')", encoding="utf-8")

    result = run_python("data.txt")

    assert result.startswith("Error:")
    assert ".py" in result


def test_run_python_file_refuses_a_script_outside_data(run_python: Callable[[str], str], tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("print('should never run')", encoding="utf-8")

    result = run_python(str(outside))

    assert result.startswith("Error:")
    assert "outside" in result
    assert "should never run" not in result


def test_run_python_file_reports_a_missing_script(run_python: Callable[[str], str]) -> None:
    assert run_python("absent.py").startswith("Error: file not found")


def test_run_python_file_runs_nothing_before_an_attachment_is_registered(
    tool: RunPythonFileTool, attachments: Path
) -> None:
    name = script(attachments, "print('should never run')")

    result = tool(name)

    assert result.startswith("Error:")
    assert "attachment" in result
    assert "should never run" not in result


def test_run_python_file_runs_only_the_attachment_of_the_question(tool: RunPythonFileTool, attachments: Path) -> None:
    attached = script(attachments, "print('the attachment')", name="attached.py")
    planted = script(attachments, "print('a script nobody attached')", name="planted.py")
    tool.set_attachment(attachments / attached)

    assert "the attachment" in tool(attached)
    refused = tool(planted)

    assert refused.startswith("Error:")
    assert "only the .py attachment" in refused
    assert "nobody attached" not in refused


def test_run_python_file_forgets_the_attachment_when_it_is_withdrawn(
    tool: RunPythonFileTool, attachments: Path
) -> None:
    name = script(attachments, "print('first question only')")
    tool.set_attachment(attachments / name)
    assert "first question only" in tool(name)

    tool.set_attachment(None)

    assert tool(name).startswith("Error:")


def test_run_python_file_does_not_accept_a_look_alike_of_the_attachment(
    tool: RunPythonFileTool, attachments: Path
) -> None:
    attached = script(attachments, "print('real')", name="task.py")
    look_alike = script(attachments, "print('fake')", name="task.py.py")
    tool.set_attachment(attachments / attached)

    assert tool(look_alike).startswith("Error:")


def test_run_python_file_still_refuses_a_registered_script_outside_the_attachments(
    tool: RunPythonFileTool, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("print('should never run')", encoding="utf-8")
    tool.set_attachment(outside)

    result = tool(str(outside))

    assert result.startswith("Error:")
    assert "outside" in result
    assert "should never run" not in result


def test_run_python_file_uses_the_environment_data_folder_by_default(attachments: Path) -> None:
    name = script(attachments, "print('default settings work')")

    default_tool = RunPythonFileTool()
    default_tool.set_attachment(attachments / name)

    assert "default settings work" in default_tool(name)


def test_run_python_file_does_not_leak_the_parent_environment_marker(
    run_python: Callable[[str], str], attachments: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GAIA_DATA_DIR_MARKER", "visible-only-to-the-parent")
    name = script(attachments, "import os\nprint(os.environ.get('GAIA_DATA_DIR_MARKER', 'absent'))")

    assert "absent" in run_python(name)
    assert os.environ["GAIA_DATA_DIR_MARKER"] == "visible-only-to-the-parent"
