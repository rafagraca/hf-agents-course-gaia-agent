"""Tests for ``run_python_file``: running the Python attachment of a question in a throw-away process (no network).

The tool is not a security sandbox (the script runs with the privileges of the user), so it runs only the file
the question came with and is off unless asked for. The scripts here are tiny invented programs written into the
temporary attachments folder and executed by a real subprocess, so the tests check what the isolation really does.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from gaia_agent.config import Settings
from gaia_agent.tools._pyrun import build_child_env
from gaia_agent.tools.files import RunPythonFileTool

SECRET_ENV = {
    "GROQ_API_KEY": "fake-groq-secret-value",
    "HF_TOKEN": "fake-hf-secret-value",
    "GEMINI_API_KEY": "fake-gemini-secret-value",
    "OPENAI_API_KEY": "fake-openai-secret-value",
    "MY_SERVICE_SECRET": "fake-service-secret-value",
    "DB_PASSWORD": "fake-password-value",
}


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


# --------------------------------------------------------------------------- results


def test_run_python_file_reports_exit_code_and_output(run_python: Callable[[str], str], attachments: Path) -> None:
    name = script(attachments, "print('hello from the attachment')")

    result = run_python(name)

    assert "exit code: 0" in result
    assert "hello from the attachment" in result


def test_run_python_file_reports_a_failing_exit_code(run_python: Callable[[str], str], attachments: Path) -> None:
    name = script(attachments, "import sys\nprint('partial')\nsys.exit(3)")

    result = run_python(name)

    assert "exit code: 3" in result
    assert "partial" in result


def test_run_python_file_shows_the_traceback_of_a_crash(run_python: Callable[[str], str], attachments: Path) -> None:
    name = script(attachments, "raise ValueError('boom')")

    result = run_python(name)

    assert "exit code: 1" in result
    assert "ValueError: boom" in result


def test_run_python_file_decodes_unicode_output(run_python: Callable[[str], str], attachments: Path) -> None:
    name = script(attachments, "print('h\\u00e9llo \\u2713 \\u65e5\\u672c')")

    assert "héllo ✓ 日本" in run_python(name)


def test_run_python_file_marks_empty_streams(run_python: Callable[[str], str], attachments: Path) -> None:
    name = script(attachments, "pass")

    result = run_python(name)

    assert result.count("(empty)") == 2


def test_run_python_file_keeps_the_start_and_the_end_of_long_output(
    run_python: Callable[[str], str], attachments: Path
) -> None:
    name = script(attachments, "print('START' + 'x' * 20000 + 'END')")

    result = run_python(name)

    assert "START" in result
    assert "END" in result
    assert f"[truncated: {len('START') + 20_000 + len('END')} chars total]" in result
    assert len(result) < 4000


def test_run_python_file_stops_a_script_that_floods_the_output(settings: Settings, attachments: Path) -> None:
    name = script(attachments, "while True:\n    print('x' * 1000)")

    result = run_attached(settings, attachments, name, timeout=30)

    assert "Output limit exceeded" in result
    assert len(result) < 8000


# --------------------------------------------------------------------------- isolation


def test_run_python_file_runs_in_isolated_mode_with_an_empty_stdin(
    run_python: Callable[[str], str], attachments: Path
) -> None:
    name = script(
        attachments,
        """
        import sys
        print('isolated', sys.flags.isolated)
        try:
            input()
        except EOFError:
            print('stdin is empty')
        """,
    )

    result = run_python(name)

    assert "isolated 1" in result
    assert "stdin is empty" in result


def test_run_python_file_can_import_the_packages_of_the_project(
    run_python: Callable[[str], str], attachments: Path
) -> None:
    name = script(attachments, "import json, numpy\nprint(json.dumps({'sum': int(numpy.arange(5).sum())}))")

    result = run_python(name)

    assert "exit code: 0" in result
    assert '{"sum": 10}' in result


def test_run_python_file_does_not_see_sibling_modules_of_the_script(
    run_python: Callable[[str], str], attachments: Path
) -> None:
    script(attachments, "VALUE = 1", name="helper_module.py")
    name = script(attachments, "import helper_module")

    result = run_python(name)

    assert "exit code: 1" in result
    assert "ModuleNotFoundError" in result


def test_run_python_file_uses_a_fresh_working_folder_each_time(
    run_python: Callable[[str], str], attachments: Path
) -> None:
    name = script(
        attachments,
        """
        import os
        print('cwd', os.getcwd())
        print('listing', os.listdir('.'))
        open('scratch.txt', 'w').write('left behind')
        """,
    )

    first = run_python(name)
    second = run_python(name)

    cwd_first = next(line for line in first.splitlines() if line.startswith("cwd ")).removeprefix("cwd ")
    cwd_second = next(line for line in second.splitlines() if line.startswith("cwd ")).removeprefix("cwd ")
    assert cwd_first != cwd_second
    assert not Path(cwd_first).exists()
    assert not Path(cwd_first).is_relative_to(attachments)
    assert "listing []" in first
    assert "listing []" in second
    assert not (attachments / "scratch.txt").exists()


def test_run_python_file_works_inside_the_data_folder_not_in_the_system_temp_folder(
    run_python: Callable[[str], str], attachments: Path, settings: Settings
) -> None:
    name = script(attachments, "import os\nprint('cwd', os.getcwd())")

    result = run_python(name)

    cwd = Path(next(line for line in result.splitlines() if line.startswith("cwd ")).removeprefix("cwd "))
    assert cwd.is_relative_to(settings.tmp_dir.resolve())
    assert list(settings.tmp_dir.iterdir()) == []  # and nothing is left behind


def test_run_python_file_hides_every_secret_from_the_script(
    run_python: Callable[[str], str], attachments: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in SECRET_ENV.items():
        monkeypatch.setenv(name, value)
    name = script(attachments, "import os\nprint(sorted(os.environ))\nprint(dict(os.environ))")

    result = run_python(name)

    assert "exit code: 0" in result
    for variable, value in SECRET_ENV.items():
        assert variable not in result
        assert value not in result
    assert "PATH" in result


def test_child_environment_is_a_minimal_allow_list(tmp_path: Path) -> None:
    parent = {"Path": "p", "SYSTEMROOT": "s", "HF_TOKEN": "t", "SOME_API_KEY": "k", "UNRELATED": "u", "LANG": "C"}

    env = build_child_env(parent, tmp_path)

    assert env["Path"] == "p"
    assert env["SYSTEMROOT"] == "s"
    assert env["LANG"] == "C"
    assert "HF_TOKEN" not in env
    assert "SOME_API_KEY" not in env
    assert "UNRELATED" not in env
    assert {env["HOME"], env["USERPROFILE"], env["TEMP"], env["TMP"]} == {str(tmp_path)}


def test_child_environment_drops_secret_looking_names_even_if_allow_listed(tmp_path: Path) -> None:
    parent = {"PATH": "p", "MY_API_KEY": "k", "SESSION_TOKEN": "t", "CLIENT_SECRET": "s", "DB_PASSWORD": "d"}

    env = build_child_env(parent, tmp_path, passthrough=tuple(parent))

    assert env["PATH"] == "p"
    assert not any(word in name.upper() for name in env for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
