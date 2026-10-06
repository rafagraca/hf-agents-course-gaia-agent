"""Tests for ``gaia_agent.tools.default_tools``: which tools the agent gets and which DATA folder they use."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from smolagents import Tool

from gaia_agent.config import Settings
from gaia_agent.tools import default_tools

CORE_TOOLS = [
    "web_search",
    "read_webpage",
    "wikipedia_search",
    "wikipedia_page",
    "read_file",
    "youtube_transcript",
    "transcribe_audio",
]
RUN_PYTHON_TOOL = "run_python_file"
IMAGE_TOOL = "describe_image"  # Groq vision
VIDEO_TOOL = "ask_about_video"  # Gemini


def media_names(settings: Settings) -> list[str]:
    """The tool names without ``deep_search``, which has its own rules (it needs a Groq key too)."""
    return [name for name in names(default_tools(settings)) if name != "deep_search"]


def names(tools: list[Tool]) -> list[str]:
    return [tool.name for tool in tools]


def test_the_core_tools_are_listed_once_each_in_a_stable_order(settings: Settings) -> None:
    assert names(default_tools(settings)) == CORE_TOOLS


def test_running_a_python_attachment_is_not_offered_by_default(settings: Settings) -> None:
    assert RUN_PYTHON_TOOL not in names(default_tools(settings))


def test_running_a_python_attachment_is_offered_when_the_environment_switches_it_on(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GAIA_ALLOW_RUN_PYTHON", "1")

    assert names(default_tools(settings)) == [*CORE_TOOLS[:5], RUN_PYTHON_TOOL, *CORE_TOOLS[5:]]


def test_other_values_of_the_variable_keep_it_off(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_ALLOW_RUN_PYTHON", "0")

    assert RUN_PYTHON_TOOL not in names(default_tools(settings))


def test_describe_image_is_added_only_when_a_groq_key_exists(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-for-tests")

    assert media_names(settings) == [*CORE_TOOLS, IMAGE_TOOL]


def test_ask_about_video_is_added_only_when_a_gemini_key_exists(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-for-tests")

    assert media_names(settings) == [*CORE_TOOLS, VIDEO_TOOL]


def test_both_optional_tools_are_added_when_both_keys_exist(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-for-tests")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-for-tests")

    assert media_names(settings) == [*CORE_TOOLS, IMAGE_TOOL, VIDEO_TOOL]


def test_every_tool_is_a_smolagents_tool_returning_text(settings: Settings) -> None:
    tools = default_tools(settings)

    assert all(isinstance(tool, Tool) and tool.output_type == "string" for tool in tools)


def test_each_call_builds_new_tool_instances(settings: Settings) -> None:
    first, second = default_tools(settings), default_tools(settings)

    assert all(a is not b for a, b in zip(first, second, strict=True))


def test_the_file_tools_read_the_attachments_folder_of_the_given_settings(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (settings.files_dir / "notes.txt").write_text("an invented attachment", encoding="utf-8")
    monkeypatch.setenv("GAIA_DATA_DIR", str(tmp_path / "another-data-folder"))  # must not be used

    read_file = {tool.name: tool for tool in default_tools(settings)}["read_file"]

    assert "an invented attachment" in read_file(file_path="notes.txt")


def test_the_file_tools_follow_a_replaced_data_folder(settings: Settings, tmp_path: Path) -> None:
    other = dataclasses.replace(settings, data_dir=tmp_path / "second-data")
    other.files_dir.mkdir(parents=True)
    (other.files_dir / "notes.txt").write_text("from the second folder", encoding="utf-8")

    read_file = {tool.name: tool for tool in default_tools(other)}["read_file"]

    assert "from the second folder" in read_file(file_path="notes.txt")
