"""Contract tests for the public interfaces fixed by the project spec.

They inspect signatures, dataclass fields and tool schemas without calling any
function, so they hold for the initial stubs and for the finished modules alike.
Extra trailing parameters with defaults are allowed; renaming, removing or
re-ordering the specified ones is not.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import time
from types import ModuleType
from typing import Any

import pytest
from smolagents import Model, Tool

# (module, qualified name, the parameters the spec fixes, in order).
# BudgetedModel.__init__ is not in the spec: the skeleton chose it and build_agent relies on it.
SIGNATURES = [
    ("gaia_agent.config", "load_settings", []),
    ("gaia_agent.config", "get_secret", ["name"]),
    ("gaia_agent.api", "ScoringClient.__init__", ["self", "settings", "session"]),
    ("gaia_agent.api", "ScoringClient.get_questions", ["self", "refresh"]),
    ("gaia_agent.api", "ScoringClient.download_file", ["self", "q"]),
    ("gaia_agent.api", "ScoringClient.submit", ["self", "username", "agent_code", "answers", "dry_run"]),
    ("gaia_agent.api", "validate_answers", ["questions", "answers"]),
    ("gaia_agent.answer", "normalize_answer", ["question", "raw"]),
    ("gaia_agent.prompts", "build_prompt_templates", ["default_templates"]),
    ("gaia_agent.budget", "TokenBudget.__init__", ["self", "tpm_limit", "clock", "sleep"]),
    ("gaia_agent.budget", "TokenBudget.estimate_tokens", ["self", "messages"]),
    ("gaia_agent.budget", "TokenBudget.reserve", ["self", "tokens"]),
    ("gaia_agent.budget", "TokenBudget.record", ["self", "tokens"]),
    ("gaia_agent.budget", "BudgetedModel.__init__", ["self", "settings", "usage_log"]),
    ("gaia_agent.memory", "make_trim_callback", ["max_last_obs_chars", "max_old_obs_chars"]),
    ("gaia_agent.tools", "default_tools", ["settings"]),
    ("gaia_agent.tools.media", "optional_media_tools", ["settings"]),
    ("gaia_agent.agent", "build_agent", ["settings", "tools", "model", "usage_log"]),
    ("gaia_agent.agent", "answer_question", ["agent", "q", "file_path"]),
    ("gaia_agent.run", "main", ["argv"]),
    ("gaia_agent.submit", "main", ["argv"]),
]

# Defaults the spec fixes.
DEFAULTS = [
    ("gaia_agent.api", "ScoringClient.__init__", {"session": None}),
    ("gaia_agent.api", "ScoringClient.get_questions", {"refresh": False}),
    ("gaia_agent.api", "ScoringClient.submit", {"dry_run": True}),
    ("gaia_agent.budget", "TokenBudget.__init__", {"clock": time.monotonic, "sleep": time.sleep}),
    ("gaia_agent.budget", "BudgetedModel.__init__", {"usage_log": None}),
    ("gaia_agent.memory", "make_trim_callback", {"max_last_obs_chars": 6000, "max_old_obs_chars": 700}),
    ("gaia_agent.agent", "build_agent", {"tools": None, "model": None, "usage_log": None}),
    ("gaia_agent.run", "main", {"argv": None}),
    ("gaia_agent.submit", "main", {"argv": None}),
]

# Frozen dataclasses and their fields, in order.
DATACLASSES = [
    (
        "gaia_agent.config",
        "Settings",
        [
            "api_url",
            "data_dir",
            "model_id",
            "fallback_model_ids",
            "tpm_limit",
            "max_prompt_tokens",
            "max_output_tokens",
            "max_steps",
            "reasoning_effort",
            "hf_dataset",
            "hf_split_dir",
        ],
    ),
    ("gaia_agent.api", "Question", ["task_id", "question", "level", "file_name"]),
    (
        "gaia_agent.agent",
        "AnswerResult",
        ["task_id", "answer", "raw_answer", "steps", "seconds", "error", "model"],
    ),
]

# Tool name -> input names (in order) for every module that defines tools.
TOOLS = {
    "gaia_agent.tools.web": {
        "web_search": ["query", "max_results"],
        "read_webpage": ["url", "query"],
    },
    "gaia_agent.tools.wikipedia": {
        "wikipedia_search": ["query", "lang"],
        "wikipedia_page": ["title", "lang", "as_of", "section", "query"],
    },
    "gaia_agent.tools.media": {
        "youtube_transcript": ["url", "query"],
        "transcribe_audio": ["file_path"],
        "describe_image": ["file_path", "question"],
        "ask_about_video": ["url", "question"],
    },
    "gaia_agent.tools.deep_search": {
        "deep_search": ["query"],
    },
    "gaia_agent.tools.files": {
        "read_file": ["file_path"],
        "run_python_file": ["file_path"],
    },
}


def _resolve(module_name: str, qualified_name: str) -> Any:
    target: Any = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        target = getattr(target, part)
    return target


def _tool_classes(module: ModuleType) -> list[type[Tool]]:
    return [
        member
        for _, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, Tool) and member is not Tool and member.__module__ == module.__name__
    ]


@pytest.mark.parametrize(("module_name", "qualified_name", "expected"), SIGNATURES)
def test_callable_keeps_the_specified_parameters(module_name: str, qualified_name: str, expected: list[str]) -> None:
    parameters = list(inspect.signature(_resolve(module_name, qualified_name)).parameters.values())

    assert [p.name for p in parameters[: len(expected)]] == expected


@pytest.mark.parametrize(("module_name", "qualified_name", "expected"), SIGNATURES)
def test_callable_extra_parameters_are_optional(module_name: str, qualified_name: str, expected: list[str]) -> None:
    parameters = list(inspect.signature(_resolve(module_name, qualified_name)).parameters.values())

    for extra in parameters[len(expected) :]:
        assert extra.default is not inspect.Parameter.empty or extra.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ), f"{qualified_name}: extra parameter {extra.name!r} must have a default"


@pytest.mark.parametrize(("module_name", "qualified_name", "defaults"), DEFAULTS)
def test_callable_keeps_the_specified_defaults(module_name: str, qualified_name: str, defaults: dict[str, Any]) -> None:
    parameters = inspect.signature(_resolve(module_name, qualified_name)).parameters

    assert {name: parameters[name].default for name in defaults} == defaults


@pytest.mark.parametrize(("module_name", "class_name", "fields"), DATACLASSES)
def test_dataclass_is_frozen_and_keeps_its_fields(module_name: str, class_name: str, fields: list[str]) -> None:
    cls = _resolve(module_name, class_name)

    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen
    assert [f.name for f in dataclasses.fields(cls)] == fields


@pytest.mark.parametrize(("module_name", "expected"), TOOLS.items())
def test_module_defines_exactly_the_specified_tools(module_name: str, expected: dict[str, list[str]]) -> None:
    classes = _tool_classes(importlib.import_module(module_name))

    assert {cls.name: list(cls.inputs) for cls in classes} == expected
    assert all(cls.output_type == "string" for cls in classes)


def test_budgeted_model_is_a_smolagents_model() -> None:
    assert issubclass(_resolve("gaia_agent.budget", "BudgetedModel"), Model)


def test_prompt_too_large_error_is_an_exception() -> None:
    assert issubclass(_resolve("gaia_agent.budget", "PromptTooLargeError"), Exception)


def test_system_prompt_is_a_string() -> None:
    assert isinstance(_resolve("gaia_agent.prompts", "SYSTEM_PROMPT"), str)
