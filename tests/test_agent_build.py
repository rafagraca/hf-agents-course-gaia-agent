"""Tests for ``gaia_agent.agent.build_agent``: how the CodeAgent is wired. No network, no real model."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from agent_fakes import EchoTool, ScriptedModel, code_reply, final_reply, request_text
from smolagents import CodeAgent
from smolagents.monitoring import LogLevel

from gaia_agent.agent import AUTHORIZED_IMPORTS, CODE_TAGS, EXECUTOR_TIMEOUT_SECONDS, build_agent
from gaia_agent.budget import BudgetedModel
from gaia_agent.config import Settings
from gaia_agent.prompts import SYSTEM_PROMPT
from gaia_agent.reply_format import FormatGuardModel

SPEC_IMPORTS = {
    "math",
    "statistics",
    "re",
    "json",
    "datetime",
    "collections",
    "itertools",
    "fractions",
    "decimal",
    "unicodedata",
    "csv",
}


def test_it_builds_a_code_agent_with_the_settings_step_limit_and_no_planning(settings: Settings) -> None:
    agent = build_agent(dataclasses.replace(settings, max_steps=7), tools=[], model=ScriptedModel())

    assert isinstance(agent, CodeAgent)
    assert agent.max_steps == 7
    assert agent.planning_interval is None


def test_the_authorized_imports_are_the_specified_ones(settings: Settings) -> None:
    agent = build_agent(settings, tools=[], model=ScriptedModel())

    assert set(AUTHORIZED_IMPORTS) == SPEC_IMPORTS
    assert set(agent.authorized_imports) >= SPEC_IMPORTS
    assert "*" not in agent.authorized_imports


def test_it_uses_the_compact_system_prompt(settings: Settings) -> None:
    agent = build_agent(settings, tools=[], model=ScriptedModel())

    assert agent.prompt_templates["system_prompt"] == SYSTEM_PROMPT
    assert agent.system_prompt.startswith("Solve the task in cycles of Thought, Code and Observation.")


def test_the_instructions_give_the_step_limit_and_forbid_json_tool_calls(settings: Settings) -> None:
    prompt = build_agent(dataclasses.replace(settings, max_steps=6), tools=[], model=ScriptedModel()).system_prompt

    assert "at most 6 steps" in prompt
    assert "never emit a tool call or JSON" in prompt


def test_a_forced_final_answer_asks_for_the_bare_answer(settings: Settings) -> None:
    templates = build_agent(settings, tools=[], model=ScriptedModel()).prompt_templates

    post = templates["final_answer"]["post_messages"]
    assert "{{task}}" in post
    assert "ONLY the answer" in post


def test_the_console_is_silent_by_default_and_can_be_raised(settings: Settings) -> None:
    """Even the errors stay off the console: their text quotes the model's own replies (they are in the traces)."""
    quiet = build_agent(settings, tools=[], model=ScriptedModel())
    chatty = build_agent(settings, tools=[], model=ScriptedModel(), verbosity=LogLevel.INFO)

    assert quiet.logger.level == LogLevel.OFF
    assert chatty.logger.level == LogLevel.INFO


def test_an_error_inside_the_agent_prints_nothing_by_default(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    model = ScriptedModel([code_reply("{}['a secret key']"), final_reply("done")])
    agent = build_agent(settings, tools=[], model=model)

    agent.run("An invented task.")

    captured = capsys.readouterr()
    assert "a secret key" not in captured.out + captured.err


def test_the_code_block_tags_are_given_to_the_agent_and_the_guard_alike(settings: Settings) -> None:
    agent = build_agent(settings, tools=[], model=ScriptedModel())

    assert tuple(agent.code_block_tags) == CODE_TAGS


def test_a_usage_log_needs_the_default_model_because_a_given_model_would_ignore_it(
    settings: Settings, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="usage_log"):
        build_agent(settings, tools=[], model=ScriptedModel(), usage_log=tmp_path / "usage.jsonl")


def test_code_blocks_may_run_longer_than_the_smolagents_default(settings: Settings) -> None:
    agent = build_agent(settings, tools=[], model=ScriptedModel())

    assert EXECUTOR_TIMEOUT_SECONDS == 120
    assert agent.python_executor.timeout_seconds == EXECUTOR_TIMEOUT_SECONDS


def test_the_default_tools_are_used_when_none_are_given(settings: Settings) -> None:
    agent = build_agent(settings, model=ScriptedModel())

    assert {"web_search", "wikipedia_page", "read_file", "transcribe_audio", "final_answer"} <= set(agent.tools)


def test_the_given_tools_replace_the_defaults(settings: Settings) -> None:
    agent = build_agent(settings, tools=[EchoTool()], model=ScriptedModel())

    assert set(agent.tools) == {"echo_tool", "final_answer"}


def test_the_default_model_is_a_budgeted_model_over_the_configured_chain(settings: Settings, tmp_path: Path) -> None:
    custom = dataclasses.replace(settings, model_id="groq/main-model", fallback_model_ids=("groq/backup-model",))

    model = build_agent(custom, tools=[], usage_log=tmp_path / "usage.jsonl").model

    assert isinstance(model, FormatGuardModel)
    assert isinstance(model.inner, BudgetedModel)
    assert model.inner.model_ids == ("groq/main-model", "groq/backup-model")


def test_a_given_model_is_wrapped_in_the_format_guard(settings: Settings) -> None:
    model = ScriptedModel()

    agent = build_agent(settings, tools=[], model=model)

    assert isinstance(agent.model, FormatGuardModel)
    assert agent.model.inner is model


def test_agents_do_not_share_their_prompt_templates(settings: Settings) -> None:
    first = build_agent(settings, tools=[], model=ScriptedModel())
    second = build_agent(settings, tools=[], model=ScriptedModel())

    assert first.prompt_templates is not second.prompt_templates
    assert first.prompt_templates["final_answer"] is not second.prompt_templates["final_answer"]


def test_old_observations_are_trimmed_before_the_next_model_call(settings: Settings) -> None:
    model = ScriptedModel(
        [
            code_reply("print(echo_tool(text='first', pad_to=5000))"),
            code_reply("print(echo_tool(text='second'))"),
            final_reply("done"),
        ]
    )
    agent = build_agent(settings, tools=[EchoTool()], model=model)

    agent.run("An invented task.")

    second_request, third_request = (request_text(request) for request in model.requests[1:])
    assert "echo: first" + "." * 1000 in second_request  # the newest observation keeps its length
    assert "echo: first" + "." * 1000 not in third_request  # once older, it is cut short
    assert "[...trimmed]" in third_request
