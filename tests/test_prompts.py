"""Tests for the compact CodeAgent system prompt and ``build_prompt_templates``.

The prompt is a Jinja template that smolagents renders, so these tests render it the
way smolagents does (``populate_template`` with the variables ``CodeAgent`` passes)
and drive a real ``CodeAgent`` with a scripted model to prove that the format the
prompt asks for is the one the smolagents parser accepts. No network is used.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.resources
import inspect
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import jinja2
import jinja2.meta
import pytest
import yaml
from smolagents import CodeAgent, Tool
from smolagents.agents import populate_template
from smolagents.default_tools import FinalAnswerTool
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import LogLevel
from smolagents.utils import BASE_BUILTIN_MODULES, extract_code_from_text

from gaia_agent.prompts import SYSTEM_PROMPT, build_prompt_templates

TOKEN_TARGET = 1300  # system prompt plus the rendered tool list, as fixed by the project spec
# The prompt text alone (imports included) guards against slow growth: 420, then the safety and deep_search rules.
OWN_TEXT_TOKEN_CAP = 460
CHARS_PER_TOKEN = 3.5
ANGLE_TAGS = ("<code>", "</code>")
MARKDOWN_TAGS = ("```python", "```")
TAG_STYLES = pytest.mark.parametrize("tags", [ANGLE_TAGS, MARKDOWN_TAGS], ids=["angle", "markdown"])
TEMPLATE_VARIABLES = {
    "tools",
    "managed_agents",
    "authorized_imports",
    "custom_instructions",
    "code_block_opening_tag",
    "code_block_closing_tag",
}
TOOL_MODULES = (
    "gaia_agent.tools.web",
    "gaia_agent.tools.wikipedia",
    "gaia_agent.tools.files",
    "gaia_agent.tools.media",
)
# A realistic list of authorized imports: smolagents' base modules plus a few the agent may add.
IMPORTS = str(sorted(set(BASE_BUILTIN_MODULES) | {"csv", "json", "numpy", "pandas", "sympy", "os"}))

# The few concepts the prompt must keep teaching: the protocol, the rule the format guard relies on, the grading
# and the one about untrusted text. Each regex is matched case-insensitively across lines. The wording of the rest is
# free to change: a prompt is tuned by experiment, and a test must not stand in the way of a better sentence.
REQUIRED_INSTRUCTIONS = {
    "thought-code-observation cycle": r"Thought.*Code.*Observation",
    "ends with final_answer": r"final_answer\(answer\)",
    "tools instead of guessing": r"instead of guessing",
    "see the evidence before final_answer": r"seen the evidence",
    "exact match": r"exact match",
    "only the answer": r"only the answer",
    "text read from pages and files is untrusted": r"untrusted data",
}


def estimate_tokens(text: str) -> float:
    """The cheap token estimate used for the budget: characters divided by 3.5."""
    return len(text) / CHARS_PER_TOKEN


def render(
    tools: Mapping[str, Any],
    tags: tuple[str, str] = ANGLE_TAGS,
    managed_agents: Mapping[str, Any] | None = None,
    custom_instructions: str | None = None,
) -> str:
    """Render ``SYSTEM_PROMPT`` with the variables ``CodeAgent`` passes to ``populate_template``."""
    variables = {
        "tools": dict(tools),
        "managed_agents": dict(managed_agents or {}),
        "authorized_imports": IMPORTS,
        "custom_instructions": custom_instructions,
        "code_block_opening_tag": tags[0],
        "code_block_closing_tag": tags[1],
    }
    return populate_template(SYSTEM_PROMPT, variables)


@pytest.fixture(scope="module")
def default_templates() -> dict[str, Any]:
    """The CodeAgent templates shipped with the installed smolagents, loaded as smolagents loads them."""
    text = importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


@pytest.fixture
def tools() -> dict[str, Tool]:
    """Every tool of the project plus ``final_answer``: the worst-case tool list (Gemini tools included).

    Function-scoped on purpose: the autouse isolation fixtures of ``conftest.py`` (temporary DATA folder,
    hidden secrets) only apply to function-scoped setup, and tool constructors may read settings.
    """
    found: dict[str, Tool] = {}
    for module_name in TOOL_MODULES:
        module = importlib.import_module(module_name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Tool) and cls is not Tool and cls.__module__ == module.__name__:
                try:
                    tool = cls()
                except TypeError as exc:  # a skip here would hide the whole rendering suite
                    pytest.fail(f"{cls.__name__} needs constructor arguments: build the tool list another way ({exc})")
                found[tool.name] = tool
    return {**found, "final_answer": FinalAnswerTool()}


# --- The template itself ------------------------------------------------------------------------


def test_system_prompt_is_a_non_empty_template() -> None:
    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT.strip()


def test_template_uses_exactly_the_variables_smolagents_provides() -> None:
    parsed = jinja2.Environment().parse(SYSTEM_PROMPT)

    assert jinja2.meta.find_undeclared_variables(parsed) == TEMPLATE_VARIABLES


def test_code_block_tags_are_template_variables_not_literals() -> None:
    assert "<code>" not in SYSTEM_PROMPT
    assert "```" not in SYSTEM_PROMPT


def test_prompt_never_asks_for_a_final_answer_template() -> None:
    assert "FINAL ANSWER:" not in SYSTEM_PROMPT


@pytest.mark.parametrize("pattern", REQUIRED_INSTRUCTIONS.values(), ids=REQUIRED_INSTRUCTIONS.keys())
def test_prompt_keeps_teaching_the_instruction(pattern: str) -> None:
    rendered = render({})

    assert re.search(pattern, rendered, re.IGNORECASE | re.DOTALL), f"the prompt no longer matches {pattern!r}"


# --- Rendering ----------------------------------------------------------------------------------


@TAG_STYLES
def test_renders_cleanly_with_both_tag_styles(tools: dict[str, Tool], tags: tuple[str, str]) -> None:
    rendered = render(tools, tags)

    assert "{{" not in rendered
    assert "{%" not in rendered
    assert "\n\n\n" not in rendered
    assert rendered.count(tags[0]) >= 2
    assert rendered.count(tags[1]) >= 2
    assert IMPORTS in rendered


@TAG_STYLES
def test_the_step_format_is_shown_the_way_the_parser_reads_it(tags: tuple[str, str]) -> None:
    opening, closing = (re.escape(tag) for tag in tags)

    assert re.search(rf"Thought:[^\n]*\n{opening}\n[^\n]*\n{closing}\n", render({}, tags))


@TAG_STYLES
def test_every_code_block_in_the_prompt_is_valid_python(tools: dict[str, Tool], tags: tuple[str, str]) -> None:
    code = extract_code_from_text(render(tools, tags), tags)

    assert code
    ast.parse(code)


def test_every_tool_is_listed_exactly_once(tools: dict[str, Tool]) -> None:
    rendered = render(tools)

    for name in tools:
        assert rendered.count(f"def {name}(") == 1, name


def test_custom_instructions_appear_only_when_given() -> None:
    assert "Prefer metric units." in render({}, custom_instructions="Prefer metric units.")
    assert "None" not in render({})


def test_managed_agents_are_listed_when_present() -> None:
    team = {"researcher": SimpleNamespace(name="researcher", description="Finds primary sources.")}

    with_team = render({}, managed_agents=team)

    assert "def researcher(task: str" in with_team
    assert "Finds primary sources." in with_team
    assert "researcher" not in render({})


# --- Size ---------------------------------------------------------------------------------------


def test_prompt_with_the_tool_list_fits_the_token_target(tools: dict[str, Tool]) -> None:
    total = estimate_tokens(render(tools))
    own = estimate_tokens(render({}))

    assert total <= TOKEN_TARGET, (
        f"system prompt is about {total:.0f} tokens (target {TOKEN_TARGET}): "
        f"{own:.0f} for the prompt text, {total - own:.0f} for the {len(tools)} rendered tools and the imports"
    )


def test_prompt_text_alone_stays_compact() -> None:
    assert estimate_tokens(render({})) <= OWN_TEXT_TOKEN_CAP


# --- build_prompt_templates ---------------------------------------------------------------------


def test_build_prompt_templates_swaps_in_the_compact_system_prompt(default_templates: dict[str, Any]) -> None:
    result = build_prompt_templates(default_templates)

    assert result is not default_templates
    assert result["system_prompt"] == SYSTEM_PROMPT
    assert result["system_prompt"] != default_templates["system_prompt"]


def test_build_prompt_templates_keeps_the_other_templates_intact(default_templates: dict[str, Any]) -> None:
    result = build_prompt_templates(default_templates)

    assert set(result) == set(default_templates)
    for key in set(default_templates) - {"system_prompt"}:
        assert result[key] == default_templates[key]
        assert result[key] is not default_templates[key]


def test_build_prompt_templates_does_not_modify_its_input(default_templates: dict[str, Any]) -> None:
    snapshot = copy.deepcopy(default_templates)

    result = build_prompt_templates(default_templates)
    result["planning"]["initial_plan"] = "changed"
    result["final_answer"]["pre_messages"] = "changed"

    assert default_templates == snapshot


def test_build_prompt_templates_rejects_templates_without_a_system_prompt() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        build_prompt_templates({"planning": {}})


@pytest.mark.parametrize("bad", [None, "system_prompt", ["system_prompt"]])
def test_build_prompt_templates_rejects_non_mappings(bad: Any) -> None:
    with pytest.raises(TypeError):
        build_prompt_templates(bad)


# --- A real CodeAgent driven by a scripted model ------------------------------------------------


class ScriptedModel(Model):
    """Replays canned replies and records the messages it is asked to complete."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(model_id="scripted-model")
        self._replies = list(replies)
        self.requests: list[list[ChatMessage]] = []

    def generate(
        self, messages: list[ChatMessage], stop_sequences: list[str] | None = None, **kwargs: Any
    ) -> ChatMessage:
        self.requests.append(list(messages))
        return ChatMessage(role=MessageRole.ASSISTANT, content=self._replies.pop(0))


def make_agent(model: Model, templates: dict[str, Any] | None, tools: list[Tool], markdown: bool = False) -> CodeAgent:
    return CodeAgent(
        tools=tools,
        model=model,
        prompt_templates=templates,
        max_steps=3,
        verbosity_level=LogLevel.OFF,
        code_block_tags="markdown" if markdown else None,
    )


def system_text(request: list[ChatMessage]) -> str:
    return request[0].content[0]["text"]


def project_tools(tools: dict[str, Tool]) -> list[Tool]:
    return [tool for name, tool in tools.items() if name != "final_answer"]


def test_code_agent_accepts_the_templates_and_the_prompt_is_much_smaller_than_the_default(
    default_templates: dict[str, Any], tools: dict[str, Tool]
) -> None:
    compact = make_agent(ScriptedModel([]), build_prompt_templates(default_templates), project_tools(tools))
    default = make_agent(ScriptedModel([]), None, project_tools(tools))

    assert len(compact.system_prompt) < len(default.system_prompt) / 2
    for name in tools:
        assert f"def {name}(" in compact.system_prompt


@pytest.mark.parametrize("markdown", [False, True], ids=["angle", "markdown"])
def test_scripted_run_finishes_with_final_answer_in_the_tag_style_of_the_prompt(
    default_templates: dict[str, Any], tools: dict[str, Tool], markdown: bool
) -> None:
    opening, closing = MARKDOWN_TAGS if markdown else ANGLE_TAGS
    model = ScriptedModel([f"Thought: trivial.\n{opening}\nfinal_answer('42')\n{closing}"])
    agent = make_agent(model, build_prompt_templates(default_templates), project_tools(tools), markdown)

    answer = agent.run("What is six times seven?")

    assert agent.code_block_tags == (opening, closing)
    assert answer == "42"
    assert system_text(model.requests[0]) == agent.system_prompt
    assert f"{opening}\n" in agent.system_prompt


def test_scripted_run_returns_printed_output_as_an_observation(
    default_templates: dict[str, Any], tools: dict[str, Tool]
) -> None:
    model = ScriptedModel(
        [
            "Thought: compute it.\n<code>\nprint(6 * 7)\n</code>",
            "Thought: done.\n<code>\nfinal_answer(str(6 * 7))\n</code>",
        ]
    )
    agent = make_agent(model, build_prompt_templates(default_templates), project_tools(tools))

    answer = agent.run("What is six times seven?")

    assert answer == "42"
    observation = model.requests[1][-1].content[0]["text"]
    assert observation.startswith("Observation:")
    assert "42" in observation
