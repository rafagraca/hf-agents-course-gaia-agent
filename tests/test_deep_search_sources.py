"""``deep_search`` must not hand the agent anything that came from a place that publishes benchmark answers.

Groq's browser search reads pages on its own servers, so the guard is applied to what comes back: every link of
the reply (citations, the answer's text and everything the browser searched or opened, not only the five
links that are shown) and the text itself. Every address, id and page here is invented.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
import responses
from test_deep_search import URL, Sleeper, groq_key, http, reply, sleeper, tool  # noqa: F401  (fixtures too)

from gaia_agent.tools._groq_search import build_payload
from gaia_agent.tools.deep_search import DeepSearchTool

BLOCKED = "https://huggingface.co/datasets/someone/gaia-validation-answers/viewer"
SAFE = "https://example.org/alpha"
EXPECTED = "deep_search unavailable: the search used a blocked source."


def task_id(number: int) -> str:
    """A UUID-shaped id that is not a task id: built here so that none appears in the source."""
    return str(uuid.UUID(int=number))


def assert_refused(result: str, secret: str = "An invented answer.") -> None:
    assert result.startswith(EXPECTED)
    assert result.endswith("Use web_search and read_webpage instead.")
    assert secret not in result
    assert "Sources:" not in result


def test_a_blocked_page_among_the_searched_results_refuses_the_whole_reply(
    http: responses.RequestsMock, tool: DeepSearchTool
) -> None:
    http.add(responses.POST, URL, json=reply(results=(SAFE, BLOCKED)))

    assert_refused(tool("q"))


def test_a_blocked_citation_refuses_the_whole_reply(http: responses.RequestsMock, tool: DeepSearchTool) -> None:
    http.add(responses.POST, URL, json=reply(annotations=(BLOCKED,)))

    assert_refused(tool("q"))


def test_a_blocked_link_quoted_in_the_answer_refuses_the_whole_reply(
    http: responses.RequestsMock, tool: DeepSearchTool
) -> None:
    http.add(responses.POST, URL, json=reply(f"It says so at {BLOCKED}."))

    assert_refused(tool("q"), secret="It says so")


def test_a_blocked_link_beyond_the_five_shown_ones_is_still_caught(
    http: responses.RequestsMock, tool: DeepSearchTool
) -> None:
    others = tuple(f"https://example.org/page{i}" for i in range(7))
    http.add(responses.POST, URL, json=reply(results=(*others, BLOCKED)))

    assert_refused(tool("q"))


def test_a_link_in_the_arguments_of_a_tool_the_browser_ran_is_caught(
    http: responses.RequestsMock, tool: DeepSearchTool
) -> None:
    body = reply()
    body["choices"][0]["message"]["executed_tools"] = [
        {"type": "browser.open", "arguments": json.dumps({"url": BLOCKED})},
    ]
    http.add(responses.POST, URL, json=body)

    assert_refused(tool("q"))


def test_a_repository_of_solutions_and_a_space_are_caught_too(
    http: responses.RequestsMock, tool: DeepSearchTool
) -> None:
    for blocked in (
        "https://github.com/someone/gaia-answers",
        "https://someone-final-assignment.hf.space/",
        "https://huggingface.co/spaces/someone/unit4-agent/blob/main/app.py",
    ):
        http.add(responses.POST, URL, json=reply(results=(blocked,)))
        tool.reset_question()
        assert_refused(tool("q"))


def test_a_reply_whose_text_looks_like_an_answer_key_is_refused(
    http: responses.RequestsMock, tool: DeepSearchTool
) -> None:
    rows = "\n".join(f'{{"task_id": "{task_id(n)}", "submitted_answer": "a{n}"}}' for n in (1, 2))
    http.add(responses.POST, URL, json=reply(rows))

    assert_refused(tool("q"), secret="submitted_answer")


def test_a_page_the_browser_fetched_that_looks_like_an_answer_key_is_refused(
    http: responses.RequestsMock, tool: DeepSearchTool
) -> None:
    rows = "\n".join(f'{{"task_id": "{task_id(n)}", "submitted_answer": "a{n}"}}' for n in (1, 2))
    body = reply()
    body["choices"][0]["message"]["executed_tools"] = [
        {"type": "search", "search_results": {"results": [{"title": "t", "url": SAFE, "content": rows}]}}
    ]
    http.add(responses.POST, URL, json=body)

    assert_refused(tool("q"))


def test_a_refused_reply_still_logs_the_tokens_it_cost(
    http: responses.RequestsMock, sleeper: Sleeper, tmp_path: Path
) -> None:
    log = tmp_path / "usage.jsonl"
    http.add(
        responses.POST, URL, json=reply(results=(BLOCKED,), usage={"prompt_tokens": 7000, "completion_tokens": 50})
    )

    DeepSearchTool(usage_log=log, sleep=sleeper)("q")

    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["input_tokens"] for line in lines] == [7000]


def test_a_refused_reply_still_counts_as_a_use(http: responses.RequestsMock, tool: DeepSearchTool) -> None:
    http.add(responses.POST, URL, json=reply(results=(BLOCKED,)))

    tool("a")
    tool("b")

    assert len(http.calls) == 2
    assert "question" in tool("c")


def test_ordinary_sources_and_ordinary_text_are_not_refused(http: responses.RequestsMock, tool: DeepSearchTool) -> None:
    text = f"An invented answer. See https://en.wikipedia.org/wiki/Gaia_(spacecraft) and task_id: {task_id(5)}."
    http.add(responses.POST, URL, json=reply(text, results=(SAFE, "https://huggingface.co/docs/smolagents")))

    result = tool("q")

    assert result.startswith("An invented answer.")
    assert SAFE in result


def test_the_system_message_forbids_the_benchmark_and_places_with_answers() -> None:
    system = build_payload("q")["messages"][0]["content"].lower()

    assert "benchmark" in system
    assert "dataset" in system
    assert "answers" in system


def test_a_reply_that_is_not_a_dict_message_does_not_crash(http: responses.RequestsMock, tool: DeepSearchTool) -> None:
    body: dict[str, Any] = reply()
    body["choices"][0]["message"]["executed_tools"] = "not a list"
    http.add(responses.POST, URL, json=body)

    assert tool("q").startswith("An invented answer.")


@pytest.mark.parametrize("junk", [None, 5, [None, 3], {"a": {"b": None}}])
def test_odd_shapes_under_the_tool_results_are_ignored(
    http: responses.RequestsMock, tool: DeepSearchTool, junk: object
) -> None:
    body = reply()
    body["choices"][0]["message"]["executed_tools"] = [{"type": "search", "search_results": junk}]
    http.add(responses.POST, URL, json=body)

    assert tool("q").startswith("An invented answer.")
