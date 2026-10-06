"""Tests for the sources the web tools refuse to read: places that publish the answers of the benchmark.

The agent has to find its answers, not copy them: a public file of other students' answers, the benchmark
dataset or a forum thread about the final assignment must stay out of reach, however a search leads to them.
Every address and page here is invented; the UUID-shaped task ids are built at run time.
"""

from __future__ import annotations

import uuid

import pytest
import responses
from web_samples import fake_ddgs, forbid_real_search, html_page, http, read, serve  # noqa: F401

from gaia_agent.tools import web
from gaia_agent.tools._sources import blocked_source, looks_like_answer_key
from gaia_agent.tools.web import search_web


def task_id(number: int) -> str:
    """A UUID-shaped task id that is not one: they are built here so that none appears in the source."""
    return str(uuid.UUID(int=number))


BLOCKED_URLS = [
    "https://huggingface.co/datasets/gaia-benchmark/GAIA",
    "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/metadata.parquet",
    "https://huggingface.co/datasets/someone/gaia-validation-answers/viewer",
    "https://huggingface.co/spaces/someone/agent_course/blob/main/answers.jsonl",
    "https://huggingface.co/spaces/someone/Final_Assignment_Template/tree/main",
    "https://huggingface.co/spaces/someone/unit4-agent/blob/main/app.py",
    "https://huggingface.co/spaces/gaia-benchmark/leaderboard",
    "https://someone-final-assignment.hf.space/",
    "https://agents-course-unit4-scoring.hf.space/questions",
    "https://discuss.huggingface.co/t/unit-4-final-assignment-answers/12345",
    "https://discuss.huggingface.co/t/gaia-level-1-questions-solutions/777",
    "https://discuss.huggingface.co/c/agents-course/45",
    "https://github.com/someone/gaia-answers",
    "https://github.com/someone/agents-course-final-assignment/blob/main/answers.json",
    "https://raw.githubusercontent.com/someone/gaia-solutions/main/answers.jsonl",
    "https://gist.github.com/someone/0123456789abcdef-gaia-answers",
    "https://gitlab.com/someone/gaia-validation",
    "https://www.kaggle.com/datasets/someone/gaia-validation-answers",
    "HTTPS://HuggingFace.co/Datasets/GAIA-Benchmark/GAIA",
    # the API, the mirrors, the rows server and repositories of any kind (models too) named after the benchmark
    "https://huggingface.co/api/datasets/gaia-benchmark/GAIA/tree/main/2023/validation",
    "https://huggingface.co/api/spaces/someone/unit4-agent",
    "https://datasets-server.huggingface.co/rows?dataset=gaia-benchmark/GAIA&config=2023_level1&split=validation",
    "https://datasets-server.huggingface.co/first-rows?dataset=someone%2Fcourse-answers&config=default",
    "https://hf-mirror.com/datasets/gaia-benchmark/GAIA",
    "https://hf-mirror.com/gaia-benchmark/GAIA/resolve/main/README.md",
    "https://hf.co/datasets/gaia-benchmark/GAIA",
    "https://huggingface.co/someone/gaia-answers",
    "https://huggingface.co/someone/gaia-answers/resolve/main/answers.jsonl",
    "https://huggingface.co/models?search=gaia-answers",
    "https://huggingface.co/spaces?search=final-assignment",
    "https://huggingface.co/datasets?search=gaia",
]

ALLOWED_URLS = [
    "https://en.wikipedia.org/wiki/Gaia_(spacecraft)",
    "https://www.esa.int/Science_Exploration/Space_Science/Gaia",
    "https://huggingface.co/learn/agents-course/unit0/introduction",
    "https://huggingface.co/docs/smolagents/index",
    "https://huggingface.co/spaces/someone/image-captioner",
    "https://huggingface.co/openai/gpt-oss-120b",
    "https://huggingface.co/docs/datasets/loading",
    "https://huggingface.co/api/models/openai/gpt-oss-120b",
    "https://hf-mirror.com/openai/gpt-oss-120b",
    "https://datasets-server.huggingface.co/rows?dataset=squad&config=plain_text&split=train",
    "https://huggingface.co/datasets/squad",
    "https://discuss.huggingface.co/t/how-do-i-fine-tune-a-model/99",
    "https://github.com/huggingface/smolagents",
    "https://github.com/someone/weather-app",
    "https://example.org/gaia-mission-overview",
    "https://www.nasa.gov/answers",
]


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_places_that_publish_benchmark_answers_are_blocked(url: str) -> None:
    assert blocked_source(url) is not None


@pytest.mark.parametrize("url", ALLOWED_URLS)
def test_ordinary_pages_are_not_blocked_even_when_they_mention_gaia_or_answers(url: str) -> None:
    assert blocked_source(url) is None


def test_the_reason_is_short_and_says_what_to_do() -> None:
    reason = blocked_source("https://github.com/someone/gaia-answers")

    assert reason is not None
    assert "answers" in reason
    assert len(reason) < 200


def test_a_url_that_cannot_be_parsed_is_not_blocked_here() -> None:
    assert blocked_source("https://[::1") is None  # the URL check reports it as malformed


# --- pages that are answer keys, wherever they are hosted ---------------------------------------------------------


def jsonl_key(count: int = 2) -> str:
    rows = [f'{{"task_id": "{task_id(n)}", "submitted_answer": "a{n}"}}' for n in range(1, count + 1)]
    return "\n".join(rows)


def test_a_file_of_task_ids_with_answers_is_an_answer_key() -> None:
    assert looks_like_answer_key(jsonl_key(3))


def test_a_markdown_table_of_task_ids_and_final_answers_is_an_answer_key() -> None:
    table = "\n".join(f"| {task_id(n)} | question {n} | Final answer: x{n} |" for n in range(1, 3))

    assert looks_like_answer_key("task_id | question | Final answer\n" + table)


def test_one_task_id_next_to_a_final_answer_is_enough() -> None:
    assert looks_like_answer_key(f"task_id: {task_id(7)}\nFinal answer: 42")


def test_a_single_task_id_in_a_tutorial_is_not_an_answer_key() -> None:
    tutorial = f'result = app.send_task("add", args=[2, 3])\nprint(result.task_id)  # task_id: {task_id(9)}'

    assert not looks_like_answer_key(tutorial)


def test_an_article_that_talks_about_final_answers_is_not_an_answer_key() -> None:
    assert not looks_like_answer_key("In the final answer of the quiz show, the contestant said Paris.")


def test_ordinary_text_with_ids_is_not_an_answer_key() -> None:
    assert not looks_like_answer_key(f"order {task_id(3)} shipped; order {task_id(4)} is pending")


def test_empty_text_is_not_an_answer_key() -> None:
    assert not looks_like_answer_key("")


# --- the web tools ------------------------------------------------------------------------------------------------


def test_read_webpage_refuses_a_blocked_address_without_a_request(http: responses.RequestsMock) -> None:
    result = read("https://huggingface.co/spaces/someone/agent_course/blob/main/answers.jsonl")

    assert result.startswith("Error: blocked source")
    assert len(http.calls) == 0


def test_a_redirect_to_a_blocked_address_is_refused_before_it_is_followed(http: responses.RequestsMock) -> None:
    http.add(
        responses.GET,
        "https://example.org/go",
        status=302,
        headers={"Location": "https://github.com/someone/gaia-answers/blob/main/answers.jsonl"},
    )

    result = read("https://example.org/go")

    assert result.startswith("Error: blocked source")
    assert len(http.calls) == 1


def test_read_webpage_refuses_a_page_that_turns_out_to_be_an_answer_key(http: responses.RequestsMock) -> None:
    serve(http, jsonl_key(3), content_type="text/plain", url="https://files.example.net/export.txt")

    result = read("https://files.example.net/export.txt")

    assert result.startswith("Error: blocked source")
    assert "a1" not in result  # none of the answers is passed on


def test_read_webpage_still_reads_ordinary_pages(http: responses.RequestsMock) -> None:
    serve(http, html_page("<p>Gaia launched in December 2013.</p>"))

    assert "Gaia launched in December 2013." in read()


def test_web_search_leaves_blocked_sources_out_of_the_results(fake_ddgs) -> None:  # noqa: F811
    fake_ddgs.rows = [
        {"title": "Answers", "href": "https://github.com/someone/gaia-answers", "body": "all the answers"},
        {"title": "A real page", "href": "https://example.org/real", "body": "useful"},
        {"title": "The dataset", "href": "https://huggingface.co/datasets/gaia-benchmark/GAIA", "body": "gated"},
    ]

    result = search_web("something to look up", 5)

    assert "https://example.org/real" in result
    assert "gaia-answers" not in result
    assert "gaia-benchmark" not in result


def test_web_search_asks_for_more_results_than_it_needs_to_make_up_for_the_ones_it_drops(fake_ddgs) -> None:  # noqa: F811
    fake_ddgs.rows = [{"title": "Real", "href": "https://example.org/real", "body": "useful"}]

    search_web("something", 5)

    assert fake_ddgs.calls[0][1]["max_results"] > 5


def test_web_search_says_so_when_nothing_but_blocked_sources_came_back(fake_ddgs, http: responses.RequestsMock) -> None:  # noqa: F811
    fake_ddgs.rows = [{"title": "Answers", "href": "https://github.com/someone/gaia-answers", "body": "all of them"}]
    http.add(responses.GET, web.WIKIPEDIA_API_URL, json={"query": {"search": []}})

    result = search_web("something to look up", 5)

    assert "gaia-answers" not in result
    assert result.startswith("No results")
