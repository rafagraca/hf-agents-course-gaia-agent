"""Tests for ``web_search``: result formatting, limits and the Wikipedia fallback (plan B)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import responses
from web_samples import (  # noqa: F401 - the fixtures are used by name
    WIKIPEDIA_API,
    ddgs_row,
    fake_ddgs,
    forbid_real_search,
    http,
    wikipedia_payload,
)

from gaia_agent.tools import web
from gaia_agent.tools.web import SearchHit, WebSearchTool


def test_search_skips_rows_without_a_url_and_titles_untitled_ones_by_their_url(fake_ddgs: SimpleNamespace) -> None:
    fake_ddgs.rows = [{"title": "No link", "body": "x"}, {"title": "", "href": "https://example.org/a", "body": ""}]

    assert WebSearchTool().forward("q") == "1. https://example.org/a\n   https://example.org/a"


def test_search_returns_at_most_the_requested_number_of_results(fake_ddgs: SimpleNamespace) -> None:
    fake_ddgs.rows = [ddgs_row(number) for number in range(1, 9)]

    result = WebSearchTool().forward("q", max_results=3)

    assert result.count("https://example.org/page") == 3


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, 5), (3, 3), (0, 1), (-4, 1), (99, 10), ("4", 4), ("many", 5), (2.9, 2)],
)
def test_search_result_count_is_clamped(fake_ddgs: SimpleNamespace, requested: object, expected: int) -> None:
    fake_ddgs.rows = [ddgs_row(1)]

    WebSearchTool().forward("q", requested)  # type: ignore[arg-type]

    assert fake_ddgs.calls[0][1] == {"max_results": expected + web.SEARCH_OVERFETCH}


@pytest.mark.parametrize("query", ["", "   ", None])
def test_search_rejects_an_empty_query_without_searching(fake_ddgs: SimpleNamespace, query: object) -> None:
    result = WebSearchTool().forward(query)  # type: ignore[arg-type]

    assert result.startswith("Error:")
    assert fake_ddgs.calls == []


def test_search_collapses_whitespace_in_the_query(fake_ddgs: SimpleNamespace) -> None:
    fake_ddgs.rows = [ddgs_row(1)]

    WebSearchTool().forward("  spaced \n  query  ")

    assert fake_ddgs.calls[0][0] == "spaced query"


def test_search_gives_ddgs_a_timeout(fake_ddgs: SimpleNamespace) -> None:
    fake_ddgs.rows = [ddgs_row(1)]

    WebSearchTool().forward("q")

    assert fake_ddgs.init_kwargs == [{"timeout": web.SEARCH_TIMEOUT_S}]


def test_format_hits_leaves_out_an_empty_snippet() -> None:
    assert web.format_hits([SearchHit("T", "https://example.org/t", "")]) == "1. T\n   https://example.org/t"


def test_the_wikipedia_fallback_uses_the_mediawiki_search_api(
    fake_ddgs: SimpleNamespace, http: responses.RequestsMock
) -> None:
    fake_ddgs.error = RuntimeError("blocked")
    http.add(responses.GET, WIKIPEDIA_API, json=wikipedia_payload(("Alpha", "x")))

    WebSearchTool().forward("some query", max_results=7)

    request = http.calls[0].request
    params = parse_qs(urlsplit(request.url).query)
    assert params["action"] == ["query"]
    assert params["list"] == ["search"]
    assert params["srsearch"] == ["some query"]
    assert params["srlimit"] == ["7"]
    assert params["format"] == ["json"]
    assert "hf-agents-course-gaia-agent" in request.headers["User-Agent"]
    assert "Mozilla" not in request.headers["User-Agent"]


def test_the_fallback_respects_the_result_limit(fake_ddgs: SimpleNamespace, http: responses.RequestsMock) -> None:
    fake_ddgs.error = RuntimeError("blocked")
    hits = [(f"Title {number}", "s") for number in range(8)]
    http.add(responses.GET, WIKIPEDIA_API, json=wikipedia_payload(*hits))

    result = WebSearchTool().forward("q", max_results=2)

    assert result.count("https://en.wikipedia.org/wiki/") == 2


def test_nothing_found_anywhere_is_reported(fake_ddgs: SimpleNamespace, http: responses.RequestsMock) -> None:
    fake_ddgs.rows = []
    http.add(responses.GET, WIKIPEDIA_API, json=wikipedia_payload())

    result = WebSearchTool().forward("zzzz qqqq")

    assert result.startswith("No results for 'zzzz qqqq'")
    assert "plan B" in result


@pytest.mark.parametrize(
    ("registration", "reason"),
    [
        ({"status": 500, "body": "oops"}, "HTTP 500"),
        ({"body": requests.exceptions.ConnectTimeout("slow")}, "timed out"),
        ({"body": requests.exceptions.ConnectionError("down")}, "ConnectionError"),
        ({"json": {"error": {"code": "internal_api_error"}}}, "Wikipedia API error"),
        ({"body": "<html>not json</html>"}, "unexpected"),
        ({"json": {"query": {"search": "nope"}}}, "unexpected"),
        ({"json": ["not", "an", "object"]}, "unexpected"),
    ],
)
def test_both_searches_failing_gives_a_short_error(
    fake_ddgs: SimpleNamespace, http: responses.RequestsMock, registration: dict[str, object], reason: str
) -> None:
    fake_ddgs.error = RuntimeError("blocked")
    http.add(responses.GET, WIKIPEDIA_API, **registration)

    result = WebSearchTool().forward("q")

    assert result.startswith("Error: the web search failed (RuntimeError)")
    assert "Wikipedia fallback failed too" in result
    assert reason in result
    assert "\n" not in result


def test_search_keeps_going_when_ddgs_raises_unexpected_errors(
    fake_ddgs: SimpleNamespace, http: responses.RequestsMock, caplog: pytest.LogCaptureFixture
) -> None:
    fake_ddgs.error = ValueError("odd")
    http.add(responses.GET, WIKIPEDIA_API, json=wikipedia_payload(("Alpha", "x")))

    with caplog.at_level(logging.WARNING, logger="gaia_agent.tools.web"):
        result = WebSearchTool().forward("alpha")

    assert "Alpha" in result
    assert "ValueError" in caplog.text
