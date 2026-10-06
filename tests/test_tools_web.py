"""Tests for the two web tools, ``web_search`` and ``read_webpage``, as an agent uses them.

The main scenarios live here; the details of searching and fetching are in ``test_web_search.py``
and ``test_web_fetch.py``. No test touches the network: HTTP is mocked with ``responses`` and
ddgs is replaced by a fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import responses
from web_samples import (  # noqa: F401 - the fixtures are used by name
    PAGE_URL,
    WIKIPEDIA_API,
    ddgs_row,
    fake_ddgs,
    forbid_real_search,
    html_page,
    http,
    make_pdf,
    paragraphs,
    read,
    serve,
    wikipedia_payload,
)

from gaia_agent.tools.web import SEARCH_OVERFETCH, ReadWebpageTool, WebSearchTool
from gaia_agent.tools.web_docs import html_to_markdown
from gaia_agent.tools.web_text import truncation_note


def test_search_lists_numbered_results_with_title_url_and_snippet(fake_ddgs: SimpleNamespace) -> None:
    fake_ddgs.rows = [ddgs_row(1, "First description."), ddgs_row(2, "Second description.")]

    result = WebSearchTool().forward("example query")

    assert result == (
        "1. Result 1\n   https://example.org/page1\n   First description.\n"
        "2. Result 2\n   https://example.org/page2\n   Second description."
    )
    assert fake_ddgs.calls == [("example query", {"max_results": 5 + SEARCH_OVERFETCH})]


def test_search_snippets_are_collapsed_and_capped_at_220_characters(fake_ddgs: SimpleNamespace) -> None:
    fake_ddgs.rows = [ddgs_row(1, "word   " * 100 + "\n tail")]

    snippet = WebSearchTool().forward("q").splitlines()[2].strip()

    assert len(snippet) <= 220
    assert snippet.endswith("...")
    assert "  " not in snippet


def test_search_can_be_called_like_any_smolagents_tool(fake_ddgs: SimpleNamespace) -> None:
    fake_ddgs.rows = [ddgs_row(1)]

    assert WebSearchTool()(query="example").startswith("1. Result 1")


def test_a_failed_search_falls_back_to_wikipedia_and_says_so(
    fake_ddgs: SimpleNamespace, http: responses.RequestsMock
) -> None:
    fake_ddgs.error = RuntimeError("blocked")
    snippet = 'The <span class="searchmatch">highest</span> mountain &amp; more'
    http.add(
        responses.GET,
        WIKIPEDIA_API,
        json=wikipedia_payload(("Mount Everest", snippet), ("K2 (mountain)", "Second highest")),
    )

    result = WebSearchTool().forward("highest mountain")

    assert "plan B" in result
    assert "failed (RuntimeError)" in result
    assert "1. Mount Everest\n   https://en.wikipedia.org/wiki/Mount_Everest\n   The highest mountain & more" in result
    assert "https://en.wikipedia.org/wiki/K2_(mountain)" in result


def test_a_search_with_no_results_also_falls_back_to_wikipedia(
    fake_ddgs: SimpleNamespace, http: responses.RequestsMock
) -> None:
    fake_ddgs.rows = []
    http.add(responses.GET, WIKIPEDIA_API, json=wikipedia_payload(("Alpha", "About alpha")))

    result = WebSearchTool().forward("alpha")

    assert "returned no results" in result
    assert "plan B" in result
    assert "https://en.wikipedia.org/wiki/Alpha" in result


def test_a_page_becomes_markdown_with_tables_and_absolute_links(http: responses.RequestsMock) -> None:
    page = html_page(
        "<h1>Albums</h1><table><tr><th>Year</th><th>Title</th></tr>"
        "<tr><td>2001</td><td><a href='/wiki/Alpha'>Alpha</a></td></tr></table>"
    )
    serve(http, page)

    result = read()

    assert "# Albums" in result
    assert "| Year | Title |" in result
    assert "| 2001 | [Alpha](https://example.org/wiki/Alpha) |" in result


def test_the_request_looks_like_a_browser_and_times_out_after_30_seconds(http: responses.RequestsMock) -> None:
    serve(http, html_page("<p>hello</p>"))

    read()

    request = http.calls[0].request
    assert request.headers["User-Agent"].startswith("Mozilla/5.0")
    assert request.req_kwargs["timeout"] == 30


def test_a_long_page_without_a_query_shows_its_start_and_the_total_length(http: responses.RequestsMock) -> None:
    page = html_page(paragraphs(60))
    serve(http, page)
    total = len(html_to_markdown(page, base_url=PAGE_URL))

    result = read()

    assert result.startswith("# Sample page")
    assert result.endswith(truncation_note(total))
    assert len(result) <= 5000 + len(truncation_note(total)) + 1
    assert "Paragraph 59." not in result


def test_a_query_returns_the_passages_that_match_it(http: responses.RequestsMock) -> None:
    needle = "The marmot colony counted 321 burrows."
    serve(http, html_page(paragraphs(60, {44: needle})))

    result = read(query="marmot burrows")

    assert needle in result
    assert "Paragraph 3." not in result
    assert "[truncated: " in result
    assert len(result) <= 4100


def test_a_short_page_is_returned_whole_even_with_a_query(http: responses.RequestsMock) -> None:
    serve(http, html_page("<p>Tiny page about okapis.</p>"))

    assert "Tiny page about okapis." in read(query="okapis")


def test_a_query_that_matches_nothing_says_so(http: responses.RequestsMock) -> None:
    serve(http, html_page(paragraphs(60)))

    result = read(query="zeppelin")

    assert result.startswith("[no passage matched the query")
    assert "Paragraph 0." in result


def test_a_missing_query_is_treated_as_no_query(http: responses.RequestsMock) -> None:
    serve(http, html_page("<p>Plain page.</p>"))

    assert "Plain page." in ReadWebpageTool().forward(PAGE_URL, None)  # type: ignore[arg-type]


def test_pages_in_other_charsets_are_decoded(http: responses.RequestsMock) -> None:
    serve(http, html_page("<p>Un café très chaud</p>").encode("latin-1"), content_type="text/html; charset=iso-8859-1")

    assert "Un café très chaud" in read()


def test_the_tool_can_be_called_like_any_smolagents_tool(http: responses.RequestsMock) -> None:
    serve(http, html_page("<p>hello there</p>"))

    assert "hello there" in ReadWebpageTool()(url=PAGE_URL)


def test_a_pdf_is_read_as_text_by_its_content_type(http: responses.RequestsMock) -> None:
    url = "https://example.org/get?id=7"
    serve(http, make_pdf([["Annual report", "Total revenue was 42 million"]]), content_type="application/pdf", url=url)

    result = read(url)

    assert "Annual report" in result
    assert "Total revenue was 42 million" in result


def test_a_pdf_is_read_by_its_extension_when_the_type_is_generic(http: responses.RequestsMock) -> None:
    url = "https://example.org/files/paper.pdf"
    serve(http, make_pdf([["Findings of the study"]]), content_type="application/octet-stream", url=url)

    assert "Findings of the study" in read(url)


def test_a_query_can_pick_passages_out_of_a_pdf(http: responses.RequestsMock) -> None:
    needle = "The glacier retreated 87 metres in 2019."
    routine = "routine remarks about the committee budget and the schedule."
    pages = [[f"Page {number}: {routine}"] * 3 + ([needle] if number == 30 else []) for number in range(40)]
    url = "https://example.org/files/report.pdf"
    serve(http, make_pdf(pages), content_type="application/pdf", url=url)

    result = read(url, "glacier retreated")

    assert needle in result
    assert "Page 3:" not in result


def test_plain_text_is_returned_as_it_is(http: responses.RequestsMock) -> None:
    url = "https://example.org/notes.txt"
    serve(http, "first line\r\nsecond line\r\n", content_type="text/plain; charset=utf-8", url=url)

    assert read(url) == "first line\nsecond line"


def test_json_is_returned_as_it_is(http: responses.RequestsMock) -> None:
    url = "https://example.org/data.json"
    serve(http, '{"year": 2001}', content_type="application/json", url=url)

    assert read(url) == '{"year": 2001}'


def test_binary_content_is_refused(http: responses.RequestsMock) -> None:
    url = "https://example.org/logo.png"
    serve(http, b"\x89PNG\r\n\x1a\n\x00\x00", content_type="image/png", url=url)

    result = read(url)

    assert result.startswith("Error:")
    assert "image/png" in result


def test_a_page_without_text_is_reported(http: responses.RequestsMock) -> None:
    serve(http, html_page("<script>document.write('rendered later')</script>"))

    result = read()

    assert result.startswith("Error:")
    assert "no readable text" in result


@pytest.mark.parametrize(("status", "reason"), [(404, "Not Found"), (403, "Forbidden"), (503, "Service Unavailable")])
def test_http_errors_become_short_messages(http: responses.RequestsMock, status: int, reason: str) -> None:
    serve(http, "<h1>error page</h1>", status=status)

    assert read() == f"Error: HTTP {status} {reason}"
