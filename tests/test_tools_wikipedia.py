"""Tests for the Wikipedia tools: searching, reading a page, languages and the smolagents interface.

The MediaWiki API is faked (see ``wikipedia_fakes.py``): nothing here touches the network.
Sections, output size and failures are covered by the two ``test_tools_wikipedia_*`` modules.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from wikipedia_fakes import (
    ARTICLE_HTML,
    ARTICLE_MARKDOWN,
    SECTIONS,
    USER_AGENT,
    FakeWikipedia,
    fake_wikipedia,
    page,
    read,
    revisions,
    search_reply,
)

from gaia_agent.tools.wikipedia import WikipediaPageTool, WikipediaSearchTool


@pytest.fixture
def wiki() -> Iterator[FakeWikipedia]:
    yield from fake_wikipedia()


# --- wikipedia_search ---------------------------------------------------------------------------


def test_search_lists_titles_with_clean_snippets(wiki: FakeWikipedia) -> None:
    wiki.reply(
        "search",
        search_reply(
            (
                "Example Band",
                '<span class="searchmatch">Example</span> Band is a &quot;fictional&quot; group &amp; more',
            ),
            ("History of Example Band", 'The <span class="searchmatch">band</span> formed in 1990.'),
        ),
    )

    result = WikipediaSearchTool().forward("example band")

    assert result == (
        'Wikipedia (en) results for "example band":\n'
        '1. Example Band - Example Band is a "fictional" group & more\n'
        "2. History of Example Band - The band formed in 1990."
    )
    params = wiki.params_of("search")
    assert params["action"] == "query"
    assert params["srsearch"] == "example band"
    assert params["srlimit"] == "8"


def test_search_identifies_itself_with_the_project_user_agent(wiki: FakeWikipedia) -> None:
    wiki.reply("search", search_reply(("Example Band", "x")))

    WikipediaSearchTool().forward("example band")

    assert wiki.seen[0].headers["User-Agent"] == USER_AGENT
    assert wiki.seen[0].host == "en.wikipedia.org"


def test_search_asks_the_requested_language_edition(wiki: FakeWikipedia) -> None:
    wiki.reply("search", search_reply(("Grupo Exemplo", "Um grupo fictício.")))

    result = WikipediaSearchTool().forward("grupo", lang="pt")

    assert wiki.seen[0].host == "pt.wikipedia.org"
    assert result.startswith('Wikipedia (pt) results for "grupo":')


def test_search_shows_at_most_eight_results_with_short_snippets(wiki: FakeWikipedia) -> None:
    hits = [(f"Title {number}", "word " * 200) for number in range(10)]
    wiki.reply("search", search_reply(*hits))

    lines = WikipediaSearchTool().forward("title").splitlines()

    assert len(lines) == 1 + 8
    assert all(len(line) < 260 for line in lines)
    assert lines[1].endswith("...")


def test_search_offers_the_suggestion_the_api_gives(wiki: FakeWikipedia) -> None:
    wiki.reply("search", search_reply(("Example Band", "A group."), suggestion="example band"))

    assert WikipediaSearchTool().forward("exampel band").endswith('Did you mean: "example band"?')


def test_search_without_results(wiki: FakeWikipedia) -> None:
    wiki.reply("search", search_reply(suggestion="example"))

    assert WikipediaSearchTool().forward("exmple") == (
        'No Wikipedia (en) results for "exmple".\nDid you mean: "example"?'
    )


def test_language_codes_are_case_insensitive(wiki: FakeWikipedia) -> None:
    wiki.reply("search", search_reply(("Example Band", "x")))

    WikipediaSearchTool().forward("example", lang=" EN ")

    assert wiki.seen[0].host == "en.wikipedia.org"


# --- wikipedia_page: reading ----------------------------------------------------------------------


def test_page_reads_the_current_revision_as_markdown(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", page(ARTICLE_HTML, SECTIONS))

    result = read(title="Example Band")

    header, _, body = result.partition("\n")
    assert header == '[Wikipedia (en) "Example Band" | revision 2002 of 2026-09-30T08:00:00Z (current)]'
    assert body == ARTICLE_MARKDOWN


def test_page_requests_follow_redirects_and_pin_the_revision(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", page(ARTICLE_HTML, SECTIONS))

    read(title="Example Band")

    revision_params = wiki.params_of("revisions")
    assert revision_params["titles"] == "Example Band"
    assert revision_params["redirects"] == "1"
    assert revision_params["rvlimit"] == "1"
    assert "rvstart" not in revision_params
    assert wiki.seen[0].headers["User-Agent"] == USER_AGENT
    assert revision_params["format"] == "json"
    page_params = wiki.params_of("page")
    assert page_params["action"] == "parse"
    assert page_params["oldid"] == "2002"
    assert "text" in page_params["prop"].split("|")


def test_page_reports_the_redirect_that_was_followed(wiki: FakeWikipedia) -> None:
    redirects = [{"from": "Old Name", "to": "Example Band", "tofragment": "History"}]
    wiki.reply("revisions", revisions(redirects=redirects))
    wiki.reply("page", page(ARTICLE_HTML, SECTIONS))

    header = read(title="Old Name").splitlines()[0]

    assert 'redirected from "Old Name" (target section: "History")' in header
    assert '"Example Band"' in header


def test_a_redirect_without_a_target_section_is_reported_plainly(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions(redirects=[{"from": "Old Name", "to": "Example Band"}]))
    wiki.reply("page", page("<p>Short.</p>", []))

    header = read(title="Old Name").splitlines()[0]

    assert header.endswith('(current) | redirected from "Old Name"]')


def test_page_by_date_reads_the_last_revision_of_that_day(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions(revid=1001, timestamp="2019-06-29T10:11:12Z"))
    wiki.reply("page", page(ARTICLE_HTML, SECTIONS))

    result = read(title="Example Band", as_of="2019-06-30")

    revision_params = wiki.params_of("revisions")
    assert revision_params["rvstart"] == "2019-06-30T23:59:59Z"
    assert revision_params["rvdir"] == "older"
    assert revision_params["rvlimit"] == "1"
    assert wiki.params_of("page")["oldid"] == "1001"
    assert result.splitlines()[0] == (
        '[Wikipedia (en) "Example Band" | revision 1001 of 2019-06-29T10:11:12Z (latest up to 2019-06-30)]'
    )


def test_a_redirect_revision_is_shown_as_such(wiki: FakeWikipedia) -> None:
    redirect_html = (
        '<div class="mw-parser-output"><div class="redirectMsg"><p>Redirect to:</p><ul class="redirectText"><li>'
        '<a href="/wiki/Example_Band" title="Example Band">Example Band</a></li></ul></div></div>'
    )
    wiki.reply("revisions", revisions(title="Old Name", revid=5))
    wiki.reply("page", page(redirect_html, [], title="Old Name"))

    result = read(title="Old Name", as_of="2015-06-01")

    assert result.endswith("Redirect to:\n\n- Example Band")


def test_page_in_another_language_edition(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions(title="Grupo Exemplo"))
    wiki.reply("page", page("<p>Um grupo fictício.</p>", [], title="Grupo Exemplo"))

    result = read(title="Grupo Exemplo", lang="pt")

    assert {seen.host for seen in wiki.seen} == {"pt.wikipedia.org"}
    assert result == (
        '[Wikipedia (pt) "Grupo Exemplo" | revision 2002 of 2026-09-30T08:00:00Z (current)]\nUm grupo fictício.'
    )


def test_none_arguments_mean_not_given(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", page("<p>Short.</p>", []))

    result = read(title="Example Band", lang=None, as_of=None, section=None, query=None)

    assert result.endswith("\nShort.")
    assert "rvstart" not in wiki.params_of("revisions")


def test_arguments_that_are_not_text_are_read_as_text(wiki: FakeWikipedia) -> None:
    wiki.reply("search", search_reply(("2009 in music", "A year.")))
    wiki.reply("revisions", revisions())
    wiki.reply("page", page("<p>Short.</p>", []))

    found = WikipediaSearchTool().forward(2009)  # type: ignore[arg-type]

    assert found.startswith('Wikipedia (en) results for "2009":')
    assert read(title="Example Band", query=1994, as_of=None).endswith("\nShort.")
    assert wiki.params_of("search")["srsearch"] == "2009"


def test_a_page_without_readable_text_says_so(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", page('<div class="mw-parser-output"><div class="navbox">Navigation only</div></div>', []))

    assert read(title="Example Band").endswith("\n(the page has no readable text)")


# --- smolagents interface -------------------------------------------------------------------------


def test_the_tools_work_through_the_smolagents_call_interface(wiki: FakeWikipedia) -> None:
    wiki.reply("search", search_reply(("Example Band", "A group.")))
    wiki.reply("revisions", revisions())
    wiki.reply("page", page("<p>Short.</p>", []))

    assert WikipediaSearchTool()(query="example band").startswith('Wikipedia (en) results for "example band":')
    assert WikipediaPageTool()({"title": "Example Band"}).endswith("\nShort.")


def test_tool_descriptions_stay_short_because_they_are_sent_with_every_prompt() -> None:
    assert len(WikipediaSearchTool.description) < 300
    assert len(WikipediaPageTool.description) < 450
