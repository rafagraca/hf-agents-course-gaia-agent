"""Tests for the failures of the Wikipedia tools: they come back as ``Error: ...`` messages, never raised.

The MediaWiki API is faked (see ``wikipedia_fakes.py``): nothing here touches the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import requests
from wikipedia_fakes import (
    USER_AGENT,
    FakeWikipedia,
    fake_wikipedia,
    missing_page,
    page,
    read,
    revisions,
    search_reply,
)

from gaia_agent.tools import wikipedia_api
from gaia_agent.tools.wikipedia import WikipediaSearchTool


@pytest.fixture
def wiki() -> Iterator[FakeWikipedia]:
    yield from fake_wikipedia()


# --- arguments refused before any request ----------------------------------------------------------


def test_search_rejects_an_empty_query_without_calling_the_api(wiki: FakeWikipedia) -> None:
    assert WikipediaSearchTool().forward("   ").startswith("Error: ")
    assert WikipediaSearchTool().forward(None).startswith("Error: ")  # type: ignore[arg-type]
    assert wiki.seen == []


@pytest.mark.parametrize("lang", ["en.evil.example/x", "EN US", "e", "../etc", "a" * 40, "en@host"])
def test_invalid_language_codes_never_reach_the_network(wiki: FakeWikipedia, lang: str) -> None:
    expected = f'Error: invalid language code "{lang}" (use e.g. "en", "pt", "simple")'

    assert WikipediaSearchTool().forward("anything", lang=lang) == expected
    assert read(title="Example Band", lang=lang) == expected
    assert wiki.seen == []


@pytest.mark.parametrize(
    "as_of", ["yesterday", "2019/06/30", "2022-13-01", "2022-02-30", "22-1-1", "2022", "2019-06-30T10:00"]
)
def test_a_bad_date_is_refused_before_any_request(wiki: FakeWikipedia, as_of: str) -> None:
    result = read(title="Example Band", as_of=as_of)

    assert result.startswith("Error: as_of must be a date in YYYY-MM-DD format")
    assert wiki.seen == []


def test_an_empty_title_is_refused_before_any_request(wiki: FakeWikipedia) -> None:
    assert read(title="  ") == "Error: title is empty"
    assert wiki.seen == []


def test_a_title_with_a_pipe_is_refused_instead_of_reading_only_its_first_part(wiki: FakeWikipedia) -> None:
    assert read(title="Example Band|Other Band") == 'Error: a title cannot contain "|" (got "Example Band|Other Band")'
    assert wiki.seen == []


# --- pages that cannot be read ---------------------------------------------------------------------


def test_missing_page_suggests_similar_titles(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", missing_page("Exampel Band"))
    wiki.reply("search", search_reply(("Example Band", "x"), ("Example (band)", "y")))

    result = read(title="Exampel Band")

    assert result == (
        'Error: page "Exampel Band" does not exist on en.wikipedia.org. Similar titles: Example Band; Example (band)'
    )
    assert wiki.params_of("search")["srlimit"] == "3"


def test_missing_page_passes_on_the_spelling_suggestion_of_the_search(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", missing_page("Exampel Band"))
    wiki.reply("search", search_reply(suggestion="example band"))

    assert read(title="Exampel Band") == (
        'Error: page "Exampel Band" does not exist on en.wikipedia.org. Did you mean "example band"?'
    )


def test_missing_page_shows_similar_titles_and_the_suggestion(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", missing_page("Exampel Band"))
    wiki.reply("search", search_reply(("Example Band", "x"), suggestion="example band"))

    assert read(title="Exampel Band") == (
        'Error: page "Exampel Band" does not exist on en.wikipedia.org. '
        'Similar titles: Example Band. Did you mean "example band"?'
    )


def test_missing_page_without_similar_titles(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", missing_page("Zzz"))
    wiki.reply("search", search_reply())

    assert read(title="Zzz") == (
        'Error: page "Zzz" does not exist on en.wikipedia.org. Use wikipedia_search to find the exact title.'
    )


def test_missing_page_survives_a_failing_suggestion_lookup(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", missing_page("Zzz"))
    wiki.reply("search", (500, "oops"))

    assert read(title="Zzz").startswith('Error: page "Zzz" does not exist on en.wikipedia.org.')


def test_an_answer_without_a_usable_page_is_treated_as_a_missing_page(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", {"query": {"pages": ["oops"]}})
    wiki.reply("search", search_reply())

    assert read(title="Example Band").startswith('Error: page "Example Band" does not exist on en.wikipedia.org.')


def test_invalid_title(wiki: FakeWikipedia) -> None:
    body = {"query": {"pages": [{"title": "Foo [bar]", "invalid": True, "invalidreason": 'Bad character "[".'}]}}
    wiki.reply("revisions", body)

    assert read(title="Foo [bar]") == 'Error: invalid title "Foo [bar]": Bad character "[".'


def test_page_that_did_not_exist_yet_on_the_date(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", {"query": {"pages": [{"pageid": 7, "ns": 0, "title": "Example Band"}]}})

    assert read(title="Example Band", as_of="1990-01-01") == (
        'Error: "Example Band" has no revision on or before 1990-01-01 (the page may have been created later).'
    )


# --- the network and the API misbehave --------------------------------------------------------------


def test_search_reports_network_failures(wiki: FakeWikipedia) -> None:
    wiki.reply("search", requests.ConnectionError("boom"))

    assert WikipediaSearchTool().forward("example") == "Error: could not reach en.wikipedia.org (ConnectionError)"


def test_search_reports_api_errors(wiki: FakeWikipedia) -> None:
    wiki.reply("search", {"error": {"code": "readapidenied", "info": "You need read permission."}})

    assert (
        WikipediaSearchTool().forward("example")
        == "Error: Wikipedia API error (readapidenied): You need read permission."
    )


def test_network_failures_are_reported_not_raised(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", requests.ConnectTimeout("slow"))

    assert read(title="Example Band") == "Error: could not reach en.wikipedia.org (ConnectTimeout)"


def test_http_errors_are_reported(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", (503, "unavailable"))

    assert read(title="Example Band") == "Error: could not reach en.wikipedia.org (HTTPError)"


def test_answers_that_are_not_json_are_reported(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", (200, "<html>maintenance</html>"))

    assert read(title="Example Band") == "Error: en.wikipedia.org did not answer with a JSON object"


def test_unexpected_json_shapes_are_reported(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", ["not", "a", "dict"])

    assert read(title="Example Band") == "Error: en.wikipedia.org did not answer with a JSON object"


def test_api_errors_are_reported(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", {"error": {"code": "nosuchrevid", "info": "There is no revision with ID 2002."}})

    assert read(title="Example Band") == (
        "Error: Wikipedia API error (nosuchrevid): There is no revision with ID 2002."
    )


def test_a_parse_answer_without_text_is_reported(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", {"parse": {"title": "Example Band"}})

    assert read(title="Example Band") == 'Error: the parse answer for revision 2002 of "Example Band" has no text'


@pytest.mark.parametrize(
    "pages",
    [
        [{"title": "Example Band", "revisions": [{}]}],
        [{"title": "Example Band", "revisions": [{"revid": "abc", "timestamp": "2026-01-01T00:00:00Z"}]}],
        [{"title": "Example Band", "revisions": ["not a dict"]}],
    ],
)
def test_a_malformed_revision_answer_is_reported_not_raised(wiki: FakeWikipedia, pages: list[dict[str, Any]]) -> None:
    wiki.reply("revisions", {"query": {"pages": pages}})

    assert read(title="Example Band") == "Error: unexpected revision data from the en.wikipedia.org API"


def test_markup_nested_too_deeply_is_reported_not_raised(wiki: FakeWikipedia) -> None:
    deep = f"<p>{'<span>' * 2000}deep text{'</span>' * 2000}</p>"
    wiki.reply("revisions", revisions())
    wiki.reply("page", page(deep, []))

    assert read(title="Example Band") == "Error: the page markup is nested too deeply to convert"


def test_the_http_session_retries_transient_server_errors_and_never_hangs() -> None:
    session = wikipedia_api._session()
    retries = session.get_adapter("https://en.wikipedia.org/w/api.php").max_retries

    assert retries.total == 2
    assert {429, 503} <= set(retries.status_forcelist)
    assert retries.raise_on_status is False
    assert session.headers["User-Agent"] == USER_AGENT
    assert all(0 < seconds <= 60 for seconds in wikipedia_api.TIMEOUT)
