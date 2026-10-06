"""Tests for ``wikipedia_page``: sections, titles with a fragment, and the cap on the output size.

The MediaWiki API is faked (see ``wikipedia_fakes.py``): nothing here touches the network.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from wikipedia_fakes import (
    ARTICLE_HTML,
    ARTICLE_MARKDOWN,
    MAX_OUTPUT_CHARS,
    SECTIONS,
    STUDIO_ALBUMS_HTML,
    FakeWikipedia,
    fake_wikipedia,
    long_page,
    page,
    read,
    revisions,
    sections_reply,
)


@pytest.fixture
def wiki() -> Iterator[FakeWikipedia]:
    yield from fake_wikipedia()


# --- sections ------------------------------------------------------------------------------------


def test_section_reads_only_that_section(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply(SECTIONS))
    wiki.reply("section_text", page(STUDIO_ALBUMS_HTML))

    result = read(title="Example Band", section="studio ALBUMS")

    assert wiki.params_of("sections")["oldid"] == "2002"
    assert "sections" in wiki.params_of("sections")["prop"].split("|")
    section_params = wiki.params_of("section_text")
    assert section_params["section"] == "3"
    assert section_params["oldid"] == "2002"
    assert result == (
        '[Wikipedia (en) "Example Band" | revision 2002 of 2026-09-30T08:00:00Z (current) | section "Studio albums"]\n'
        "### Studio albums\n\n| Title | Year |\n| --- | --- |\n| Blue Morning | 1992 |"
    )
    assert wiki.requests_of("page") == []


def test_section_matching_prefers_exact_titles_then_prefixes_then_substrings(wiki: FakeWikipedia) -> None:
    sections = [
        {"line": "Early history", "index": "1", "number": "1"},
        {"line": "History of the name", "index": "2", "number": "2"},
        {"line": "History", "index": "3", "number": "3"},
    ]
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply(sections))
    wiki.reply("section_text", page("<p>x</p>"))

    read(title="Example Band", section="history")
    assert wiki.params_of("section_text")["section"] == "3"


@pytest.mark.parametrize(("wanted", "index"), [("history of", "2"), ("early", "1"), ("== Name ==", "2")])
def test_section_matching_falls_back_to_prefix_and_substring(wiki: FakeWikipedia, wanted: str, index: str) -> None:
    sections = [{"line": "Early history", "index": "1"}, {"line": "History of the name", "index": "2"}]
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply(sections))
    wiki.reply("section_text", page("<p>x</p>"))

    read(title="Example Band", section=wanted)

    assert wiki.params_of("section_text")["section"] == index


def test_section_lookup_understands_the_newer_tocdata_answer(wiki: FakeWikipedia) -> None:
    tocdata = {"parse": {"title": "Example Band", "tocdata": {"sections": [{"line": "History", "index": "1"}]}}}
    wiki.reply("revisions", revisions())
    wiki.reply("sections", tocdata)
    wiki.reply("section_text", page("<p>x</p>"))

    read(title="Example Band", section="History")

    assert wiki.params_of("section_text")["section"] == "1"
    assert "tocdata" in wiki.params_of("sections")["prop"].split("|")


def test_missing_section_lists_the_sections_that_exist(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply(SECTIONS))

    result = read(title="Example Band", section="Awards")

    assert result == (
        'Error: section "Awards" not found in "Example Band". Sections: History | Discography | Studio albums'
    )
    assert wiki.requests_of("section_text") == []


def test_section_of_a_page_without_sections(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply([]))

    assert read(title="Example Band", section="History") == (
        'Error: section "History" not found in "Example Band" (the page has no sections at that revision).'
    )


def test_a_section_made_only_of_markup_characters_matches_nothing(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply(SECTIONS))

    result = read(title="Example Band", section=" == ## ")

    assert result.startswith('Error: section "== ##" not found in "Example Band". Sections: History')
    assert wiki.requests_of("section_text") == []


def test_a_long_list_of_sections_is_cut_with_an_ellipsis(wiki: FakeWikipedia) -> None:
    many = [{"line": f"Section number {number} of the article", "index": str(number)} for number in range(1, 60)]
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply(many))

    result = read(title="Example Band", section="Awards")

    sections_line = result.partition(". Sections: ")[2]
    assert sections_line.startswith("Section number 1 of the article | Section number 2 of the article")
    assert sections_line.endswith(" | ...")
    assert len(sections_line) < 520


def test_section_and_date_combine(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions(revid=1001, timestamp="2019-06-29T10:11:12Z"))
    wiki.reply("sections", sections_reply(SECTIONS))
    wiki.reply("section_text", page(STUDIO_ALBUMS_HTML))

    result = read(title="Example Band", as_of="2019-06-30", section="Discography")

    assert wiki.params_of("section_text")["oldid"] == "1001"
    assert 'latest up to 2019-06-30) | section "Discography"]' in result


def test_a_title_with_a_fragment_reads_that_section(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply(SECTIONS))
    wiki.reply("section_text", page(STUDIO_ALBUMS_HTML))

    result = read(title="Example Band#Studio albums")

    assert wiki.params_of("revisions")["titles"] == "Example Band"
    assert wiki.params_of("section_text")["section"] == "3"
    assert 'section "Studio albums"]' in result


def test_the_section_argument_wins_over_a_title_fragment(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("sections", sections_reply(SECTIONS))
    wiki.reply("section_text", page("<p>x</p>"))

    read(title="Example Band#Discography", section="History")

    assert wiki.params_of("section_text")["section"] == "1"


# --- output size ---------------------------------------------------------------------------------


def test_long_pages_are_truncated_with_a_marker_and_a_section_hint(wiki: FakeWikipedia) -> None:
    html = long_page()
    wiki.reply("revisions", revisions())
    wiki.reply("page", page(html, SECTIONS))

    result = read(title="Example Band")

    assert len(result) <= MAX_OUTPUT_CHARS
    assert re.search(r"\[truncated: \d+ chars total\]", result)
    assert result.splitlines()[-1] == "Sections (use the section argument): History | Discography | Studio albums"
    assert "Lead sentence." in result
    assert "Paragraph 299" not in result


def test_a_query_selects_the_relevant_passages_of_a_long_page(wiki: FakeWikipedia) -> None:
    extra = "<p>The saxophone player Quentin Blake joined in 1994.</p>"
    wiki.reply("revisions", revisions())
    wiki.reply("page", page(long_page(extra), SECTIONS))

    result = read(title="Example Band", query="saxophone 1994")

    assert len(result) <= MAX_OUTPUT_CHARS
    assert "The saxophone player Quentin Blake joined in 1994." in result
    assert "## History" in result
    assert "Paragraph 12:" not in result
    assert re.search(r'\[excerpts relevant to "saxophone 1994"; page has \d+ chars\]', result)
    assert result.splitlines()[-1].startswith("Sections (use the section argument):")


def test_a_query_that_matches_nothing_falls_back_to_the_start_of_the_page(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", page(long_page(), SECTIONS))

    result = read(title="Example Band", query="zeppelin")

    assert len(result) <= MAX_OUTPUT_CHARS
    assert "Lead sentence." in result
    assert '[no passage matches "zeppelin"]' in result
    assert re.search(r"\[truncated: \d+ chars total\]", result)


def test_a_query_is_ignored_when_the_whole_page_fits(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", page(ARTICLE_HTML, SECTIONS))

    result = read(title="Example Band", query="Blue Morning")

    assert result.partition("\n")[2] == ARTICLE_MARKDOWN


def test_a_very_long_query_is_cut_before_it_is_echoed(wiki: FakeWikipedia) -> None:
    wiki.reply("revisions", revisions())
    wiki.reply("page", page(long_page(), SECTIONS))

    result = read(title="Example Band", query="saxophone " * 100)

    echoed = re.search(r'\[(?:excerpts relevant to|no passage matches) "([^"]*)"', result)
    assert echoed is not None
    assert len(echoed.group(1)) <= 200
    assert len(result) <= MAX_OUTPUT_CHARS
