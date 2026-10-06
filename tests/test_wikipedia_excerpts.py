"""Tests for the pure text helpers behind the Wikipedia page tool: truncation and excerpt selection.

All texts are invented; nothing here comes from a real article or from the benchmark.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from gaia_agent.tools import wikipedia_excerpts
from gaia_agent.tools.wikipedia_excerpts import select_relevant, truncate

ARTICLE = """\
Example Band is a fictional group from Nowhere. They released a few records.

## History

The band formed in 1990 in the city of Exampleton. Early gigs were local.

## Discography

### Studio albums

Table: Studio albums

| Title | Year | Peak position |
| --- | --- | --- |
| Blue Morning | 1992 | 12 |
| Red Evening | 1995 | 4 |
| Green Night | 1999 | 2 |

### Compilation albums

- Best of Example (2004)

## Awards

The band won the Fictional Music Award in 1996.
"""

BIG_LIMIT = 10_000


# --- truncate -----------------------------------------------------------------------------------


def test_truncate_returns_short_text_unchanged() -> None:
    assert truncate("short text", 100) == "short text"


def test_truncate_marks_the_total_length_and_respects_the_limit() -> None:
    text = "lorem ipsum dolor sit amet " * 400

    result = truncate(text, 500)

    assert len(result) <= 500
    assert result.endswith(f"[truncated: {len(text)} chars total]")


def test_truncate_cuts_at_a_line_boundary() -> None:
    text = "\n".join(f"line {number:03d}" for number in range(300))

    result = truncate(text, 400)

    kept = result.splitlines()[:-1]
    assert kept
    assert all(re.fullmatch(r"line \d{3}", line) for line in kept)


def test_truncate_cuts_at_a_word_boundary_when_there_is_no_newline() -> None:
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    text = " ".join(words * 200)

    result = truncate(text, 300)

    kept = result.splitlines()[0].split(" ")
    assert set(kept) <= set(words)


def test_truncate_hard_cuts_a_single_giant_token() -> None:
    result = truncate("x" * 5000, 200)

    assert len(result) <= 200
    assert result.startswith("xxx")
    assert "[truncated: 5000 chars total]" in result


def test_truncate_with_a_tiny_limit_still_reports_the_total() -> None:
    result = truncate("y" * 100, 5)

    assert result == "[truncated: 100 chars total]"


# --- select_relevant ----------------------------------------------------------------------------


def test_select_returns_only_the_matching_passage_with_its_heading() -> None:
    result = select_relevant(ARTICLE, "award 1996", BIG_LIMIT)

    assert "## Awards" in result
    assert "Fictional Music Award" in result
    assert "Exampleton" not in result
    assert "Best of Example" not in result


def test_select_keeps_the_table_header_with_the_matching_row() -> None:
    result = select_relevant(ARTICLE, "Green Night", BIG_LIMIT)

    assert "| Title | Year | Peak position |" in result
    assert "| --- | --- | --- |" in result
    assert "| Green Night | 1999 | 2 |" in result
    assert "Blue Morning" not in result
    assert "Red Evening" not in result


def test_select_prints_the_headings_of_the_section_once() -> None:
    result = select_relevant(ARTICLE, "Blue Morning Red Evening", BIG_LIMIT)

    assert result.count("## Discography") == 1
    assert result.count("### Studio albums") == 1
    assert result.count("| --- | --- | --- |") == 1
    assert result.index("Blue Morning") < result.index("Red Evening")


def test_select_marks_the_gap_between_distant_passages() -> None:
    result = select_relevant(ARTICLE, "Exampleton Fictional Award", BIG_LIMIT)

    assert "[...]" in result
    assert result.index("Exampleton") < result.index("Fictional Music Award")


def test_select_does_not_mark_a_gap_between_adjacent_passages() -> None:
    result = select_relevant(ARTICLE, "Blue Morning Red Evening", BIG_LIMIT)

    assert "[...]" not in result


def test_select_ignores_case_accents_and_plurals() -> None:
    text = "Intro line.\n\n## Life\n\nZoé Müller released one album in 2003.\n\nUnrelated closing remark."

    result = select_relevant(text, "ZOE MULLER ALBUMS", BIG_LIMIT)

    assert "Zoé Müller released one album" in result
    assert "Unrelated" not in result


def test_select_never_exceeds_the_limit() -> None:
    text = "\n\n".join(f"The band played show number {number} in a big hall." for number in range(200))

    result = select_relevant(text, "band show hall", 500)

    assert result
    assert len(result) <= 500


def test_select_prefers_rare_terms_when_the_budget_is_tight() -> None:
    common = [f"The band played gig {number} somewhere nice." for number in range(20)]
    text = "\n\n".join([*common, "A lone saxophone solo closed the night.", *common])

    result = select_relevant(text, "band saxophone", 45)

    assert "saxophone" in result
    assert "gig" not in result


def test_select_returns_document_order_not_score_order() -> None:
    text = "first rare word here.\n\nfiller.\n\nsecond rare word and another rare word here."

    result = select_relevant(text, "rare", BIG_LIMIT)

    assert result.index("first") < result.index("second")


def test_select_returns_nothing_when_no_passage_matches() -> None:
    assert select_relevant(ARTICLE, "zeppelin", BIG_LIMIT) == ""


def test_select_returns_nothing_for_an_empty_or_stopword_only_query() -> None:
    assert select_relevant(ARTICLE, "   ", BIG_LIMIT) == ""
    assert select_relevant(ARTICLE, "the of and", BIG_LIMIT) == ""


def test_select_returns_nothing_when_no_passage_fits_the_limit() -> None:
    assert select_relevant(ARTICLE, "Fictional Award", 5) == ""


def test_select_scores_rows_through_their_section_heading() -> None:
    text = ARTICLE + "\n## Tours\n\nA long tour of the north.\n"

    result = select_relevant(text, "compilation", BIG_LIMIT)

    assert "### Compilation albums" in result
    assert "Best of Example (2004)" in result


def test_select_handles_tables_without_a_header_separator() -> None:
    text = "## Info\n\n| Born | 1970 |\n| Genre | Pop |\n| Label | Acme Records |\n"

    result = select_relevant(text, "label", BIG_LIMIT)

    assert "| Label | Acme Records |" in result
    assert "| Born | 1970 |" not in result


def test_select_prints_the_table_header_again_after_other_content() -> None:
    text = (
        "| H1 | H2 |\n| --- | --- |\n| alpha | one |\n\nnarrative about alpha things\n\n"
        "| H1 | H2 |\n| --- | --- |\n| alpha | two |\n"
    )

    result = select_relevant(text, "alpha", BIG_LIMIT)

    assert result.count("| H1 | H2 |") == 2


def test_select_finds_rows_through_the_words_of_their_table_header() -> None:
    result = select_relevant(ARTICLE, "position", BIG_LIMIT)

    assert "| Title | Year | Peak position |" in result
    assert "| Blue Morning | 1992 | 12 |" in result
    assert "| Green Night | 1999 | 2 |" in result
    assert "Exampleton" not in result


def test_select_matches_plural_and_singular_forms_both_ways() -> None:
    text = "Intro line.\n\nOne album was released in 2003.\n\nThe albums sold well.\n\nUnrelated closing remark."

    both = "One album was released in 2003.\nThe albums sold well."

    assert select_relevant(text, "albums", BIG_LIMIT) == both
    assert select_relevant(text, "album", BIG_LIMIT) == both


def test_select_skips_passages_that_do_not_fit_instead_of_giving_up() -> None:
    giants = [f"giant passage number {number} " + "filler " * 200 for number in range(40)]
    text = "\n\n".join([*giants, "A small giant remark."])

    assert select_relevant(text, "giant", 300) == "A small giant remark."


def test_select_only_prints_the_headings_that_enclose_the_passage() -> None:
    text = "## Alpha\n\nfirst paragraph\n\n## Beta\n\nsecond paragraph\n\n### Child\n\nthird paragraph"

    assert select_relevant(text, "second", BIG_LIMIT) == "## Beta\nsecond paragraph"
    assert select_relevant(text, "third", BIG_LIMIT) == "## Beta\n### Child\nthird paragraph"


def test_select_prints_the_table_caption_above_the_header() -> None:
    result = select_relevant(ARTICLE, "Green Night", BIG_LIMIT)

    assert result.index("Table: Studio albums") < result.index("| Title | Year | Peak position |")
    assert result.count("Table: Studio albums") == 1


def test_select_finds_rows_through_the_table_caption() -> None:
    text = "Table: Medal winners\n| Nation | Gold |\n| --- | --- |\n| Norway | 3 |\n\nUnrelated paragraph.\n"

    result = select_relevant(text, "medal", BIG_LIMIT)

    assert result == "Table: Medal winners\n| Nation | Gold |\n| --- | --- |\n| Norway | 3 |"


def test_select_handles_a_caption_over_a_table_without_a_separator() -> None:
    text = "Table: Facts\n| Born | 1970 |\n| Label | Acme |\n"

    assert select_relevant(text, "label", BIG_LIMIT) == "Table: Facts\n| Label | Acme |"


def test_a_caption_line_without_a_table_is_an_ordinary_line() -> None:
    text = "Table: not really a caption\n\nSome other text."

    assert select_relevant(text, "caption", BIG_LIMIT) == "Table: not really a caption"


def test_select_keeps_single_digit_numbers_in_the_query() -> None:
    text = "Season 1 aired in 2001.\n\nSeason 2 aired in 2002.\n\nSeason 3 aired later."

    assert select_relevant(text, "season 2", len("Season 2 aired in 2002.")) == "Season 2 aired in 2002."


def test_truncate_keeps_the_whole_cut_when_it_ends_exactly_at_a_word_boundary() -> None:
    assert truncate("abcde " * 50, 52) == "abcde abcde abcde abcde\n[truncated: 300 chars total]"


def test_select_prints_a_chosen_heading_only_once() -> None:
    result = select_relevant(ARTICLE, "awards", BIG_LIMIT)

    assert result.count("## Awards") == 1
    assert "The band won the Fictional Music Award in 1996." in result


def test_select_stops_trying_once_the_budget_is_full(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real_render = wikipedia_excerpts._render

    def counting_render(units: Any, chosen: Any) -> str:
        nonlocal calls
        calls += 1
        return real_render(units, chosen)

    monkeypatch.setattr(wikipedia_excerpts, "_render", counting_render)
    text = "\n\n".join(f"match line number {number}" for number in range(2000))

    result = select_relevant(text, "match", 200)

    assert 0 < len(result) <= 200
    assert calls < 400


def test_select_keeps_single_letters_such_as_group_c() -> None:
    text = "Group A won.\n\nGroup B lost.\n\nGroup C drew."

    assert select_relevant(text, "group c", len("Group C drew.")) == "Group C drew."
