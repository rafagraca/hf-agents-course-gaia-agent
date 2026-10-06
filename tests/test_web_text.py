"""Tests for the pure text helpers behind the web tools: truncation and query-focused excerpts."""

from __future__ import annotations

import pytest

from gaia_agent.tools.web_text import (
    DEFAULT_PREVIEW_CHARS,
    QUERY_BUDGET_CHARS,
    build_excerpt,
    query_terms,
    select_relevant_chunks,
    shorten,
    split_into_chunks,
    truncate_with_note,
    truncation_note,
)

FILLER = "The committee reviewed routine budget items and adjourned without further discussion. "


def filler_text(paragraphs: int, inserts: dict[int, str] | None = None) -> str:
    """Numbered filler paragraphs (about 280 characters each), with optional extra sentences."""
    extra = inserts or {}
    return "\n\n".join(f"Section {index}. {FILLER * 3}{extra.get(index, '')}" for index in range(paragraphs))


# --- truncation -------------------------------------------------------------------------------


def test_truncation_note_names_the_total_length() -> None:
    assert truncation_note(42) == "[truncated: 42 chars total]"


def test_short_text_is_not_truncated() -> None:
    assert truncate_with_note("short text", limit=100) == "short text"


def test_long_text_is_cut_and_the_note_gives_the_total_length() -> None:
    text = "word " * 4000

    result = truncate_with_note(text, limit=5000)

    body, _, note = result.rpartition("\n")
    assert note == "[truncated: 20000 chars total]"
    assert 0 < len(body) <= 5000
    assert text.startswith(body)


def test_truncation_prefers_a_word_boundary() -> None:
    text = "abcdefghij " * 1000

    body = truncate_with_note(text, limit=5000).rpartition("\n")[0]

    assert body.endswith("abcdefghij")


def test_truncation_prefers_a_line_boundary_over_a_word_boundary() -> None:
    text = ("alpha beta gamma delta epsilon\n" * 400)[:-1]

    body = truncate_with_note(text, limit=5000).rpartition("\n")[0]

    assert body.endswith("epsilon")


def test_truncation_hard_cuts_text_without_whitespace() -> None:
    body = truncate_with_note("x" * 9000, limit=5000).rpartition("\n")[0]

    assert body == "x" * 5000


def test_default_limit_is_the_preview_size() -> None:
    result = truncate_with_note("a " * 10_000)

    assert len(result.rpartition("\n")[0]) <= DEFAULT_PREVIEW_CHARS


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("short", 10, "short"),
        ("exactly ten", 11, "exactly ten"),
        ("a longer sentence here", 12, "a longer..."),
        ("x" * 50, 10, "xxxxxxx..."),
    ],
)
def test_shorten_never_exceeds_the_limit(text: str, limit: int, expected: str) -> None:
    result = shorten(text, limit)

    assert result == expected
    assert len(result) <= limit


# --- query terms ------------------------------------------------------------------------------


def test_query_terms_are_lowercase_and_skip_words_of_one_or_two_letters() -> None:
    assert query_terms("The Quick ON a Fox") == ["the", "quick", "fox"]


def test_query_terms_keep_numbers_and_drop_punctuation() -> None:
    assert query_terms("Albums, 1980-1989!") == ["albums", "1980", "1989"]


def test_query_terms_keep_numbers_of_any_length() -> None:
    assert query_terms("top 10 albums of 2019 in 5 days") == ["top", "10", "albums", "2019", "5", "days"]


def test_query_terms_are_unique_and_keep_their_order() -> None:
    assert query_terms("fox dog Fox cat dog") == ["fox", "dog", "cat"]


def test_query_terms_ignore_accents() -> None:
    assert query_terms("Zoë Müller") == ["zoe", "muller"]


@pytest.mark.parametrize("query", ["", "   ", "a of in", "!!! ??"])
def test_queries_without_searchable_words_have_no_terms(query: str) -> None:
    assert query_terms(query) == []


# --- chunking ---------------------------------------------------------------------------------


def test_short_text_is_a_single_chunk() -> None:
    assert split_into_chunks("just a few words", chunk_chars=800, overlap=150) == [(0, 16)]


def test_empty_text_has_no_chunks() -> None:
    assert split_into_chunks("") == []


def test_chunks_cover_the_text_and_overlap_without_gaps() -> None:
    text = filler_text(30)

    spans = split_into_chunks(text, chunk_chars=800, overlap=150)

    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(end - start <= 800 for start, end in spans)
    for (_, previous_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert next_start <= previous_end  # the next chunk starts inside the previous one


def test_chunks_start_at_word_boundaries() -> None:
    text = filler_text(10)

    spans = split_into_chunks(text, chunk_chars=300, overlap=60)

    assert all(start == 0 or text[start - 1].isspace() for start, _ in spans)


def test_chunking_makes_progress_on_text_without_whitespace() -> None:
    spans = split_into_chunks("x" * 5000, chunk_chars=800, overlap=150)

    assert spans[0] == (0, 800)
    assert spans[-1][1] == 5000
    assert len(spans) < 20


@pytest.mark.parametrize(("chunk_chars", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_chunk_sizes_are_rejected(chunk_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_into_chunks("some text", chunk_chars=chunk_chars, overlap=overlap)


# --- selecting relevant chunks ----------------------------------------------------------------


def test_selection_returns_the_passage_that_mentions_the_query_words() -> None:
    needle = " The quokka population on Rottnest Island reached 12000 individuals."
    text = filler_text(40, {25: needle})

    result = select_relevant_chunks(text, "quokka population Rottnest")

    assert "quokka population on Rottnest Island reached 12000" in result
    assert len(result) <= QUERY_BUDGET_CHARS
    assert "Section 3." not in result


def test_selected_passages_keep_the_order_of_the_text() -> None:
    # The later passage repeats the query word, so it scores higher than the earlier one.
    later = " Second sentinel: beta. Sentinel again, sentinel again, sentinel again."
    text = filler_text(40, {5: " First sentinel: alpha.", 30: later})

    result = select_relevant_chunks(text, "sentinel")

    assert result.index("alpha") < result.index("beta")


def test_distant_passages_are_separated_by_a_marker() -> None:
    text = filler_text(40, {5: " Sentinel one.", 35: " Sentinel two."})

    result = select_relevant_chunks(text, "sentinel")

    assert "\n[...]\n" in result


def test_selection_stays_within_the_character_budget() -> None:
    text = filler_text(60)  # every paragraph matches "committee"

    result = select_relevant_chunks(text, "committee", max_chars=2000)

    assert 0 < len(result) <= 2000


def test_nothing_is_selected_when_no_chunk_matches() -> None:
    assert select_relevant_chunks(filler_text(30), "zeppelin") == ""


def test_selection_ignores_case_and_accents() -> None:
    text = filler_text(30, {12: " Notes on the CAFÉ Münster archive."})

    result = select_relevant_chunks(text, "cafe MUNSTER")

    assert "CAFÉ Münster archive" in result


def test_rare_query_words_outweigh_common_ones() -> None:
    crowded = "the " * 150  # many hits for a common word
    needle = "A single platypus was seen near the creek."
    text = "\n\n".join([crowded] + [f"Section {i}. {FILLER * 3}" for i in range(12)] + [needle])

    result = select_relevant_chunks(text, "the platypus", max_chars=800)

    assert "platypus" in result
    assert "the the the" not in result


def test_overlapping_chunks_are_merged_without_repeated_text() -> None:
    text = " ".join(f"word{i:03d}" for i in range(300))

    result = select_relevant_chunks(
        text, "word100 word101 word102 word103", max_chars=10_000, chunk_chars=100, overlap=40
    )

    tokens = [token for token in result.split() if token != "[...]"]
    assert "word100" in tokens
    assert len(tokens) == len(set(tokens))


def test_the_budget_may_be_smaller_than_one_chunk() -> None:
    text = filler_text(20, {7: " Needle phrase about okapis."})

    result = select_relevant_chunks(text, "okapis", max_chars=300)

    assert 0 < len(result) <= 300


@pytest.mark.parametrize("query", ["", "of a", "   "])
def test_selection_without_searchable_words_returns_nothing(query: str) -> None:
    assert select_relevant_chunks(filler_text(10), query) == ""


def test_selection_from_empty_text_returns_nothing() -> None:
    assert select_relevant_chunks("", "anything") == ""


# --- the excerpt an agent sees ----------------------------------------------------------------


def test_short_text_without_query_is_returned_whole() -> None:
    assert build_excerpt("a short page") == "a short page"


def test_long_text_without_query_shows_the_start_and_the_total_length() -> None:
    text = filler_text(60)

    excerpt = build_excerpt(text)

    assert excerpt.startswith("Section 0.")
    assert excerpt.endswith(truncation_note(len(text)))
    assert len(excerpt) <= DEFAULT_PREVIEW_CHARS + 60


def test_a_query_selects_the_matching_passages_and_reports_the_total_length() -> None:
    text = filler_text(60, {44: " The marmot colony counted 321 burrows."})

    excerpt = build_excerpt(text, "marmot burrows")

    assert "marmot colony counted 321 burrows" in excerpt
    assert excerpt.endswith(truncation_note(len(text)))
    assert len(excerpt) <= QUERY_BUDGET_CHARS + 60


def test_a_query_that_matches_nothing_says_so_and_shows_the_start() -> None:
    text = filler_text(60)

    excerpt = build_excerpt(text, "zeppelin")

    assert excerpt.startswith("[no passage matched the query")
    assert "Section 0." in excerpt
    assert excerpt.endswith(truncation_note(len(text)))


def test_a_query_made_of_tiny_words_is_explained() -> None:
    excerpt = build_excerpt(filler_text(60), "of an")

    assert excerpt.startswith("[the query has no searchable words")
    assert "Section 0." in excerpt


def test_text_that_fits_the_query_budget_is_returned_whole_for_a_query() -> None:
    text = "A page of moderate length. " * 40

    assert build_excerpt(text, "moderate") == text


def test_blank_query_behaves_like_no_query() -> None:
    text = filler_text(60)

    assert build_excerpt(text, "   ") == build_excerpt(text)
