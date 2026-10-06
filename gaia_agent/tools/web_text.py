"""Pure text helpers for the web tools: truncation and query-focused excerpts.

An agent pays for every character a tool returns, so pages are never returned
whole. Without a query the tool shows the start of the page; with a query it
shows only the passages that mention the query's words. Both end with a
``[truncated: N chars total]`` note whenever text was left out.

Everything here is a pure function of its arguments.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

DEFAULT_PREVIEW_CHARS = 5000
QUERY_BUDGET_CHARS = 4000
CHUNK_CHARS = 800
CHUNK_OVERLAP_CHARS = 150
MIN_TERM_LENGTH = 3  # query words of one or two letters are ignored (numbers are kept)
PASSAGE_SEPARATOR = "\n[...]\n"

_BM25_K1 = 1.2  # how fast repeating a word stops adding to a chunk's score
_WORD = re.compile(r"\w+")
_WHITESPACE = re.compile(r"\s")
# Accents and other combining marks, which accent folding removes after NFKD decomposition.
_COMBINING_MARKS = re.compile("[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")

Span = tuple[int, int]  # [start, end) offsets into the text


def truncation_note(total_chars: int) -> str:
    """The marker appended to any output that leaves text out."""
    return f"[truncated: {total_chars} chars total]"


def shorten(text: str, limit: int) -> str:
    """Return ``text`` unchanged, or cut to at most ``limit`` characters ending in ``...``."""
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _break_point(text: str, low: int, high: int) -> int:
    """Index just after the last newline (else the last space) in ``text[low:high]``.

    Returns ``high`` when the window holds no whitespace, so callers always
    get a usable cut position.
    """
    for separator in ("\n", " "):
        found = text.rfind(separator, low, high)
        if found != -1:
            return found + 1
    return high


def truncate_with_note(text: str, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    """Keep the first ``limit`` characters of ``text``, cut at a line or word boundary.

    Text that already fits is returned untouched; otherwise the result ends
    with a ``[truncated: N chars total]`` line.
    """
    if len(text) <= limit:
        return text
    cut = _break_point(text, max(limit - limit // 5, 0), limit)
    return f"{text[:cut].rstrip()}\n{truncation_note(len(text))}"


def fold_text(text: str) -> str:
    """Lowercase ``text`` and strip accents, so that matching ignores case and diacritics."""
    folded = text.casefold()
    if folded.isascii():
        return folded
    return _COMBINING_MARKS.sub("", unicodedata.normalize("NFKD", folded))


def query_terms(query: str) -> list[str]:
    """The distinct searchable words of ``query``: folded, without words of 1-2 letters.

    Numbers are always kept, whatever their length: in a question, "2" or "10" is usually
    the point, and a number that appears everywhere weighs little in the scoring anyway.
    """
    words = _WORD.findall(fold_text(query))
    return list(dict.fromkeys(word for word in words if len(word) >= MIN_TERM_LENGTH or word.isdigit()))


def _word_start(text: str, position: int) -> int:
    """Move ``position`` forward to the start of the next word (it stays when already at one)."""
    if position >= len(text) or text[position - 1].isspace():
        return position
    match = _WHITESPACE.search(text, position)
    return match.end() if match else position


def split_into_chunks(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[Span]:
    """Split ``text`` into overlapping chunks of at most ``chunk_chars`` characters.

    Chunks end at a line or word boundary when one is close and start at a word
    boundary; each one repeats the last ``overlap`` characters of the previous
    one, so a sentence cut by a boundary still appears whole in some chunk.
    Returns ``(start, end)`` offsets that always cover the whole text.
    """
    if chunk_chars <= 0 or not 0 <= overlap < chunk_chars:
        raise ValueError("chunk_chars must be positive and overlap must be smaller than chunk_chars")
    spans: list[Span] = []
    start, size = 0, len(text)
    while start < size:
        hard_end = min(start + chunk_chars, size)
        low = max(start + 1, hard_end - chunk_chars // 4)
        end = hard_end if hard_end == size else _break_point(text, low, hard_end)
        spans.append((start, end))
        if end >= size:
            break
        start = min(_word_start(text, max(end - overlap, start + 1)), end)
    return spans


def _score_spans(text: str, spans: list[Span], terms: list[str]) -> dict[int, float]:
    """Score every chunk that contains at least one query word (others are left out).

    A simplified BM25: each distinct query word adds its inverse document
    frequency across the chunks, so rare words (names, numbers) count for much
    more than words that appear in every chunk, and repeating a word saturates.
    """
    counts: dict[int, Counter[str]] = {}
    for index, (start, end) in enumerate(spans):
        folded = fold_text(text[start:end])
        if not any(term in folded for term in terms):  # cheap test before tokenising
            continue
        tokens = Counter(_WORD.findall(folded))
        if any(term in tokens for term in terms):
            counts[index] = tokens
    scores: dict[int, float] = {}
    for term in terms:
        containing = sum(1 for tokens in counts.values() if term in tokens)
        if not containing:
            continue
        idf = math.log(1 + (len(spans) - containing + 0.5) / (containing + 0.5))
        for index, tokens in counts.items():
            frequency = tokens.get(term, 0)
            if frequency:
                scores[index] = scores.get(index, 0.0) + idf * frequency / (frequency + _BM25_K1)
    return scores


def _merge_spans(spans: list[Span]) -> list[Span]:
    """Union of ``spans``: overlapping or touching spans become one, in text order."""
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _rendered_length(spans: list[Span]) -> int:
    """Characters the passages take once joined (an upper bound: passages are stripped)."""
    return sum(end - start for start, end in spans) + len(PASSAGE_SEPARATOR) * max(len(spans) - 1, 0)


def select_relevant_chunks(
    text: str,
    query: str,
    max_chars: int = QUERY_BUDGET_CHARS,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> str:
    """Return the passages of ``text`` that best match ``query``, in text order.

    The text is split into overlapping chunks, each chunk is scored on the
    query's words (case and accents ignored, words of 1-2 letters skipped but
    numbers kept, rare words weighing more than common ones) and the best chunks
    are kept until ``max_chars`` is reached. Chunks that overlap are merged, so no
    text is repeated, and passages that are not adjacent are joined with ``[...]``.
    Returns ``""`` when no chunk contains any query word.
    """
    terms = query_terms(query)
    if not terms or not text.strip() or max_chars <= 0:
        return ""
    chunk_chars = min(chunk_chars, max_chars)
    overlap = min(overlap, chunk_chars // 2)
    spans = split_into_chunks(text, chunk_chars, overlap)
    scores = _score_spans(text, spans, terms)
    chosen: list[Span] = []
    for index in sorted(scores, key=lambda i: (-scores[i], i)):
        merged = _merge_spans([*chosen, spans[index]])
        if _rendered_length(merged) <= max_chars:
            chosen = merged
    passages = (text[start:end].strip() for start, end in chosen)
    return PASSAGE_SEPARATOR.join(passage for passage in passages if passage)


def build_excerpt(text: str, query: str = "") -> str:
    """The text an agent gets from a page: its start, or the passages matching ``query``.

    Whenever part of the text is left out the result ends with
    ``[truncated: N chars total]``; when the query cannot be used, a bracketed
    first line says why and the start of the page is shown instead.
    """
    if not query.strip():
        return truncate_with_note(text, DEFAULT_PREVIEW_CHARS)
    if not query_terms(query):
        reason = "the query has no searchable words (words of 1-2 letters are ignored)"
        return f"[{reason}; showing the start of the page]\n{truncate_with_note(text)}"
    if len(text) <= QUERY_BUDGET_CHARS:
        return text
    selected = select_relevant_chunks(text, query)
    if not selected:
        reason = "no passage matched the query"
        return f"[{reason}; showing the start of the page]\n{truncate_with_note(text)}"
    return f"{selected}\n{truncation_note(len(text))}"
