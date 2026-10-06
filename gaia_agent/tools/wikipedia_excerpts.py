"""Pure text helpers for the Wikipedia page tool: truncation and query-driven excerpts.

Nothing here touches the network.  ``select_relevant`` reads the markdown that
``wikipedia_html.html_to_markdown`` produces: headings (``## Title``), pipe tables
(``| a | b |`` rows, optionally preceded by a ``| --- |`` header separator) and plain
lines.  Every non-empty line is a *unit*; a unit is scored against the query and the
best ones are printed in document order, each with the headings (and, for table rows,
the table header) that give it context.
"""

from __future__ import annotations

import math
import re
import unicodedata
from bisect import insort
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

GAP_MARKER = "[...]"
TABLE_CAPTION_PREFIX = "Table: "  # a line starting like this, right above a table, is that table's caption
MAX_CONSECUTIVE_MISSES = 25  # stop looking once this many candidates in a row no longer fit
PATH_WEIGHT = 0.5  # a query term found only in a heading above the unit
HEADER_WEIGHT = 0.3  # a query term found only in the header of the unit's table

_WORD = re.compile(r"\w+")
_HEADING = re.compile(r"(#{1,6}) +\S")
_SEPARATOR = re.compile(r"\|(?:\s*:?-{3,}:?\s*\|)+")
_STOPWORDS = frozenset(
    """
    a an and are as at be but by did do does for from had has have how in is it its many much of on or
    than that the their there these this those to was were what when where which who whom whose why will
    with
    """.split()
)


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, marker included.

    The cut falls on a line or word boundary when one is close enough, and the
    result ends with ``[truncated: N chars total]``.  Text that already fits is
    returned unchanged.  A limit too small for the marker returns the marker alone.
    """
    if len(text) <= limit:
        return text
    marker = f"[truncated: {len(text)} chars total]"
    room = limit - len(marker) - 1
    if room <= 0:
        return marker
    cut = text[:room]
    if not text[room].isspace():
        for separator in ("\n", " "):
            position = cut.rfind(separator)
            if position >= room // 2:
                cut = cut[:position]
                break
    return f"{cut.rstrip()}\n{marker}"


def select_relevant(text: str, query: str, limit: int) -> str:
    """Return the passages of markdown ``text`` that best match ``query``, within ``limit`` characters.

    Passages come back in document order, preceded by their headings and table
    headers, with ``[...]`` where text was skipped.  The result is empty when the
    query has no usable term or nothing matches.
    """
    terms = _stems(query)
    if not terms:
        return ""
    units = _split_units(text)
    scores = _scores(units, terms)
    ranked = sorted((i for i, score in enumerate(scores) if score > 0), key=lambda i: (-scores[i], i))
    chosen: list[int] = []
    misses = 0
    for index in (i for i in ranked if len(_render(units, [i])) <= limit):
        trial = [*chosen]
        insort(trial, index)
        if len(_render(units, trial)) <= limit:
            chosen, misses = trial, 0
            continue
        misses += 1
        if misses >= MAX_CONSECUTIVE_MISSES:
            break
    return _render(units, chosen) if chosen else ""


@dataclass(frozen=True)
class _Unit:
    """One line of the text with the context needed to understand it on its own."""

    text: str
    path: tuple[str, ...]  # heading lines above the unit, outermost first
    header: tuple[str, ...]  # caption and header lines of its table, up to the separator (rows only)

    @property
    def context(self) -> tuple[str, ...]:
        """The lines that explain the unit: the headings above it, then the header lines of its table."""
        return self.path + self.header


def _stem(word: str) -> str:
    """Drop a plural ``s`` (``albums`` -> ``album``); short words and ``-ss`` words stay as they are."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _stems(text: str) -> frozenset[str]:
    """Lower-cased, accent-free word stems of ``text`` without stopwords (``c`` in ``Group C`` and ``2`` stay)."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    words = (word for word in _WORD.findall(folded) if word not in _STOPWORDS)
    return frozenset(_stem(word) for word in words)


@lru_cache(maxsize=512)
def _context_stems(lines: tuple[str, ...]) -> frozenset[str]:
    return _stems(" ".join(lines))


def _split_units(text: str) -> list[_Unit]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    units: list[_Unit] = []
    headings: list[tuple[int, str]] = []
    position = 0
    while position < len(lines):
        line = lines[position]
        path = tuple(heading for _, heading in headings)
        match = _HEADING.match(line)
        if match:
            level = len(match.group(1))
            headings = [(lv, heading) for lv, heading in headings if lv < level]
            units.append(_Unit(line, tuple(heading for _, heading in headings), ()))
            headings.append((level, line))
            position += 1
        elif line.startswith("|") or _is_caption(lines, position):
            caption = "" if line.startswith("|") else line
            start = position + 1 if caption else position
            end = start
            while end < len(lines) and lines[end].startswith("|"):
                end += 1
            units.extend(_table_units(lines[start:end], path, caption))
            position = end
        else:
            units.append(_Unit(line, path, ()))
            position += 1
    return units


def _is_caption(lines: list[str], position: int) -> bool:
    following = lines[position + 1] if position + 1 < len(lines) else ""
    return lines[position].startswith(TABLE_CAPTION_PREFIX) and following.startswith("|")


def _table_units(rows: list[str], path: tuple[str, ...], caption: str) -> list[_Unit]:
    """Units for the data rows of one table; its caption and header lines travel with every row."""
    context = (caption,) if caption else ()
    separator = next((i for i, row in enumerate(rows) if _SEPARATOR.fullmatch(row)), None)
    if separator is None:
        return [_Unit(row, path, context) for row in rows]
    header = (*context, *rows[: separator + 1])
    return [_Unit(row, path, header) for row in rows[separator + 1 :]]


def _scores(units: Sequence[_Unit], terms: frozenset[str]) -> list[float]:
    """Sum of inverse document frequencies of the query terms found in each unit or its context."""
    unit_stems = [_stems(unit.text) for unit in units]
    frequency = Counter(stem for stems in unit_stems for stem in stems)
    total = len(units)
    idf = {term: math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5)) for term in terms}
    return [_score(unit, stems, idf) for unit, stems in zip(units, unit_stems, strict=True)]


def _score(unit: _Unit, stems: frozenset[str], idf: dict[str, float]) -> float:
    path_stems = _context_stems(unit.path)
    header_stems = _context_stems(unit.header)
    score = 0.0
    for term, weight in idf.items():
        if term in stems:
            score += weight
        elif term in path_stems:
            score += PATH_WEIGHT * weight
        elif term in header_stems:
            score += HEADER_WEIGHT * weight
    return score


def _render(units: Sequence[_Unit], chosen: Sequence[int]) -> str:
    """Print the chosen units (ascending indexes) with the context lines not already shown."""
    lines: list[str] = []
    shown: tuple[str, ...] = ()
    previous = -2
    for index in chosen:
        unit = units[index]
        if lines and index != previous + 1:
            lines.append(GAP_MARKER)
        lines.extend(unit.context[_shared_prefix(shown, unit.context) :])
        lines.append(unit.text)
        shown = (*unit.path, unit.text) if _HEADING.match(unit.text) else unit.context
        previous = index
    return "\n".join(lines)


def _shared_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    shared = 0
    for first, second in zip(left, right, strict=False):
        if first != second:
            break
        shared += 1
    return shared
