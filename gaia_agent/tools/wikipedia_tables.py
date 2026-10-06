"""Tables of MediaWiki HTML as ``| a | b |`` rows, so that statistics and discographies survive.

Spans are expanded so that every row has the same columns: a ``rowspan`` repeats its text
in the rows below (each data row then stands on its own), a ``colspan`` leaves the extra
columns empty.  Header rows get a ``| --- |`` separator when they really are column
headers.  Pure functions over BeautifulSoup nodes; no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import Tag

from gaia_agent.tools.wikipedia_excerpts import TABLE_CAPTION_PREFIX
from gaia_agent.tools.wikipedia_inline import SEPARATOR, clean, inline_text, is_noise, text_of

MAX_SPAN = 100

_CELL_SEPARATORS = re.compile(rf"\s*{SEPARATOR}[\s{SEPARATOR}]*")


@dataclass(frozen=True)
class _Row:
    cells: tuple[str, ...]  # spans already expanded: one entry per column
    header: bool  # every cell the row defines itself is a <th>


def table_chunk(table: Tag) -> list[str]:
    """The markdown chunk for one ``<table>`` (empty when it has no text): caption line, then the rows."""
    lines = _render_table(_table_rows(table))
    if not lines:
        return []
    caption = table.find("caption", recursive=False)
    title = text_of(caption) if isinstance(caption, Tag) and not is_noise(caption) else ""
    return ["\n".join([f"{TABLE_CAPTION_PREFIX}{title}", *lines] if title else lines)]


def _table_rows(table: Tag) -> list[_Row]:
    carried: dict[int, tuple[int, str]] = {}  # column -> (rows still spanned, text to repeat)
    rows: list[_Row] = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table or is_noise(tr):
            continue
        cells = [cell for cell in tr.find_all(["td", "th"], recursive=False) if not is_noise(cell)]
        rows.append(_build_row(cells, carried))
    return rows


def _build_row(cells: list[Tag], carried: dict[int, tuple[int, str]]) -> _Row:
    """Lay the cells of one row on the grid, honouring spans that started in earlier rows.

    A ``rowspan`` repeats its text in the rows below, so every data row stands on its
    own; header cells are not repeated.  A ``colspan`` puts the text in its first column
    and leaves the others empty.
    """
    placed: dict[int, str] = {}
    for column, (remaining, text) in sorted(carried.items()):
        placed[column] = text
        if remaining > 1:
            carried[column] = (remaining - 1, text)
        else:
            del carried[column]
    is_header = bool(cells) and all(cell.name == "th" for cell in cells)
    column = 0
    for cell in cells:
        while column in placed:
            column += 1
        text = _cell_text(cell)
        colspan, rowspan = _span(cell, "colspan"), _span(cell, "rowspan")
        for offset in range(colspan):
            shown = text if offset == 0 else ""
            placed[column + offset] = shown
            if rowspan > 1:
                carried[column + offset] = (rowspan - 1, "" if is_header else shown)
        column += colspan
    width = max(placed, default=-1) + 1
    return _Row(tuple(placed.get(index, "") for index in range(width)), is_header)


def _span(cell: Tag, attribute: str) -> int:
    value = str(cell.get(attribute) or "").strip()
    return min(int(value), MAX_SPAN) if value.isdecimal() and int(value) > 0 else 1


def _cell_text(cell: Tag) -> str:
    """Cell text on one line; a cell holding only icons falls back to their alt text."""
    return _one_line(inline_text(cell)) or _one_line(inline_text(cell, icons=True))


def _one_line(raw: str) -> str:
    """Line breaks and list items become ``; ``, pipes are escaped."""
    text = clean(_CELL_SEPARATORS.sub("; ", raw)).strip("; ")
    return text.replace("|", "\\|")


def _trimmed(cells: tuple[str, ...]) -> tuple[str, ...]:
    end = len(cells)
    while end and not cells[end - 1]:
        end -= 1
    return cells[:end]


def _pipe_row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(_trimmed(cells)) + " |"


def _header_row_count(rows: list[_Row]) -> int:
    """Leading rows that are column headers (0 for a lone title row or a table with no body)."""
    count = 0
    for row in rows:
        if not row.header:
            break
        count += 1
    if count == len(rows):
        return 0
    has_columns = any(sum(1 for cell in row.cells if cell) >= 2 for row in rows[:count])
    return count if has_columns else 0


def _render_table(rows: list[_Row]) -> list[str]:
    rows = [row for row in rows if any(row.cells)]
    if not rows:
        return []
    width = max(len(_trimmed(row.cells)) for row in rows)
    header_rows = _header_row_count(rows)
    lines: list[str] = []
    for position, row in enumerate(rows, start=1):
        lines.append(_pipe_row(row.cells))
        if position == header_rows:
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return lines
