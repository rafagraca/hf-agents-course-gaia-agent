"""Convert MediaWiki parser HTML into compact markdown that keeps the tables.

The Wikipedia tools hand the model text, not HTML: headings become ``## Title``, lists
become ``- item`` lines and tables become ``| a | b |`` rows (see ``wikipedia_tables``),
so discographies, medal tables and statistics survive.  Citation markers, edit links,
navigation and message boxes, images and hidden sort keys are dropped.  Everything here
is pure (HTML in, text out) and never touches the network.
"""

from __future__ import annotations

import html
import re
import warnings
from collections.abc import Callable

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, NavigableString, Tag
from bs4.element import PageElement

from gaia_agent.tools.wikipedia_inline import clean, inline_children, inline_tag, is_noise, text_of
from gaia_agent.tools.wikipedia_tables import table_chunk

CONTAINER_TAGS = frozenset(
    {"div", "section", "article", "main", "center", "blockquote", "details", "summary", "aside", "header"}
    | {"footer", "nav", "body", "html"}
)

_TAG = re.compile(r"<[^>]*>")
_HEADING_LINE = re.compile(r"(#{1,6}) ")


def html_to_markdown(markup: str) -> str:
    """Convert the HTML of a MediaWiki page (or section) into compact markdown.

    Blocks are separated by blank lines; list and table lines are not.  Headings
    with nothing under them (a ``References`` heading whose list was removed) are dropped.
    """
    return "\n\n".join(_drop_empty_headings(_blocks(_parse(markup))))


def plain_text(fragment: str | None) -> str:
    """Text of an HTML fragment (a search snippet, a section title): tags removed, entities decoded."""
    if not fragment:
        return ""
    return clean(html.unescape(_TAG.sub("", fragment)))


def _parse(markup: str) -> BeautifulSoup:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        return BeautifulSoup(markup, "html.parser")


# --- block structure ----------------------------------------------------------------------------


def _blocks(container: Tag) -> list[str]:
    """Markdown chunks (heading, paragraph, list or table) for the children of ``container``."""
    chunks: list[str] = []
    pending: list[str] = []
    for child in container.children:
        if isinstance(child, Tag):
            if is_noise(child):
                continue
            handler = _BLOCK_HANDLERS.get(child.name)
            if handler is None:
                pending.append(inline_tag(child))
                continue
            chunks.extend(_flush(pending))
            chunks.extend(handler(child))
        elif type(child) is NavigableString:
            pending.append(str(child))
    chunks.extend(_flush(pending))
    return chunks


def _flush(pending: list[str]) -> list[str]:
    """Turn the loose inline text collected so far into at most one chunk and reset it."""
    text = clean("".join(pending))
    pending.clear()
    return [text] if text else []


def _heading(tag: Tag) -> list[str]:
    text = text_of(tag)
    return [f"{'#' * int(tag.name[1])} {text}"] if text else []


def _paragraph(tag: Tag) -> list[str]:
    text = text_of(tag)
    return [text] if text else []


def _preformatted(tag: Tag) -> list[str]:
    text = tag.get_text().strip("\n")
    return [f"```\n{text}\n```"] if text.strip() else []


def _definition_list(tag: Tag) -> list[str]:
    lines: list[str] = []
    for entry in tag.find_all(["dt", "dd"], recursive=False):
        text = "" if is_noise(entry) else text_of(entry)
        if text:
            lines.append(text if entry.name == "dt" else f"  {text}")
    return ["\n".join(lines)] if lines else []


def _skip(_: Tag) -> list[str]:
    return []


def _heading_level(chunk: str) -> int:
    match = _HEADING_LINE.match(chunk)
    return len(match.group(1)) if match else 0


def _has_content(following: str | None, level: int) -> bool:
    """A heading has content when what follows it is not a heading, or is a deeper one."""
    if following is None:
        return False
    following_level = _heading_level(following)
    return following_level == 0 or following_level > level


def _drop_empty_headings(chunks: list[str]) -> list[str]:
    kept: list[str] = []  # built back to front so that a chain of empty headings collapses
    for chunk in reversed(chunks):
        level = _heading_level(chunk)
        if level and not _has_content(kept[-1] if kept else None, level):
            continue
        kept.append(chunk)
    kept.reverse()
    return kept


# --- lists --------------------------------------------------------------------------------------


def _list_chunk(tag: Tag) -> list[str]:
    lines = _list_lines(tag, 0)
    return ["\n".join(lines)] if lines else []


def _list_lines(tag: Tag, depth: int) -> list[str]:
    lines: list[str] = []
    value = str(tag.get("start") or "").strip()
    number = int(value) if value.isdecimal() else 1
    for item in tag.find_all("li", recursive=False):
        if is_noise(item):
            continue
        own, nested = _split_item(item)
        text = clean(inline_children(own))
        if text:
            marker = f"{number}. " if tag.name == "ol" else "- "
            lines.append("  " * depth + marker + text)
            number += 1
        for child_list in nested:
            lines.extend(_list_lines(child_list, depth + 1 if text else depth))
    return lines


def _split_item(item: Tag) -> tuple[list[PageElement], list[Tag]]:
    """Separate the text of a list item from the lists nested inside it."""
    own: list[PageElement] = []
    nested: list[Tag] = []
    for child in item.children:
        if isinstance(child, Tag) and child.name in ("ul", "ol"):
            if not is_noise(child):
                nested.append(child)
        else:
            own.append(child)
    return own, nested


_BLOCK_HANDLERS: dict[str, Callable[[Tag], list[str]]] = {
    "p": _paragraph,
    "ul": _list_chunk,
    "ol": _list_chunk,
    "dl": _definition_list,
    "table": table_chunk,
    "pre": _preformatted,
    "hr": _skip,
    **dict.fromkeys(CONTAINER_TAGS, _blocks),
    **{f"h{level}": _heading for level in range(1, 7)},
}
