"""Wikipedia tools: search for articles and read them, tables included.

Both tools use only the MediaWiki API (see ``wikipedia_api``).  Reading follows redirects,
can pick the revision that was current on a given day, can return a single section and
can return only the passages that match a query.  The output is capped at
``MAX_OUTPUT_CHARS``.

Problems the model can act on (missing page, bad date, network failure) are returned as
short ``Error: ...`` messages instead of being raised, so the agent sees them and retries.
"""

from __future__ import annotations

import operator
from typing import Any

from smolagents import Tool

from gaia_agent.tools.wikipedia_api import (
    Revision,
    SearchResult,
    WikipediaError,
    as_text,
    cutoff_for,
    find_revision,
    normalize_language,
    parse_revision,
    search,
)
from gaia_agent.tools.wikipedia_excerpts import select_relevant, truncate
from gaia_agent.tools.wikipedia_html import html_to_markdown, plain_text

MAX_OUTPUT_CHARS = 6000
SEARCH_LIMIT = 8
SNIPPET_CHARS = 200
SECTION_HINT_CHARS = 500
MAX_QUERY_CHARS = 200
EMPTY_PAGE_NOTE = "(the page has no readable text)"


# --- search results -----------------------------------------------------------------------------


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _format_search(lang: str, query: str, result: SearchResult) -> str:
    if result.hits:
        lines = [f'Wikipedia ({lang}) results for "{query}":']
        for number, hit in enumerate(result.hits, start=1):
            snippet = f" - {_shorten(hit.snippet, SNIPPET_CHARS)}" if hit.snippet else ""
            lines.append(f"{number}. {hit.title}{snippet}")
    else:
        lines = [f'No Wikipedia ({lang}) results for "{query}".']
    if result.suggestion:
        lines.append(f'Did you mean: "{result.suggestion}"?')
    return "\n".join(lines)


# --- sections -----------------------------------------------------------------------------------


def _sections_of(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Sections that can be read on their own (numeric index), from the old or the new API shape."""
    raw = parsed.get("sections") or (parsed.get("tocdata") or {}).get("sections") or []
    return [section for section in raw if isinstance(section, dict) and str(section.get("index", "")).isdecimal()]


def _fold(text: str) -> str:
    return " ".join(text.replace("_", " ").casefold().split())


def _find_section(sections: list[dict[str, Any]], wanted: str) -> dict[str, Any] | None:
    """The section called ``wanted``: an exact title first, then a title starting with it, then containing it."""
    target = _fold(wanted.strip(" =#"))
    if not target:
        return None
    named = [(section, _fold(plain_text(str(section.get("line", ""))))) for section in sections]
    for matches in (operator.eq, str.startswith, operator.contains):
        for section, name in named:
            if matches(name, target):
                return section
    return None


def _titles_line(sections: list[dict[str, Any]], limit: int) -> str:
    line = ""
    for section in sections:
        title = plain_text(str(section.get("line", "")))
        candidate = f"{line} | {title}" if line else title
        if len(candidate) > limit:
            return f"{line} | ..."
        line = candidate
    return line


def _select_section(lang: str, revision: Revision, wanted: str) -> dict[str, Any]:
    sections = _sections_of(parse_revision(lang, revision, "sections|tocdata"))
    match = _find_section(sections, wanted)
    if match is not None:
        return match
    where = f'section "{wanted}" not found in "{revision.title}"'
    if not sections:
        raise WikipediaError(f"{where} (the page has no sections at that revision).")
    raise WikipediaError(f"{where}. Sections: {_titles_line(sections, SECTION_HINT_CHARS)}")


# --- output -------------------------------------------------------------------------------------


def _markdown_of(parsed: dict[str, Any], revision: Revision) -> str:
    """The page text of a parse answer as markdown (empty when nothing readable is left)."""
    html = parsed.get("text")
    if not isinstance(html, str):
        raise WikipediaError(f'the parse answer for revision {revision.revid} of "{revision.title}" has no text')
    try:
        return html_to_markdown(html)
    except RecursionError as error:  # hostile or broken markup nested hundreds of levels deep
        raise WikipediaError("the page markup is nested too deeply to convert") from error


def _header(lang: str, revision: Revision, cutoff: str, section_label: str) -> str:
    when = f"latest up to {cutoff[:10]}" if cutoff else "current"
    parts = [f'Wikipedia ({lang}) "{revision.title}"', f"revision {revision.revid} of {revision.timestamp} ({when})"]
    if revision.redirected_from:
        origin = f'redirected from "{revision.redirected_from}"'
        if revision.fragment:
            origin += f' (target section: "{revision.fragment}")'
        parts.append(origin)
    if section_label:
        parts.append(f'section "{section_label}"')
    return "[" + " | ".join(parts) + "]"


def _excerpts_or_start(text: str, query: str, room: int) -> str:
    """The passages of ``text`` that match ``query`` when there are any, else the start of the text."""
    if not query:
        return truncate(text, room)
    marker = f'[excerpts relevant to "{query}"; page has {len(text)} chars]'
    excerpts = select_relevant(text, query, room - len(marker) - 1)
    if excerpts:
        return f"{excerpts}\n{marker}"
    marker = f'[no passage matches "{query}"]'
    return f"{truncate(text, room - len(marker) - 1)}\n{marker}"


def _present(header: str, text: str, query: str, hint: str) -> str:
    """Header, then the text (whole if it fits, else excerpts or its start), then the sections hint."""
    room = MAX_OUTPUT_CHARS - len(header) - 1
    if len(text) <= room:
        return f"{header}\n{text}"
    body = _excerpts_or_start(text, query, room - (len(hint) + 1 if hint else 0))
    return "\n".join(part for part in (header, body, hint) if part)


def _read_page(title: str | None, lang: str | None, as_of: str | None, section: str | None, query: str | None) -> str:
    code = normalize_language(lang)
    name, _, fragment = as_text(title).partition("#")  # "#" cannot occur in a title: "Page#Section" names a section
    name = name.strip()
    if not name:
        raise WikipediaError("title is empty")
    if "|" in name:  # the API would read it as several titles and only the first would be shown
        raise WikipediaError(f'a title cannot contain "|" (got "{name}")')
    cutoff = cutoff_for(as_of)
    revision = find_revision(code, name, cutoff)
    wanted = as_text(section).strip() or fragment.strip()
    if wanted:
        chosen = _select_section(code, revision, wanted)
        parsed = parse_revision(code, revision, "text", str(chosen["index"]))
        label, hint = plain_text(str(chosen.get("line", ""))), ""
    else:
        parsed = parse_revision(code, revision, "text|sections|tocdata")
        sections = _sections_of(parsed)
        label = ""
        hint = f"Sections (use the section argument): {_titles_line(sections, SECTION_HINT_CHARS)}" if sections else ""
    text = _markdown_of(parsed, revision) or EMPTY_PAGE_NOTE
    focus = " ".join(as_text(query).split())[:MAX_QUERY_CHARS]
    return _present(_header(code, revision, cutoff, label), text, focus, hint)


# --- tools --------------------------------------------------------------------------------------


class WikipediaSearchTool(Tool):
    """Search Wikipedia for article titles."""

    name = "wikipedia_search"
    description = "Search Wikipedia. Returns up to 8 article titles with a short snippet; read one with wikipedia_page."
    inputs = {
        "query": {"type": "string", "description": "The search query."},
        "lang": {
            "type": "string",
            "description": "Wikipedia language edition, e.g. 'en' (default).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, lang: str = "en") -> str:
        try:
            code = normalize_language(lang)
            text = " ".join(as_text(query).split())
            if not text:
                raise WikipediaError("query is empty")
            return _format_search(code, text, search(code, text, SEARCH_LIMIT))
        except WikipediaError as error:
            return f"Error: {error}"


class WikipediaPageTool(Tool):
    """Read a Wikipedia article, or one section of it."""

    name = "wikipedia_page"
    description = (
        "Read a Wikipedia article as markdown, tables included (output capped at about 6000 characters). "
        "Use as_of (YYYY-MM-DD) for the revision of that date, section for one section, "
        "query for only the passages that match it."
    )
    inputs = {
        "title": {"type": "string", "description": "Exact article title."},
        "lang": {
            "type": "string",
            "description": "Wikipedia language edition, e.g. 'en' (default).",
            "nullable": True,
        },
        "as_of": {
            "type": "string",
            "description": "Read the last revision up to the end of this day, YYYY-MM-DD (optional).",
            "nullable": True,
        },
        "section": {
            "type": "string",
            "description": "Only return the section with this title (optional).",
            "nullable": True,
        },
        "query": {
            "type": "string",
            "description": "What to look for: long pages return only the matching passages (optional).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, title: str, lang: str = "en", as_of: str = "", section: str = "", query: str = "") -> str:
        try:
            return _read_page(title, lang, as_of, section, query)
        except WikipediaError as error:
            return f"Error: {error}"
