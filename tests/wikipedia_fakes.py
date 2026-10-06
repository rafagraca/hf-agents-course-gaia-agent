"""Fakes and builders for the Wikipedia tool tests: a canned MediaWiki API and invented pages.

``FakeWikipedia`` answers the HTTP requests made with ``requests`` (through ``responses``),
one canned reply per kind of request, and records every request so that the tests can check
the parameters that were sent.  Every page, title and number below is invented.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import responses

from gaia_agent.tools.wikipedia import WikipediaPageTool

USER_AGENT = "hf-agents-course-gaia-agent/0.1 (https://github.com/rafagraca)"
MAX_OUTPUT_CHARS = 6000

ARTICLE_HTML = (
    '<div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr">'
    '<table class="infobox"><tbody><tr><th colspan="2">Example Band</th></tr>'
    '<tr><th scope="row">Origin</th><td>Exampleton</td></tr></tbody></table>'
    '<p><b>Example Band</b> is a fictional group.<sup class="reference"><a href="#n1">[1]</a></sup></p>'
    '<div class="mw-heading mw-heading2"><h2 id="History">History</h2><span class="mw-editsection">'
    '<a href="/w/index.php?action=edit">edit</a></span></div>'
    "<p>The band formed in 1990.</p>"
    '<div class="mw-heading mw-heading2"><h2 id="Discography">Discography</h2></div>'
    '<table class="wikitable"><caption>Studio albums</caption><tbody>'
    "<tr><th>Title</th><th>Year</th></tr>"
    "<tr><td>Blue Morning</td><td>1992</td></tr><tr><td>Red Evening</td><td>1995</td></tr>"
    "</tbody></table>"
    '<div class="navbox"><table><tr><td>Navigation</td></tr></table></div>'
    "</div>"
)

ARTICLE_MARKDOWN = """\
| Example Band |
| Origin | Exampleton |

Example Band is a fictional group.

## History

The band formed in 1990.

## Discography

Table: Studio albums
| Title | Year |
| --- | --- |
| Blue Morning | 1992 |
| Red Evening | 1995 |"""

STUDIO_ALBUMS_HTML = (
    '<div class="mw-content-ltr mw-parser-output"><div class="mw-heading mw-heading3">'
    '<h3 id="Studio_albums">Studio albums</h3></div>'
    "<table><tr><th>Title</th><th>Year</th></tr><tr><td>Blue Morning</td><td>1992</td></tr></table></div>"
)

SECTIONS = [
    {"toclevel": 1, "level": "2", "line": "History", "number": "1", "index": "1", "anchor": "History"},
    {"toclevel": 1, "level": "2", "line": "Discography", "number": "2", "index": "2", "anchor": "Discography"},
    {
        "toclevel": 2,
        "level": "3",
        "line": "<i>Studio</i> albums",
        "number": "2.1",
        "index": "3",
        "anchor": "Studio_albums",
    },
    {"toclevel": 1, "level": "2", "line": "Templated", "number": "3", "index": "T-1", "anchor": "Templated"},
]


@dataclass(frozen=True)
class Seen:
    """One request received by the fake API."""

    host: str
    params: dict[str, str]
    headers: dict[str, str]


class FakeWikipedia:
    """A canned MediaWiki API: one reply per kind of request, every request recorded.

    Kinds: ``search``, ``revisions``, ``sections`` (parse without text), ``page`` (parse with the
    whole text) and ``section_text`` (parse of one section).  A reply is a JSON-able value, an
    exception to raise, or a ``(status, text)`` pair for raw HTTP answers.
    """

    def __init__(self, mock: responses.RequestsMock) -> None:
        self.replies: dict[str, Any] = {}
        self.seen: list[Seen] = []
        mock.add_callback(responses.GET, re.compile(r"https://[a-z\-]+\.wikipedia\.org/w/api\.php"), self._serve)

    def reply(self, kind: str, body: Any) -> None:
        self.replies[kind] = body

    def requests_of(self, kind: str) -> list[Seen]:
        return [seen for seen in self.seen if _kind(seen.params) == kind]

    def params_of(self, kind: str) -> dict[str, str]:
        (only,) = self.requests_of(kind)
        return only.params

    def _serve(self, request: Any) -> Any:
        parts = urlsplit(request.url)
        seen = Seen(parts.hostname or "", dict(parse_qsl(parts.query)), dict(request.headers))
        self.seen.append(seen)
        kind = _kind(seen.params)
        if kind not in self.replies:
            raise AssertionError(f"no canned reply for a {kind!r} request: {seen.params}")
        reply = self.replies[kind]
        if isinstance(reply, Exception):
            return reply
        if isinstance(reply, tuple):
            status, text = reply
            return status, {}, text
        return 200, {"Content-Type": "application/json"}, json.dumps(reply)


def _kind(params: dict[str, str]) -> str:
    if params.get("list") == "search":
        return "search"
    if params.get("prop") == "revisions":
        return "revisions"
    if params.get("action") == "parse":
        if "section" in params:
            return "section_text"
        return "page" if "text" in params.get("prop", "").split("|") else "sections"
    raise AssertionError(f"unexpected request: {params}")


def fake_wikipedia() -> Iterator[FakeWikipedia]:
    """Generator for a pytest fixture: ``yield from fake_wikipedia()``."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield FakeWikipedia(mock)


def revisions(
    title: str = "Example Band",
    revid: int = 2002,
    timestamp: str = "2026-09-30T08:00:00Z",
    redirects: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "pages": [{"pageid": 7, "ns": 0, "title": title, "revisions": [{"revid": revid, "timestamp": timestamp}]}]
    }
    if redirects:
        query["redirects"] = redirects
    return {"batchcomplete": True, "query": query}


def missing_page(title: str) -> dict[str, Any]:
    return {"query": {"pages": [{"title": title, "missing": True}]}}


def page(html: str, sections: list[dict[str, Any]] | None = None, title: str = "Example Band") -> dict[str, Any]:
    parsed: dict[str, Any] = {"title": title, "pageid": 7, "revid": 2002, "text": html}
    if sections is not None:
        parsed["sections"] = sections
    return {"parse": parsed}


def sections_reply(sections: list[dict[str, Any]]) -> dict[str, Any]:
    return {"parse": {"title": "Example Band", "pageid": 7, "revid": 2002, "sections": sections}}


def search_reply(*hits: tuple[str, str], suggestion: str = "") -> dict[str, Any]:
    body: dict[str, Any] = {
        "batchcomplete": True,
        "query": {"search": [{"ns": 0, "title": title, "snippet": snippet} for title, snippet in hits]},
    }
    if suggestion:
        body["query"]["searchinfo"] = {"suggestion": suggestion}
    return body


def read(**arguments: Any) -> str:
    return WikipediaPageTool().forward(**arguments)


def long_page(extra: str = "") -> str:
    filler = "".join(
        f"<p>Paragraph {number}: the band toured venue number {number} and played songs for hours.</p>"
        for number in range(300)
    )
    return (
        '<div class="mw-parser-output"><p>Lead sentence.</p>'
        f'<div class="mw-heading mw-heading2"><h2>History</h2></div>{filler}{extra}</div>'
    )
