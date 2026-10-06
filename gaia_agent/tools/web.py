"""Web tools: search the web and read web pages. Outputs are compact and truncated.

``web_search`` asks the search engines behind ``ddgs`` and, when they fail or
find nothing, falls back to the Wikipedia search API ("plan B"), saying so.
``read_webpage`` downloads a page (HTML, PDF or plain text) and returns its
text; with a ``query`` it returns only the passages that match it.

Both tools return a short ``Error: ...`` message instead of raising, so the
agent can react to a failure. Fetching refuses non-http(s) URLs and addresses
on the local or a private network, redirects included, in three layers: the
text of the URL, what its host name resolves to, and the address the socket
really connects to (see ``web_urls`` and ``_guarded_http``). Pages are
untrusted data: nothing read here may ask for more than a text answer.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import quote, urljoin

import requests
from ddgs import DDGS
from smolagents import Tool

from gaia_agent.tools._guarded_http import BlockedAddressError, guarded_session
from gaia_agent.tools._sources import BLOCKED_MESSAGE, blocked_source, looks_like_answer_key
from gaia_agent.tools.web_docs import UnreadableDocumentError, document_text
from gaia_agent.tools.web_text import build_excerpt, shorten
from gaia_agent.tools.web_urls import UrlError, check_public_destination, normalize_url

__all__ = [
    "FetchedPage",
    "ReadWebpageTool",
    "SearchHit",
    "WebSearchTool",
    "WebToolError",
    "build_excerpt",
    "document_text",
    "fetch_page",
    "format_hits",
    "read_limited",
    "read_page",
    "search_web",
]

logger = logging.getLogger(__name__)

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
API_USER_AGENT = "hf-agents-course-gaia-agent/0.1 (+https://github.com/rafagraca/hf-agents-course-gaia-agent)"
REQUEST_HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_TIMEOUT_S = 30  # per socket operation, as in ``requests``
DOWNLOAD_DEADLINE_S = 90  # for a whole download, however slowly the server drips
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
READ_CHUNK_BYTES = 64 * 1024

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_CAP = 10
SNIPPET_CHARS = 220
TITLE_CHARS = 120
SEARCH_TIMEOUT_S = 10
SEARCH_OVERFETCH = 5  # extra results asked for, to make up for the blocked sources that are dropped
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_URL = "https://en.wikipedia.org/wiki/"
WIKIPEDIA_TIMEOUT_S = 15

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MARKUP = re.compile(r"<[^>]*>")


class WebToolError(Exception):
    """A web tool could not do its job; the message is short and safe to show the agent."""


def _squash(text: object) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(str(text or "").split())


# --- web_search -------------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchHit:
    """One search result."""

    title: str
    url: str
    snippet: str = ""


def format_hits(hits: Sequence[SearchHit]) -> str:
    """A compact numbered list: title, URL and snippet on three lines per hit."""
    lines: list[str] = []
    for number, hit in enumerate(hits, start=1):
        lines += [f"{number}. {hit.title}", f"   {hit.url}"]
        if hit.snippet:
            lines.append(f"   {hit.snippet}")
    return "\n".join(lines)


def _hit_from_ddgs(row: Mapping[str, Any]) -> SearchHit | None:
    url = _squash(row.get("href") or row.get("url"))
    if not url:
        return None
    title = shorten(_squash(row.get("title")), TITLE_CHARS) or url
    return SearchHit(title, url, shorten(_squash(row.get("body")), SNIPPET_CHARS))


def _search_ddgs(query: str, limit: int) -> tuple[list[SearchHit], int]:
    """The first ``limit`` results of the search that may be read, and how many blocked sources were dropped."""
    rows = DDGS(timeout=SEARCH_TIMEOUT_S).text(query, max_results=limit + SEARCH_OVERFETCH)
    found = [hit for hit in (_hit_from_ddgs(row) for row in rows or []) if hit is not None]
    allowed = [hit for hit in found if blocked_source(hit.url) is None]
    return allowed[:limit], len(found) - len(allowed)


def _wikipedia_url(title: str) -> str:
    return WIKIPEDIA_PAGE_URL + quote(title.replace(" ", "_"), safe="/:()_,!*'@$;~")


def _wikipedia_rows(payload: object) -> list[Mapping[str, Any]]:
    """The ``search`` entries of a MediaWiki ``list=search`` response."""
    if not isinstance(payload, dict):
        raise WebToolError("unexpected response from the Wikipedia API")
    if "error" in payload:
        raise WebToolError("Wikipedia API error")
    block = payload.get("query")
    rows = block.get("search") if isinstance(block, dict) else None
    if not isinstance(rows, list):
        raise WebToolError("unexpected response from the Wikipedia API")
    return [row for row in rows if isinstance(row, dict)]


def _search_wikipedia(query: str, limit: int) -> list[SearchHit]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srprop": "snippet",
        "format": "json",
        "formatversion": 2,
    }
    try:
        response = requests.get(
            WIKIPEDIA_API_URL, params=params, headers={"User-Agent": API_USER_AGENT}, timeout=WIKIPEDIA_TIMEOUT_S
        )
        if response.status_code >= 400:
            raise WebToolError(f"HTTP {response.status_code}")
        payload = response.json()
    except requests.exceptions.Timeout as exc:
        raise WebToolError("timed out") from exc
    except requests.exceptions.JSONDecodeError as exc:
        raise WebToolError("unexpected response from the Wikipedia API") from exc
    except requests.exceptions.RequestException as exc:
        raise WebToolError(f"connection failed ({type(exc).__name__})") from exc
    hits: list[SearchHit] = []
    for row in _wikipedia_rows(payload):
        title = _squash(row.get("title"))
        if title:
            snippet = _squash(unescape(_MARKUP.sub("", str(row.get("snippet") or ""))))
            hits.append(SearchHit(shorten(title, TITLE_CHARS), _wikipedia_url(title), shorten(snippet, SNIPPET_CHARS)))
    return hits[:limit]


def _result_limit(value: Any) -> int:
    """The requested number of results, clamped to ``1..MAX_RESULTS_CAP`` (default when unusable)."""
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_MAX_RESULTS
    return max(1, min(MAX_RESULTS_CAP, limit))


def _plan_b(query: str, limit: int, problem: str) -> str:
    """Answer from the Wikipedia search, explaining why the web search did not."""
    try:
        hits = _search_wikipedia(query, limit)
    except WebToolError as exc:
        return f"Error: {problem} and the Wikipedia fallback failed too ({exc})"
    if not hits:
        return f"No results for '{query}': {problem} and the Wikipedia search (plan B) found nothing either"
    return f"Note: {problem}; these results come from the Wikipedia search (plan B).\n{format_hits(hits)}"


def search_web(query: object, max_results: object = DEFAULT_MAX_RESULTS) -> str:
    """Search the web; return a numbered list of results, or a short note or error message."""
    text = _squash(query)
    if not text:
        return "Error: the search query is empty"
    limit = _result_limit(max_results)
    try:
        hits, dropped = _search_ddgs(text, limit)
        problem = (
            "the web search found only sources that publish benchmark answers, which are blocked"
            if dropped and not hits
            else "the web search returned no results"
        )
    except Exception as exc:  # noqa: BLE001 - ddgs raises several kinds of exception
        logger.warning("web search failed (%s)", type(exc).__name__)
        hits, problem = [], f"the web search failed ({type(exc).__name__})"
    return format_hits(hits) if hits else _plan_b(text, limit, problem)


# --- read_webpage: fetching -------------------------------------------------------------------


def _checked_url(raw: object) -> str:
    """:func:`normalize_url`, the blocked sources and :func:`check_public_destination`, as a :class:`WebToolError`.

    Applied to the first address and to every redirect hop, before anything is requested.
    """
    try:
        url = normalize_url(raw)
        blocked = blocked_source(url)
        if blocked is not None:
            raise WebToolError(blocked)
        check_public_destination(url)
    except UrlError as exc:
        raise WebToolError(str(exc)) from exc
    return url


@dataclass(frozen=True)
class FetchedPage:
    """A downloaded page: its final address, ``Content-Type`` header, bytes and whether they were cut."""

    url: str
    content_type: str
    body: bytes
    cut: bool


def read_limited(
    response: Any,
    max_bytes: int,
    *,
    deadline_s: float = DOWNLOAD_DEADLINE_S,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[bytes, bool]:
    """Read at most ``max_bytes`` of a streamed response, within ``deadline_s`` seconds in total.

    Returns the bytes and whether the body was longer than the limit.
    """
    deadline = clock() + deadline_s
    received = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=READ_CHUNK_BYTES):
            received.extend(chunk)
            if len(received) > max_bytes:
                return bytes(received[:max_bytes]), True
            if clock() > deadline:
                raise WebToolError("the download took too long")
    except requests.exceptions.RequestException as exc:
        raise WebToolError(f"the download failed ({type(exc).__name__})") from exc
    return bytes(received), False


def _get_once(session: requests.Session, url: str) -> requests.Response:
    """One GET without following redirects, with request failures turned into short messages."""
    try:
        return session.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_S, stream=True, allow_redirects=False)
    except BlockedAddressError as exc:  # the connection guard: the name resolved to a local or private address
        raise WebToolError(str(exc)) from exc
    except requests.exceptions.Timeout as exc:
        raise WebToolError(f"the request timed out after {REQUEST_TIMEOUT_S} s") from exc
    except requests.exceptions.SSLError as exc:
        raise WebToolError("TLS/SSL error: the secure connection failed") from exc
    except requests.exceptions.ConnectionError as exc:
        raise WebToolError(f"could not connect ({type(exc).__name__})") from exc
    except requests.exceptions.RequestException as exc:
        raise WebToolError(f"the request failed ({type(exc).__name__})") from exc
    except ValueError as exc:  # ``requests`` parses a redirect address as soon as it sees one
        raise WebToolError("the URL or a redirect address is malformed") from exc


def _redirect_target(response: requests.Response, current: str) -> str | None:
    location = response.headers.get("Location")
    if response.status_code not in _REDIRECT_STATUSES or not location:
        return None
    return urljoin(current, location.strip())


def _page_from(response: requests.Response, url: str, max_bytes: int) -> FetchedPage:
    if response.status_code >= 400:
        raise WebToolError(f"HTTP {response.status_code} {response.reason or ''}".strip())
    body, cut = read_limited(response, max_bytes)
    return FetchedPage(url=url, content_type=response.headers.get("Content-Type", ""), body=body, cut=cut)


def fetch_page(url: object, max_bytes: int | None = None) -> FetchedPage:
    """Download ``url`` (at most ``max_bytes``, default :data:`MAX_DOWNLOAD_BYTES`).

    Redirects are followed by hand, up to :data:`MAX_REDIRECTS`, and every hop is
    checked like the first address. Raises :class:`WebToolError` on any failure.
    """
    limit = MAX_DOWNLOAD_BYTES if max_bytes is None else max_bytes
    current = _checked_url(url)
    with guarded_session() as session:  # one session, so cookies set by a redirect are kept
        for _ in range(MAX_REDIRECTS + 1):
            response = _get_once(session, current)
            try:
                target = _redirect_target(response, current)
                if target is None:
                    return _page_from(response, current, limit)
            finally:
                response.close()
            current = _checked_url(target)
    raise WebToolError(f"too many redirects (more than {MAX_REDIRECTS})")


def _size_label(size: int) -> str:
    megabyte = 1024 * 1024
    return f"{size // megabyte} MB" if size >= megabyte and size % megabyte == 0 else f"{size} bytes"


def read_page(url: object, query: object = "") -> str:
    """Fetch ``url`` and return its text, or only the passages matching ``query``.

    Returns a short ``Error: ...`` message instead of raising.
    """
    try:
        page = fetch_page(url)
        text = document_text(page.body, page.content_type, page.url, complete=not page.cut)
    except (WebToolError, UnreadableDocumentError) as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001 - a tool reports problems to the agent instead of raising
        logger.exception("read_webpage failed unexpectedly")
        return f"Error: unexpected failure while reading the page ({type(exc).__name__})"
    if not text.strip():
        return "Error: the page has no readable text (it may need JavaScript, or be a scanned document)"
    if looks_like_answer_key(text):
        return f"Error: {BLOCKED_MESSAGE}"
    excerpt = build_excerpt(text, str(query or ""))
    if page.cut:
        excerpt += f"\n[the page is larger than {_size_label(MAX_DOWNLOAD_BYTES)}: only the start was read]"
    return excerpt


# --- the tools --------------------------------------------------------------------------------


class WebSearchTool(Tool):
    """Search the web and return the top results."""

    name = "web_search"
    description = "Search the web. Returns the top results as title, url and a short snippet."
    inputs = {
        "query": {"type": "string", "description": "The search query."},
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default 5).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
        return search_web(query, max_results)


class ReadWebpageTool(Tool):
    """Fetch a web page and return its text."""

    name = "read_webpage"
    description = (
        "Fetch a web page, PDF or text file and return its text. Pass `query` to get only the passages relevant to it."
    )
    inputs = {
        "url": {"type": "string", "description": "The page URL."},
        "query": {
            "type": "string",
            "description": "What to look for in the page (optional).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, url: str, query: str = "") -> str:
        return read_page(url, query)
