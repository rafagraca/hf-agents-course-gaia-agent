"""Client for the MediaWiki API of a Wikipedia language edition.

Everything goes through ``https://{lang}.wikipedia.org/w/api.php`` with ``requests``: a
timeout, a few retries on transient server errors and an identified User-Agent.  Every
failure the caller can act on (network, HTTP status, bad JSON, API error, missing page,
bad date or language) is raised as a :class:`WikipediaError` with a short message.
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gaia_agent.tools.wikipedia_html import plain_text

logger = logging.getLogger(__name__)

API_URL = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "hf-agents-course-gaia-agent/0.1 (https://github.com/rafagraca)"
TIMEOUT = (10, 30)  # connect and read timeouts, in seconds
SUGGESTION_LIMIT = 3

_LANG = re.compile(r"[a-z]{2,12}(?:-[a-z0-9]{1,12}){0,2}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class WikipediaError(Exception):
    """A problem to report to the model as a short message (never a bug in this module)."""


@dataclass(frozen=True)
class Revision:
    """The revision of an article that is going to be read."""

    title: str
    revid: int
    timestamp: str
    redirected_from: str = ""
    fragment: str = ""  # section a redirect pointed at, if any


@dataclass(frozen=True)
class Hit:
    """One search result: the page title and a plain-text snippet around the match."""

    title: str
    snippet: str


@dataclass(frozen=True)
class SearchResult:
    """The results of a search and the spelling the wiki suggests instead (empty when it has none)."""

    hits: tuple[Hit, ...]
    suggestion: str


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=2,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=False,
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _api(lang: str, params: dict[str, Any]) -> dict[str, Any]:
    """One MediaWiki API answer as a dict; every failure becomes a ``WikipediaError``."""
    host = f"{lang}.wikipedia.org"
    query = {**params, "format": "json", "formatversion": 2}
    try:
        response = _session().get(API_URL.format(lang=lang), params=query, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as error:
        logger.warning("Wikipedia request to %s failed: %s", host, error)
        raise WikipediaError(f"could not reach {host} ({type(error).__name__})") from error
    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise WikipediaError(f"{host} did not answer with a JSON object")
    failure = data.get("error")
    if isinstance(failure, dict):
        raise WikipediaError(f"Wikipedia API error ({failure.get('code', 'unknown')}): {failure.get('info', '')}")
    return data


def as_text(value: object) -> str:
    """A tool argument as text: ``None`` is empty and anything else the model passed (``2009``) is its ``str``."""
    return "" if value is None else str(value)


def normalize_language(lang: str | None) -> str:
    """The language edition code (``en`` when empty); only plain codes are accepted, never a host name."""
    code = as_text(lang).strip().lower() or "en"
    if not _LANG.fullmatch(code):
        raise WikipediaError(f'invalid language code "{lang}" (use e.g. "en", "pt", "simple")')
    return code


def _is_calendar_date(text: str) -> bool:
    try:
        datetime.date.fromisoformat(text)
    except ValueError:
        return False
    return True


def cutoff_for(as_of: str | None) -> str:
    """The timestamp that ends the day ``as_of`` (YYYY-MM-DD); empty when no date was given."""
    text = as_text(as_of).strip()
    if not text:
        return ""
    if not _DATE.fullmatch(text) or not _is_calendar_date(text):
        raise WikipediaError(f'as_of must be a date in YYYY-MM-DD format, got "{as_of}"')
    return f"{text}T23:59:59Z"


def search(lang: str, query: str, limit: int) -> SearchResult:
    """Article titles matching ``query`` with cleaned snippets, and the API's spelling suggestion."""
    data = _api(
        lang,
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet",
            "srinfo": "suggestion",
        },
    )
    found = data.get("query") or {}
    items = [item for item in (found.get("search") or []) if isinstance(item, dict) and item.get("title")]
    hits = tuple(Hit(str(item["title"]), plain_text(item.get("snippet"))) for item in items[:limit])
    return SearchResult(hits, str((found.get("searchinfo") or {}).get("suggestion") or ""))


def _missing_page_message(lang: str, title: str) -> str:
    """Page not found, with what a search for the title suggests (similar titles, a corrected spelling)."""
    message = f'page "{title}" does not exist on {lang}.wikipedia.org.'
    try:
        found = search(lang, title, SUGGESTION_LIMIT)
    except WikipediaError as error:
        logger.info("No title suggestions for %r: %s", title, error)
        found = SearchResult((), "")
    hints = []
    if found.hits:
        hints.append(f"Similar titles: {'; '.join(hit.title for hit in found.hits)}")
    if found.suggestion:
        hints.append(f'Did you mean "{found.suggestion}"?')
    if not hints:
        return f"{message} Use wikipedia_search to find the exact title."
    return f"{message} {'. '.join(hints)}"


def find_revision(lang: str, title: str, cutoff: str) -> Revision:
    """The latest revision of ``title`` (up to ``cutoff`` when given), after following redirects."""
    params: dict[str, Any] = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids|timestamp",
        "rvlimit": 1,
        "rvdir": "older",
        "redirects": 1,
    }
    if cutoff:
        params["rvstart"] = cutoff
    query = _api(lang, params).get("query") or {}
    pages = query.get("pages") or [{}]
    found = pages[0] if isinstance(pages[0], dict) else {}
    if found.get("invalid"):
        reason = found.get("invalidreason")
        raise WikipediaError(f'invalid title "{title}"' + (f": {reason}" if reason else ""))
    if found.get("missing") or not found:
        raise WikipediaError(_missing_page_message(lang, title))
    revisions = found.get("revisions") or []
    if not revisions:
        since = f" on or before {cutoff[:10]} (the page may have been created later)" if cutoff else ""
        raise WikipediaError(f'"{found.get("title", title)}" has no revision{since}.')
    redirects = query.get("redirects") or []
    try:
        return Revision(
            title=str(found.get("title", title)),
            revid=int(revisions[0]["revid"]),
            timestamp=str(revisions[0]["timestamp"]),
            redirected_from=str(redirects[0].get("from", "")) if redirects else "",
            fragment=next((str(r["tofragment"]) for r in reversed(redirects) if r.get("tofragment")), ""),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise WikipediaError(f"unexpected revision data from the {lang}.wikipedia.org API") from error


def parse_revision(lang: str, revision: Revision, prop: str, section: str | None = None) -> dict[str, Any]:
    """``action=parse`` of one revision (optionally one section): the ``parse`` object, or ``{}``."""
    params: dict[str, Any] = {
        "action": "parse",
        "oldid": revision.revid,
        "prop": prop,
        "disablelimitreport": 1,
        "disableeditsection": 1,
        "disabletoc": 1,
    }
    if section is not None:
        params["section"] = section
    parsed = _api(lang, params).get("parse")
    return parsed if isinstance(parsed, dict) else {}
