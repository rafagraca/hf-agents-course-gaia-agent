"""Turn fetched bytes into compact text: HTML to markdown, PDF to text, charset handling.

Used by the web tools. Everything here works on bytes or strings already in
memory, so it needs no network and can be tested with small samples.
"""

from __future__ import annotations

import codecs
import io
import logging
import re
import warnings
from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup, FeatureNotFound, MarkupResemblesLocatorWarning, Tag
from bs4.element import ProcessingInstruction
from markdownify import ATX, MarkdownConverter
from pypdf import PdfReader
from pypdf.errors import DependencyError

logger = logging.getLogger(__name__)

# bs4 warns when a tiny input looks like a file name or URL; for a fetched page that is only noise.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

PREFERRED_PARSER = "lxml"  # fixes unclosed <li>/<p>/<td> like a browser; "html.parser" is the fallback
MAX_PDF_PAGES = 200

_DROPPED_TAGS = ["head", "script", "style", "noscript", "template", "svg", "iframe", "canvas", "nav"]
_ARTICLE_CHROME_TAGS = ["header", "footer"]  # page furniture, except inside an <article>
_DEAD_LINK_PREFIXES = ("#", "javascript:", "data:")
_TITLE_LOOKAHEAD_CHARS = 400

_INVISIBLE = re.compile("[\u00ad\u200b-\u200d\u2060\ufeff]")  # soft hyphen, zero-width characters, byte-order mark
_ODD_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")  # no-break and other typographic spaces
_BLANK_RUNS = re.compile(r"\n{3,}")
_LONG_WHITESPACE = re.compile(r"\s{40,}")

_HEADER_CHARSET = re.compile(r"charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE)
_META_CHARSET = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([\w.:-]+)", re.IGNORECASE)
_HTML_START = re.compile(r"\s*(?:<!doctype\s+html|<html|<head|<body)", re.IGNORECASE)

_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_PDF_TYPES = frozenset({"application/pdf", "application/x-pdf"})
_GENERIC_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})
_TEXT_TYPES = frozenset({"application/json", "application/xml", "application/x-ndjson", "application/yaml"})


class UnreadableDocumentError(Exception):
    """The content cannot be turned into text; the message is short enough to show an agent."""


def media_type(content_type: str) -> str:
    """The lower-case media type of a ``Content-Type`` header, without its parameters."""
    return content_type.split(";", 1)[0].strip().lower()


# --- bytes to str -----------------------------------------------------------------------------


def _candidate_charsets(body: bytes, content_type: str) -> list[str]:
    """Charsets worth trying, in order: the header's, UTF-8, then one declared in the markup."""
    charsets: list[str] = []
    in_header = _HEADER_CHARSET.search(content_type)
    if in_header:
        charsets.append(in_header.group(1))
    charsets.append("utf-8")
    in_markup = _META_CHARSET.search(body[:2048])
    if in_markup:
        charsets.append(in_markup.group(1).decode("ascii", errors="replace"))
    return charsets


def decode_body(body: bytes, content_type: str = "") -> str:
    """Decode a response body to text without needing the server to be right.

    A byte-order mark decides first. Otherwise the ``charset`` of the
    ``Content-Type`` header is tried, then UTF-8, then a charset declared by a
    ``<meta>`` tag, and finally Windows-1252 (undecodable bytes are replaced).
    """
    if body.startswith(codecs.BOM_UTF8):
        return body[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace")
    if body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return body.decode("utf-16", errors="replace")
    for charset in _candidate_charsets(body, content_type):
        try:
            return body.decode(charset)
        except (UnicodeDecodeError, LookupError):
            continue
    return body.decode("cp1252", errors="replace")


# --- HTML to markdown -------------------------------------------------------------------------


class _CompactConverter(MarkdownConverter):
    """markdownify with compact output: ``#`` headings, no escaping, images reduced to alt text."""

    class Options(MarkdownConverter.DefaultOptions):
        heading_style = ATX
        table_infer_header = True  # a table without <th> still gets a header row
        escape_asterisks = False
        escape_underscores = False

    def convert_img(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        alt = " ".join(str(el.attrs.get("alt") or "").split())
        return f"[image: {alt}]" if alt else ""


def _parse_html(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, PREFERRED_PARSER)
    except FeatureNotFound:
        logger.info("HTML parser %r is not installed; using html.parser", PREFERRED_PARSER)
        return BeautifulSoup(html, "html.parser")


def _page_title(soup: BeautifulSoup) -> str:
    head = soup.find("head")
    title = (soup if head is None else head).find("title")
    return " ".join(title.get_text().split()) if title is not None else ""


def _remove_noise(soup: BeautifulSoup) -> None:
    """Drop scripts, styles, navigation, page furniture (kept inside an ``<article>``) and ``<?...?>`` nodes."""
    for tag in soup.find_all(_DROPPED_TAGS):
        tag.extract()
    for tag in soup.find_all(_ARTICLE_CHROME_TAGS):
        if tag.find_parent("article") is None:
            tag.extract()
    for node in soup.find_all(string=lambda text: isinstance(text, ProcessingInstruction)):
        node.extract()


def _absolute(base_url: str, href: str) -> str:
    try:
        return urljoin(base_url, href)
    except ValueError:  # e.g. a malformed IPv6 literal: keep the link as written
        return href


def _drop_redundant_title(anchor: Tag) -> None:
    """Remove a link's tooltip when it only repeats the link text or the address (it costs tokens)."""
    title = " ".join(str(anchor.get("title") or "").split()).casefold()
    if not title:
        return
    text = " ".join(anchor.get_text().split()).casefold()
    address = unquote(str(anchor.get("href", ""))).replace("_", " ").casefold()
    if title == text or title in address:
        del anchor["title"]


def _tidy_links(soup: BeautifulSoup, base_url: str) -> None:
    """Make links followable: resolve them against ``base_url``, drop dead ones and redundant tooltips."""
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.lower().startswith(_DEAD_LINK_PREFIXES):
            anchor.unwrap()
            continue
        if base_url:
            anchor["href"] = _absolute(base_url, href)
        _drop_redundant_title(anchor)


def _to_markdown(soup: BeautifulSoup) -> str:
    try:
        return _CompactConverter().convert_soup(soup)
    except RecursionError:  # markdownify recurses once per nesting level
        logger.warning("HTML is nested too deeply for the markdown converter; falling back to plain text")
        return soup.get_text("\n")


def _tidy(text: str) -> str:
    """Normalise odd spaces, drop invisible characters, trim lines and collapse runs of blank lines."""
    text = _ODD_SPACES.sub(" ", _INVISIBLE.sub("", text))
    text = "\n".join(line.rstrip() for line in text.split("\n"))  # a regex for this is quadratic on long runs
    return _BLANK_RUNS.sub("\n\n", text).strip()


def _squeeze(run: re.Match[str]) -> str:
    return "\n\n" if "\n" in run.group() else " " * 16


def _squeeze_long_whitespace(text: str) -> str:
    """Shrink runs of 40+ whitespace characters: markdownify's regexes are quadratic on them."""
    return _LONG_WHITESPACE.sub(_squeeze, text)


def html_to_markdown(html: str, base_url: str = "") -> str:
    """Convert an HTML page to compact markdown.

    Scripts, styles, ``<head>``, ``<nav>``, and ``<header>``/``<footer>`` blocks
    outside an ``<article>`` are dropped; tables, links, headings and lists are
    kept. With ``base_url``, relative links become absolute so they can be
    followed; links to ``#anchors`` and ``javascript:`` keep only their text.
    Images are reduced to their alt text. The page title is put first unless
    the body already shows it. Returns ``""`` for a page without body text.
    """
    soup = _parse_html(_squeeze_long_whitespace(html))
    title = _page_title(soup)
    _remove_noise(soup)
    _tidy_links(soup, base_url)
    body = _tidy(_to_markdown(soup))
    if body and title and title.casefold() not in body[:_TITLE_LOOKAHEAD_CHARS].casefold():
        return f"# {title}\n\n{body}"
    return body


# --- PDF to text ------------------------------------------------------------------------------


def _unlock(reader: PdfReader) -> None:
    if reader.is_encrypted and not reader.decrypt(""):
        raise UnreadableDocumentError("the PDF is password protected")


def _page_texts(pages: Sequence[Any], count: int) -> tuple[list[str], int]:
    """Text of the first ``count`` pages, and how many pages could not be read."""
    texts: list[str] = []
    unreadable = 0
    for index in range(count):
        try:
            texts.append(pages[index].extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - pypdf raises assorted errors on malformed pages
            unreadable += 1
            logger.warning("PDF page %d could not be read (%s)", index + 1, type(exc).__name__)
    return texts, unreadable


def pdf_to_text(data: bytes, max_pages: int = MAX_PDF_PAGES) -> str:
    """Extract the text of a PDF, page by page (at most ``max_pages`` pages).

    Notes on skipped or unreadable pages are appended in brackets. Returns
    ``""`` for a PDF without a text layer (a scan). Raises
    :class:`UnreadableDocumentError` when the file cannot be parsed or needs a password.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        _unlock(reader)
        total = len(reader.pages)
        texts, unreadable = _page_texts(reader.pages, min(total, max_pages))
    except UnreadableDocumentError:
        raise
    except DependencyError as exc:
        raise UnreadableDocumentError(f"the PDF cannot be decrypted ({exc})") from exc
    except Exception as exc:  # noqa: BLE001 - pypdf raises assorted errors on malformed files
        logger.warning("PDF parsing failed (%s)", type(exc).__name__)
        raise UnreadableDocumentError(f"the PDF could not be parsed ({type(exc).__name__})") from exc
    text = _tidy("\n\n".join(texts))
    if not text:
        return ""
    notes = []
    if total > max_pages:
        notes.append(f"[PDF has {total} pages; only the first {max_pages} were read]")
    if unreadable:
        notes.append(f"[{unreadable} page{'' if unreadable == 1 else 's'} could not be read]")
    return "\n".join([text, *notes])


# --- dispatch on the kind of content ----------------------------------------------------------


def _is_textual(media: str) -> bool:
    return media.startswith("text/") or media in _TEXT_TYPES or media.endswith(("+json", "+xml"))


def _is_pdf(media: str, url: str, body: bytes) -> bool:
    """A PDF by its first bytes, its content type, or its ``.pdf`` extension (when not served as text)."""
    if body[:1024].lstrip().startswith(b"%PDF-") or media in _PDF_TYPES:
        return True
    return urlsplit(url).path.lower().endswith(".pdf") and not _is_textual(media)


def _looks_binary(body: bytes) -> bool:
    if body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return False
    return b"\x00" in body[:2048]


def _plain(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def document_text(body: bytes, content_type: str = "", url: str = "", complete: bool = True) -> str:
    """Read a fetched document as text: PDF, HTML (as markdown) or plain text.

    The kind is decided from the bytes, the ``Content-Type`` header and the
    URL's extension, so mislabelled documents still work. ``complete=False``
    says the download was cut short, which makes a PDF unreadable. Raises
    :class:`UnreadableDocumentError` for anything that is not text.
    """
    media = media_type(content_type)
    if _is_pdf(media, url, body):
        if not complete:
            raise UnreadableDocumentError("the PDF is larger than the download limit")
        return pdf_to_text(body)
    if media in _HTML_TYPES:
        return html_to_markdown(decode_body(body, content_type), base_url=url)
    if media in _GENERIC_TYPES and not _looks_binary(body):
        text = decode_body(body, content_type)
        return html_to_markdown(text, base_url=url) if _HTML_START.match(text[:1000]) else _plain(text)
    if _is_textual(media):
        return _plain(decode_body(body, content_type))
    raise UnreadableDocumentError(f"cannot read content of type {media or 'unknown'}")
