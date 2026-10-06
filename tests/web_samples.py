"""Builders and fixtures for the web tool tests: sample pages, a tiny generated PDF, a fake ddgs and mocked HTTP.

The fixtures are imported by name into the test modules that use them, for example
``from web_samples import fake_ddgs, http``. Everything here is invented sample data;
nothing comes from a benchmark.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import SimpleNamespace

import pytest
import responses

from gaia_agent.tools import web
from gaia_agent.tools.web import ReadWebpageTool

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
PAGE_URL = "https://example.org/dir/page.html"


def html_page(body: str, title: str = "Sample page") -> str:
    """Wrap ``body`` in a minimal HTML document."""
    return f"<!doctype html><html><head><title>{title}</title></head><body>{body}</body></html>"


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream(lines: Sequence[str]) -> str:
    shown = " T* ".join(f"({_escape_pdf_text(line)}) Tj" for line in lines)
    return f"BT /F1 12 Tf 72 720 Td 16 TL {shown} ET"


def make_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a small, valid PDF whose pages carry the given lines of ASCII text.

    ``make_pdf([["Hello", "world"], ["Second page"]])`` makes a two-page document.
    A page with no lines is blank (it has no text layer).
    """
    page_ids = [4 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, lines in zip(page_ids, pages, strict=True):
        stream = _page_stream(lines)
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {page_id + 1} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objects[page_id + 1] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number in sorted(objects):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("latin-1")
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n").encode("ascii")
    return bytes(out)


@pytest.fixture(autouse=True)
def forbid_real_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that forgets to fake ddgs must fail loudly instead of searching the web."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test reached the real ddgs; use the fake_ddgs fixture")

    monkeypatch.setattr(web, "DDGS", forbidden)


@pytest.fixture
def fake_ddgs(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A stand-in for ``ddgs.DDGS``: set ``rows`` or ``error``, inspect ``calls``."""
    state = SimpleNamespace(rows=[], error=None, calls=[], init_kwargs=[])

    class FakeDDGS:
        def __init__(self, *args: object, **kwargs: object) -> None:
            state.init_kwargs.append(kwargs)

        def text(self, query: str, **kwargs: object) -> list[dict[str, str]]:
            state.calls.append((query, kwargs))
            if state.error is not None:
                raise state.error
            return list(state.rows)

    monkeypatch.setattr(web, "DDGS", FakeDDGS)
    return state


@pytest.fixture
def http() -> Iterator[responses.RequestsMock]:
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


def ddgs_row(number: int, body: str = "A short description.") -> dict[str, str]:
    return {"title": f"Result {number}", "href": f"https://example.org/page{number}", "body": body}


def wikipedia_payload(*hits: tuple[str, str]) -> dict[str, object]:
    search = [
        {"ns": 0, "title": title, "pageid": index, "snippet": snippet} for index, (title, snippet) in enumerate(hits)
    ]
    return {"batchcomplete": True, "query": {"searchinfo": {"totalhits": len(hits)}, "search": search}}


def serve(
    http: responses.RequestsMock,
    body: str | bytes,
    content_type: str = "text/html; charset=utf-8",
    url: str = PAGE_URL,
    status: int = 200,
) -> None:
    http.add(responses.GET, url, body=body, status=status, content_type=content_type)


def read(url: str = PAGE_URL, query: str = "") -> str:
    return ReadWebpageTool().forward(url, query)


def paragraphs(count: int, inserts: dict[int, str] | None = None) -> str:
    extra = inserts or {}
    return "".join(
        f"<p>Paragraph {index}. The committee reviewed routine budget items and adjourned. "
        f"Nothing else was said about the matter at all. {extra.get(index, '')}</p>"
        for index in range(count)
    )
