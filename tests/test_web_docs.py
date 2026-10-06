"""Tests for reading fetched bytes as text: charsets, PDF extraction and content-type dispatch."""

from __future__ import annotations

import codecs
import io
import time

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import DependencyError
from web_samples import html_page, make_pdf

from gaia_agent.tools import web_docs
from gaia_agent.tools.web_docs import (
    UnreadableDocumentError,
    decode_body,
    document_text,
    media_type,
    pdf_to_text,
)

PAGE_URL = "https://example.org/dir/page.html"


def test_utf8_is_the_default_encoding() -> None:
    assert decode_body("café".encode()) == "café"


def test_a_utf8_bom_is_removed() -> None:
    assert decode_body(codecs.BOM_UTF8 + b"hello") == "hello"


def test_utf16_with_a_bom_is_decoded() -> None:
    assert decode_body("héllo".encode("utf-16")) == "héllo"


def test_the_charset_of_the_content_type_header_wins() -> None:
    body = "café".encode("latin-1")

    assert decode_body(body, "text/html; charset=ISO-8859-1") == "café"


def test_a_meta_charset_is_used_when_the_header_has_none() -> None:
    body = b'<meta charset="windows-1252"><p>caf\xe9</p>'

    assert "café" in decode_body(body, "text/html")


def test_an_http_equiv_meta_charset_is_used() -> None:
    body = b'<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">caf\xe9'

    assert decode_body(body).endswith("café")


def test_undeclared_legacy_bytes_fall_back_to_windows_1252() -> None:
    assert decode_body(b"price \x80 5") == "price € 5"


def test_an_unknown_header_charset_is_ignored() -> None:
    assert decode_body("café".encode(), "text/plain; charset=nonsense-9") == "café"


def test_a_header_charset_that_cannot_decode_the_bytes_is_skipped() -> None:
    assert decode_body("café".encode(), "text/plain; charset=ascii") == "café"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("text/html; charset=UTF-8", "text/html"),
        ("Application/PDF", "application/pdf"),
        ("  text/plain ;charset=x", "text/plain"),
        ("", ""),
    ],
)
def test_media_type_drops_parameters_and_case(header: str, expected: str) -> None:
    assert media_type(header) == expected


def test_text_is_extracted_from_a_pdf() -> None:
    text = pdf_to_text(make_pdf([["Quarterly report", "Revenue grew (a lot)"]]))

    assert "Quarterly report" in text
    assert "Revenue grew (a lot)" in text


def test_pdf_pages_are_joined_in_order() -> None:
    text = pdf_to_text(make_pdf([["First page"], ["Second page"], ["Third page"]]))

    assert text.index("First page") < text.index("Second page") < text.index("Third page")


def test_the_pdf_page_limit_is_reported() -> None:
    data = make_pdf([["Page one"], ["Page two"], ["Page three"]])

    text = pdf_to_text(data, max_pages=2)

    assert "Page two" in text
    assert "Page three" not in text
    assert "[PDF has 3 pages; only the first 2 were read]" in text


def test_the_default_pdf_page_limit_is_200_pages() -> None:
    assert web_docs.MAX_PDF_PAGES == 200
    pages = [[f"Page {number}"] for number in range(1, 204)]

    text = pdf_to_text(make_pdf(pages))

    assert "Page 200" in text
    assert "Page 201" not in text
    assert "[PDF has 203 pages; only the first 200 were read]" in text


def test_a_pdf_without_a_text_layer_gives_no_text() -> None:
    assert pdf_to_text(make_pdf([[]])) == ""


def test_bytes_that_are_not_a_pdf_are_unreadable() -> None:
    with pytest.raises(UnreadableDocumentError, match="PDF"):
        pdf_to_text(b"definitely not a pdf")


def test_a_password_protected_pdf_is_reported() -> None:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf([["Secret"]]))))
    writer.encrypt("hunter2", algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(UnreadableDocumentError, match="password"):
        pdf_to_text(buffer.getvalue())


class _FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str:
        if self._text is None:
            raise ValueError("broken content stream")
        return self._text


def _fake_reader(pages: list[_FakePage]) -> type:
    class FakeReader:
        is_encrypted = False

        def __init__(self, stream: object) -> None:
            self.pages = pages

    return FakeReader


def test_pages_that_cannot_be_read_are_skipped_and_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [_FakePage("kept one"), _FakePage(None), _FakePage("kept two")]
    monkeypatch.setattr(web_docs, "PdfReader", _fake_reader(pages))

    text = pdf_to_text(b"%PDF-")

    assert "kept one" in text
    assert "kept two" in text
    assert "[1 page could not be read]" in text


def test_pages_that_cannot_be_read_are_counted_in_the_plural(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [_FakePage(None), _FakePage("kept"), _FakePage(None)]
    monkeypatch.setattr(web_docs, "PdfReader", _fake_reader(pages))

    assert "[2 pages could not be read]" in pdf_to_text(b"%PDF-")


def test_an_aes_encrypted_pdf_without_the_crypto_package_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def needs_crypto(stream: object) -> None:
        raise DependencyError("cryptography>=3.1 is required for AES algorithm")

    monkeypatch.setattr(web_docs, "PdfReader", needs_crypto)

    with pytest.raises(UnreadableDocumentError, match="cannot be decrypted"):
        pdf_to_text(b"%PDF-")


def test_a_huge_run_of_spaces_in_a_pdf_page_is_handled_quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_docs, "PdfReader", _fake_reader([_FakePage("left" + " " * 300_000 + "right")]))

    started = time.perf_counter()
    text = pdf_to_text(b"%PDF-")

    assert "left" in text
    assert "right" in text
    assert time.perf_counter() - started < 3


def test_unexpected_parser_errors_become_unreadable_document_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(stream: object) -> None:
        raise KeyError("/Root")

    monkeypatch.setattr(web_docs, "PdfReader", explode)

    with pytest.raises(UnreadableDocumentError, match="KeyError"):
        pdf_to_text(b"%PDF-")


def test_html_is_converted_and_its_links_resolved() -> None:
    body = html_page('<p><a href="/wiki/Beta">Beta</a></p>').encode()

    text = document_text(body, "text/html; charset=utf-8", PAGE_URL)

    assert "[Beta](https://example.org/wiki/Beta)" in text


def test_a_pdf_is_recognised_by_its_content_type() -> None:
    text = document_text(make_pdf([["Typed as PDF"]]), "application/pdf", "https://example.org/get?id=7")

    assert "Typed as PDF" in text


def test_a_pdf_is_recognised_by_its_first_bytes_whatever_the_content_type() -> None:
    text = document_text(make_pdf([["Mislabelled PDF"]]), "text/html", "https://example.org/get?id=7")

    assert "Mislabelled PDF" in text


def test_a_pdf_is_recognised_by_its_extension_when_the_type_is_generic() -> None:
    body = make_pdf([["Download PDF"]])

    assert "Download PDF" in document_text(body, "application/octet-stream", "https://example.org/a/paper.PDF")
    assert "Download PDF" in document_text(body, "", "https://example.org/a/paper.pdf?dl=1")


def test_a_pdf_extension_is_trusted_when_the_type_is_generic() -> None:
    with pytest.raises(UnreadableDocumentError, match="PDF"):
        document_text(b"truncated garbage", "application/octet-stream", "https://example.org/a/paper.pdf")


def test_a_pdf_extension_is_not_trusted_for_text_responses() -> None:
    assert document_text(b"Service unavailable", "text/plain", "https://example.org/a.pdf") == "Service unavailable"


def test_a_pdf_extension_does_not_turn_an_html_page_into_a_pdf() -> None:
    body = html_page("<p>Not a pdf after all</p>").encode()

    assert "Not a pdf after all" in document_text(body, "text/html", "https://example.org/missing.pdf")


def test_an_incomplete_pdf_download_is_refused() -> None:
    with pytest.raises(UnreadableDocumentError, match="larger"):
        document_text(make_pdf([["Cut short"]]), "application/pdf", PAGE_URL, complete=False)


def test_plain_text_is_left_alone_apart_from_line_endings() -> None:
    assert document_text(b"line one\r\nline two\r\n", "text/plain", PAGE_URL) == "line one\nline two"


def test_text_labelled_plain_is_not_parsed_as_html() -> None:
    assert document_text(b"<p>raw markup</p>", "text/plain", PAGE_URL) == "<p>raw markup</p>"


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("application/json", b'{"year": 2001}'),
        ("application/vnd.api+json", b'{"year": 2001}'),
        ("application/xml", b"<year>2001</year>"),
        ("text/csv", b"year,album\n2001,Alpha"),
    ],
)
def test_structured_text_formats_are_returned_as_text(content_type: str, body: bytes) -> None:
    assert document_text(body, content_type, PAGE_URL) == body.decode()


def test_text_served_with_a_generic_type_is_read_as_text() -> None:
    assert document_text(b"plain words", "application/octet-stream", PAGE_URL) == "plain words"


def test_utf16_text_served_with_a_generic_type_is_not_mistaken_for_binary() -> None:
    body = "héllo wörld".encode("utf-16")

    assert document_text(body, "application/octet-stream", PAGE_URL) == "héllo wörld"


def test_html_served_with_a_generic_type_is_converted() -> None:
    body = b"<html><body><h2>Part</h2><p>Hi there</p></body></html>"

    assert "## Part" in document_text(body, "", PAGE_URL)


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("image/png", b"\x89PNG\r\n\x1a\n"),
        ("application/zip", b"PK\x03\x04"),
        ("application/octet-stream", b"\x00\x01\x02\x03"),
        ("audio/mpeg", b"ID3"),
    ],
)
def test_binary_content_is_refused(content_type: str, body: bytes) -> None:
    with pytest.raises(UnreadableDocumentError, match="cannot read"):
        document_text(body, content_type, PAGE_URL)


def test_the_unreadable_type_is_named_in_the_error() -> None:
    with pytest.raises(UnreadableDocumentError, match="image/png"):
        document_text(b"\x89PNG", "image/png", PAGE_URL)
