"""Tests for fetching pages: failures, redirects, the size limit and what is refused. No network."""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Iterator

import pytest
import requests
import responses
from web_samples import PAGE_URL, html_page, http, make_pdf, paragraphs, read, serve  # noqa: F401

from gaia_agent.tools import web, web_urls
from gaia_agent.tools.web import WebToolError


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("", "empty"),
        ("ftp://example.org/file", "only http"),
        ("/home/user/file.txt", "read_file"),
        ("https://[::1", "malformed"),
        ("http://127.0.0.1:8000/", "private"),
        ("http://169.254.169.254/latest/meta-data/", "private"),
    ],
)
def test_addresses_that_may_not_be_fetched_are_refused_without_a_request(
    http: responses.RequestsMock, url: str, message: str
) -> None:
    result = read(url)

    assert result.startswith("Error:")
    assert message in result
    assert len(http.calls) == 0


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (requests.exceptions.ConnectTimeout("slow"), "timed out after 30 s"),
        (requests.exceptions.ReadTimeout("slow"), "timed out after 30 s"),
        (requests.exceptions.SSLError("bad certificate"), "TLS/SSL"),
        (requests.exceptions.ConnectionError("refused"), "could not connect (ConnectionError)"),
        (requests.exceptions.InvalidURL("bad"), "the request failed (InvalidURL)"),
    ],
)
def test_connection_failures_become_short_messages(
    http: responses.RequestsMock, failure: Exception, message: str
) -> None:
    http.add(responses.GET, PAGE_URL, body=failure)

    result = read()

    assert result.startswith("Error:")
    assert message in result
    assert "\n" not in result


def test_a_public_ip_address_is_allowed(http: responses.RequestsMock) -> None:
    url = "http://93.184.216.34/page"
    serve(http, "public text", content_type="text/plain", url=url)

    assert read(url) == "public text"


def test_a_url_without_a_scheme_is_read_over_https(http: responses.RequestsMock) -> None:
    serve(http, "bare host text", content_type="text/plain", url="https://example.org/dir/page.html")

    assert read("example.org/dir/page.html") == "bare host text"
    assert read("//example.org/dir/page.html") == "bare host text"


def test_redirects_are_followed_and_links_resolve_against_the_final_page(http: responses.RequestsMock) -> None:
    http.add(responses.GET, "https://example.org/old", status=301, headers={"Location": "/new/index.html"})
    serve(http, html_page('<p><a href="other">Other</a></p>'), url="https://example.org/new/index.html")

    result = read("https://example.org/old")

    assert "[Other](https://example.org/new/other)" in result
    assert [call.request.url for call in http.calls] == [
        "https://example.org/old",
        "https://example.org/new/index.html",
    ]


def test_a_redirect_to_a_private_address_is_refused_before_it_is_followed(http: responses.RequestsMock) -> None:
    http.add(
        responses.GET, "https://example.org/go", status=302, headers={"Location": "http://169.254.169.254/latest/"}
    )

    result = read("https://example.org/go")

    assert result.startswith("Error:")
    assert "private" in result
    assert len(http.calls) == 1


def resolve_names(monkeypatch: pytest.MonkeyPatch, table: dict[str, list[str]]) -> list[str]:
    """Make host names resolve as in ``table`` (public addresses for the others); return the names looked up."""
    looked_up: list[str] = []

    def resolve(host: str, port: int) -> list[str]:
        looked_up.append(host)
        return table.get(host, ["93.184.216.34"])

    monkeypatch.setattr(web_urls, "resolve_host", resolve)
    return looked_up


def test_a_name_that_resolves_to_a_private_address_is_refused_without_a_request(
    http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolve_names(monkeypatch, {"localtest.example": ["127.0.0.1"]})

    result = read("http://localtest.example/admin")

    assert result.startswith("Error:")
    assert "private" in result
    assert len(http.calls) == 0


def test_a_redirect_to_a_name_that_resolves_to_a_private_address_is_refused_before_it_is_followed(
    http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolve_names(monkeypatch, {"intranet.example": ["10.0.0.5"]})
    http.add(responses.GET, "https://example.org/go", status=302, headers={"Location": "https://intranet.example/x"})

    result = read("https://example.org/go")

    assert result.startswith("Error:")
    assert "private" in result
    assert len(http.calls) == 1


def test_a_redirect_that_hides_a_private_host_behind_a_backslash_is_refused(http: responses.RequestsMock) -> None:
    location = "http://127.0.0.1:11434\\@example.org/../api/tags"
    http.add(responses.GET, "https://example.org/go", status=302, headers={"Location": location})

    result = read("https://example.org/go")

    assert result.startswith("Error:")
    assert len(http.calls) == 1


def test_every_hop_is_looked_up_before_it_is_requested(
    http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    looked_up = resolve_names(monkeypatch, {})
    http.add(responses.GET, "https://example.org/old", status=301, headers={"Location": "https://cdn.example.net/new"})
    serve(http, "moved text", content_type="text/plain", url="https://cdn.example.net/new")

    assert read("https://example.org/old") == "moved text"
    assert looked_up == ["example.org", "cdn.example.net"]


def test_a_name_that_does_not_resolve_is_reported(
    http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(host: str, port: int) -> list[str]:
        raise socket.gaierror("no such host")

    monkeypatch.setattr(web_urls, "resolve_host", fail)

    assert read("https://nowhere.example/") == "Error: the host name could not be resolved"
    assert len(http.calls) == 0


def test_a_redirect_to_another_scheme_is_refused(http: responses.RequestsMock) -> None:
    http.add(responses.GET, "https://example.org/go", status=302, headers={"Location": "ftp://example.org/file"})

    assert read("https://example.org/go").startswith("Error: only http")


def test_a_malformed_redirect_address_is_reported(http: responses.RequestsMock) -> None:
    http.add(responses.GET, "https://example.org/go", status=302, headers={"Location": "http://[::1"})

    assert read("https://example.org/go") == "Error: the URL or a redirect address is malformed"


def test_a_redirect_loop_is_cut_short(http: responses.RequestsMock) -> None:
    http.add(responses.GET, "https://example.org/loop", status=302, headers={"Location": "/loop"})

    result = read("https://example.org/loop")

    assert result == f"Error: too many redirects (more than {web.MAX_REDIRECTS})"
    assert len(http.calls) == web.MAX_REDIRECTS + 1


def test_cookies_set_by_a_redirect_are_sent_on_the_next_request(http: responses.RequestsMock) -> None:
    http.add(
        responses.GET,
        "https://example.org/gate",
        status=302,
        headers={"Location": "/page", "Set-Cookie": "consent=yes; Path=/"},
    )
    serve(http, "welcome", content_type="text/plain", url="https://example.org/page")

    read("https://example.org/gate")

    assert "consent=yes" in http.calls[1].request.headers.get("Cookie", "")


def test_a_page_bigger_than_the_limit_is_cut_and_the_cut_is_reported(
    http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web, "MAX_DOWNLOAD_BYTES", 3000)
    serve(http, html_page(paragraphs(40)))

    result = read()

    assert "Paragraph 0." in result
    assert "Paragraph 39." not in result
    assert result.endswith("[the page is larger than 3000 bytes: only the start was read]")


def test_a_pdf_bigger_than_the_limit_cannot_be_read(
    http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web, "MAX_DOWNLOAD_BYTES", 500)
    url = "https://example.org/big.pdf"
    serve(http, make_pdf([["Page text " * 100]]), content_type="application/pdf", url=url)

    result = read(url)

    assert result.startswith("Error:")
    assert "larger than the download limit" in result


def test_the_limit_is_five_megabytes() -> None:
    assert web.MAX_DOWNLOAD_BYTES == 5 * 1024 * 1024


class _FakeResponse:
    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        yield from self._chunks
        if self._error is not None:
            raise self._error


def test_reading_stops_at_the_size_limit() -> None:
    body, cut = web.read_limited(_FakeResponse([b"a" * 600] * 5), max_bytes=1000)

    assert body == b"a" * 1000
    assert cut is True


def test_a_body_exactly_at_the_limit_is_not_cut() -> None:
    body, cut = web.read_limited(_FakeResponse([b"a" * 500, b"b" * 500]), max_bytes=1000)

    assert len(body) == 1000
    assert cut is False


def test_a_download_that_takes_too_long_is_abandoned() -> None:
    ticks = iter([0.0, 10.0, 200.0, 300.0])
    clock: Callable[[], float] = lambda: next(ticks)  # noqa: E731

    with pytest.raises(WebToolError, match="too long"):
        web.read_limited(_FakeResponse([b"x"] * 5), max_bytes=10**6, deadline_s=60, clock=clock)


def test_a_connection_that_breaks_mid_download_is_reported() -> None:
    response = _FakeResponse([b"partial"], error=requests.exceptions.ChunkedEncodingError("reset"))

    with pytest.raises(WebToolError, match=r"download failed \(ChunkedEncodingError\)"):
        web.read_limited(response, max_bytes=1000)


def test_an_unexpected_failure_becomes_a_message_and_is_logged(
    http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    serve(http, html_page("<p>fine</p>"))

    def explode(*args: object, **kwargs: object) -> str:
        raise ValueError("converter bug")

    monkeypatch.setattr(web, "document_text", explode)

    with caplog.at_level(logging.ERROR, logger="gaia_agent.tools.web"):
        result = read()

    assert result == "Error: unexpected failure while reading the page (ValueError)"
    assert "converter bug" in caplog.text
