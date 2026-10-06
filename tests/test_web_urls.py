"""Tests for the URL policy of the web tools: which addresses may be fetched."""

from __future__ import annotations

import socket

import pytest
import requests

from gaia_agent.tools import web_urls
from gaia_agent.tools.web_urls import UrlError, normalize_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  example.org/a ", "https://example.org/a"),
        ("example.org:8080/a", "https://example.org:8080/a"),
        ("//example.org/a", "https://example.org/a"),
        ("HTTP://example.org/x", "HTTP://example.org/x"),
        ("https://example.org/x?q=1#top", "https://example.org/x?q=1#top"),
        ("https://user:secret@example.org/", "https://user:secret@example.org/"),
    ],
)
def test_urls_are_normalised_before_they_are_requested(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/",
        "http://93.184.216.34/page",
        "http://8.8.8.8/",
        "https://[2606:4700:4700::1111]/",
        "http://localhost.example.org/",
        "https://internal.example.org/",
    ],
)
def test_public_addresses_are_accepted(url: str) -> None:
    assert normalize_url(url) == url


@pytest.mark.parametrize("url", ["", "   ", None])
def test_an_empty_url_is_rejected(url: object) -> None:
    with pytest.raises(UrlError, match="empty"):
        normalize_url(url)


@pytest.mark.parametrize(
    ("url", "scheme"),
    [
        ("ftp://example.org/file", "ftp"),
        ("file:///etc/hosts", "file"),
        ("data:text/plain,hi", "data"),
        ("mailto:a@example.org", "mailto"),
        ("javascript:alert(1)", "javascript"),
    ],
)
def test_only_http_and_https_urls_are_accepted(url: str, scheme: str) -> None:
    with pytest.raises(UrlError, match=f"only http.*'{scheme}'"):
        normalize_url(url)


@pytest.mark.parametrize(
    "path",
    [
        "C:\\data\\file.txt",
        "c:/data/file.txt",
        "/home/user/file.txt",
        "\\\\server\\share\\f.txt",
        "./file.txt",
        "../file.txt",
        "~/file.txt",
    ],
)
def test_local_paths_are_pointed_to_the_file_tool(path: str) -> None:
    with pytest.raises(UrlError, match="local file path.*read_file"):
        normalize_url(path)


@pytest.mark.parametrize(
    ("url", "message"),
    [("https://[::1", "malformed"), ("http:///only/a/path", "no host name"), ("https://", "no host name")],
)
def test_malformed_urls_are_explained(url: str, message: str) -> None:
    with pytest.raises(UrlError, match=message):
        normalize_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://LocalHost:8000/",
        "http://localhost./",
        "http://127.0.0.1:8000/",
        "http://127.1/",
        "http://10.1.2.3/",
        "http://172.16.5.4/",
        "http://192.168.0.10/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.64.0.1/",
        "http://0.0.0.0/",
        "http://224.0.0.1/",
        "http://[::1]/",
        "http://[::]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:10.0.0.1]/",
        "http://[fe80::1]/",
        "http://[fd00::1]/",
        "http://2130706433/",
        "http://0x7f.0.0.1/",
        "http://017700000001/",
        "http://service.internal/",
        "http://printer.local/",
        "http://app.localhost/",
        "https://user:secret@127.0.0.1/",
    ],
)
def test_local_and_private_addresses_are_refused(url: str) -> None:
    with pytest.raises(UrlError, match="private"):
        normalize_url(url)


# --- the address the request really goes to ---------------------------------------------------------------------
#
# ``urlsplit`` (used to judge a URL) and ``urllib3`` (used to connect) disagree about a backslash: for
# ``http://127.0.0.1:11434\@x/../api/tags`` the first sees the host ``x`` while the second connects to 127.0.0.1.
# The judge must look at the address the way the connection will.

PARSER_DIFFERENTIALS = [
    "http://127.0.0.1:11434\\@x/../api/tags",
    "http://169.254.169.254\\@example.org/latest/meta-data/",
    "http://192.168.1.1:8080\\@example.org/",
    "http://localhost:8000\\@example.org/admin",
    "https://[::1]:8443\\@example.org/",
    "http://x\\@127.0.0.1/",
    "http://example.org\\.127.0.0.1/",
]


@pytest.mark.parametrize("url", PARSER_DIFFERENTIALS[:4])
def test_a_backslash_cannot_hide_the_real_host_from_the_check(url: str) -> None:
    with pytest.raises(UrlError, match="private"):
        normalize_url(url)


@pytest.mark.parametrize("url", PARSER_DIFFERENTIALS)
def test_a_url_with_a_backslash_is_never_accepted(url: str) -> None:
    with pytest.raises(UrlError):
        normalize_url(url)


def test_the_hosts_are_read_the_way_the_connection_will_read_them() -> None:
    hosts = web_urls.request_hosts("http://127.0.0.1:11434\\@x/../api/tags")

    assert "127.0.0.1" in hosts  # what urllib3 connects to
    assert "x" in hosts  # what urlsplit sees: both views are kept, and both are judged


@pytest.mark.parametrize("url", ["https://example.org/a\tb", "https://example.org/a\nb", "https://exa\x00mple.org/"])
def test_control_characters_are_refused(url: str) -> None:
    with pytest.raises(UrlError, match="malformed|control"):
        normalize_url(url)


IDEOGRAPHIC_FULL_STOP = chr(0x3002)  # looks like a dot; some resolvers read it as one
TWO_DOT_LEADER = chr(0x2025)


@pytest.mark.parametrize(
    "url",
    [
        "https://exa mple.org/",
        f"http://127{IDEOGRAPHIC_FULL_STOP}0{IDEOGRAPHIC_FULL_STOP}0{IDEOGRAPHIC_FULL_STOP}1/",
        f"http://ex{TWO_DOT_LEADER}ample.org/",
    ],
)
def test_hosts_that_requests_cannot_even_parse_are_refused(url: str) -> None:
    with pytest.raises(UrlError, match="malformed"):
        normalize_url(url)


def test_a_space_in_the_path_is_fine_because_requests_encodes_it() -> None:
    assert normalize_url("https://example.org/a b?q=x y") == "https://example.org/a b?q=x y"


# --- what a host name resolves to -------------------------------------------------------------------------------


def resolving_to(monkeypatch: pytest.MonkeyPatch, *addresses: str) -> list[tuple[str, int]]:
    """Make every host name resolve to ``addresses``; return the list that records the lookups."""
    lookups: list[tuple[str, int]] = []

    def resolve(host: str, port: int) -> list[str]:
        lookups.append((host, port))
        return list(addresses)

    monkeypatch.setattr(web_urls, "resolve_host", resolve)
    return lookups


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "169.254.169.254",
        "10.0.0.7",
        "192.168.1.20",
        "100.64.0.9",
        "::1",
        "fe80::1",
        "fd00::5",
        "::ffff:127.0.0.1",
    ],
)
def test_a_name_that_resolves_to_a_private_address_is_refused(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    resolving_to(monkeypatch, address)

    with pytest.raises(UrlError, match="private"):
        web_urls.check_public_destination("http://innocent.example/page")


def test_the_system_resolver_gives_every_address_of_every_family(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:4700:4700::1111", 443, 0, 0)),
    ]
    asked: list[tuple[object, ...]] = []

    def getaddrinfo(host: str, port: int, **kwargs: object) -> list[tuple[object, ...]]:
        asked.append((host, port, kwargs))
        return answers

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    assert web_urls._system_resolver("example.org", 443) == ["93.184.216.34", "2606:4700:4700::1111"]
    assert asked == [("example.org", 443, {"type": socket.SOCK_STREAM})]


def test_an_address_that_requests_leaves_without_a_url_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests.models.PreparedRequest, "prepare_url", lambda self, url, params: None)

    with pytest.raises(UrlError, match="malformed"):
        normalize_url("https://example.org/")


def test_one_private_address_among_public_ones_is_enough_to_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    resolving_to(monkeypatch, "93.184.216.34", "127.0.0.1")

    with pytest.raises(UrlError, match="private"):
        web_urls.check_public_destination("https://innocent.example/")


def test_a_name_that_resolves_only_to_public_addresses_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = resolving_to(monkeypatch, "93.184.216.34", "2606:4700:4700::1111")

    web_urls.check_public_destination("https://innocent.example/page")

    assert lookups == [("innocent.example", 443)]


def test_the_port_of_the_url_is_the_one_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = resolving_to(monkeypatch, "93.184.216.34")

    web_urls.check_public_destination("http://innocent.example:8080/")
    web_urls.check_public_destination("http://innocent.example/")

    assert lookups == [("innocent.example", 8080), ("innocent.example", 80)]


def test_an_ip_address_is_not_looked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = resolving_to(monkeypatch, "127.0.0.1")

    web_urls.check_public_destination("http://93.184.216.34/page")

    assert lookups == []


def test_a_name_that_does_not_resolve_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(host: str, port: int) -> list[str]:
        raise socket.gaierror("no such host")

    monkeypatch.setattr(web_urls, "resolve_host", fail)

    with pytest.raises(UrlError, match="could not be resolved"):
        web_urls.check_public_destination("https://nowhere.example/")


def test_an_empty_answer_from_the_resolver_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    resolving_to(monkeypatch)

    with pytest.raises(UrlError, match="could not be resolved"):
        web_urls.check_public_destination("https://nowhere.example/")


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.216.34", True),
        ("8.8.8.8", True),
        ("2606:4700:4700::1111", True),
        ("127.0.0.1", False),
        ("0.0.0.0", False),
        ("169.254.169.254", False),
        ("172.16.0.1", False),
        ("224.0.0.1", False),
        ("::1", False),
        ("::", False),
        ("fe80::1%eth0", False),
        ("::ffff:10.0.0.1", False),
        ("not an address", False),
        ("", False),
    ],
)
def test_which_addresses_count_as_public(address: str, public: bool) -> None:
    assert web_urls.is_public_address(address) is public
