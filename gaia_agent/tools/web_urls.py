"""Which addresses the web tools may fetch: web URLs only, never the local or a private network.

A model that browses the web can be steered by what it reads, so every address the tools request, redirect hops
included, goes through three layers:

1. :func:`normalize_url` judges the *text*. It refuses local paths, other schemes, backslashes and control
   characters, and judges the host the way both the standard library and ``requests``/``urllib3`` read it (the
   two disagree about a backslash, and the connection follows ``urllib3``), so a URL cannot show one host to the
   check and another to the connection.
2. :func:`check_public_destination` resolves the host name and refuses it when any address it points to is not
   public (``localtest.me`` and ``127.0.0.1.nip.io`` both name 127.0.0.1).
3. ``gaia_agent.tools._guarded_http`` checks the address the socket really connected to, which also closes DNS
   rebinding (a name that answers differently the second time).
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable, Sequence
from urllib.parse import urlsplit

import requests
from urllib3.util import parse_url

__all__ = [
    "ALLOWED_SCHEMES",
    "UrlError",
    "check_public_destination",
    "is_public_address",
    "normalize_url",
    "request_hosts",
    "resolve_host",
]

ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_PORTS = {"http": 80, "https": 443}

_LOCAL_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/(?!/)|\\|~|\.{1,2}[\\/])")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):(?!\d+(?:/|$))")  # "host:8080/x" is not a scheme
_BACKSLASH_OR_CONTROL = re.compile(r"[\\\x00-\x1f\x7f]")
_PRIVATE_SUFFIXES = (".localhost", ".local", ".internal")

Resolver = Callable[[str, int], Sequence[str]]


class UrlError(ValueError):
    """The address cannot be fetched; the message is short and safe to show an agent."""


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse an IP literal, including the legacy IPv4 spellings (decimal, hex, short) resolvers accept."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            return ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def is_public_address(address: str) -> bool:
    """True for an IP address on the public internet; anything else, even text that is no address, is False."""
    parsed = _parse_ip(address.strip("[]"))
    return parsed is not None and parsed.is_global and not parsed.is_multicast


def _is_private_host(host: str) -> bool:
    """True for ``localhost``, internal-only names and IP literals that are not publicly routable."""
    name = host.strip("[]").rstrip(".").lower()
    if name == "localhost" or name.endswith(_PRIVATE_SUFFIXES):
        return True
    return _parse_ip(name) is not None and not is_public_address(name)


def _request_url(url: str) -> str:
    """``url`` as ``requests`` will send it: scheme lower-cased, path quoted, host IDNA-encoded."""
    prepared = requests.models.PreparedRequest()
    try:
        prepared.prepare_url(url, None)
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise UrlError("the URL is malformed") from exc
    if not prepared.url:
        raise UrlError("the URL is malformed")
    return prepared.url


def request_hosts(url: str) -> list[str]:
    """The host of ``url`` as each parser that handles it reads it: ``urlsplit`` and the ``urllib3`` connection.

    Normally one and the same host; for a hostile URL they differ, and every one of them must be judged.
    Raises :class:`UrlError` when ``requests`` cannot make sense of the address.
    """
    prepared = _request_url(url)
    candidates = [urlsplit(url).hostname, urlsplit(prepared).hostname, parse_url(prepared).host]
    return list(dict.fromkeys(host.strip("[]").lower() for host in candidates if host))


def normalize_url(raw: object) -> str:
    """Check that ``raw`` is a fetchable web URL and return it (``https://`` is added to bare hosts).

    Raises :class:`UrlError` for empty input, local file paths, schemes other than http and https, malformed
    addresses (backslashes and control characters included) and hosts on the local or a private network. It reads
    the text only; :func:`check_public_destination` is the check that resolves names.
    """
    url = str(raw or "").strip()
    if not url:
        raise UrlError("the URL is empty")
    if _LOCAL_PATH.match(url):
        raise UrlError("this looks like a local file path, not a web URL; use read_file for local files")
    if url.startswith("//"):
        url = "https:" + url
    scheme = _SCHEME.match(url)
    if scheme is None:
        url = "https://" + url
    elif scheme.group(1).lower() not in ALLOWED_SCHEMES:
        raise UrlError(f"only http:// and https:// URLs can be read (got '{scheme.group(1)}')")
    try:
        host = urlsplit(url).hostname
    except ValueError as exc:
        raise UrlError("the URL is malformed") from exc
    if not host:
        raise UrlError("the URL has no host name")
    if any(_is_private_host(name) for name in (host, *request_hosts(url))):
        raise UrlError("local and private network addresses are not allowed")
    if _BACKSLASH_OR_CONTROL.search(url):
        raise UrlError("the URL is malformed: it contains a backslash or a control character")
    return url


def _system_resolver(host: str, port: int) -> list[str]:
    """Every address the operating system resolves ``host`` to."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


# The seam the checks (and the tests) use to look a name up; call it through the module so that a patch is seen.
resolve_host: Resolver = _system_resolver


def check_public_destination(url: str) -> None:
    """Raise :class:`UrlError` unless every address the host name of ``url`` resolves to is public.

    ``url`` has been through :func:`normalize_url`. IP literals were judged there and are not looked up.
    One private address among several public ones is enough to refuse: the connection might pick it.
    """
    prepared = parse_url(_request_url(url))
    host = (prepared.host or "").strip("[]")
    if not host or _parse_ip(host) is not None:
        return
    port = prepared.port or DEFAULT_PORTS.get((prepared.scheme or "https").lower(), 443)
    try:
        addresses = list(resolve_host(host, port))
    except OSError as exc:
        raise UrlError("the host name could not be resolved") from exc
    if not addresses:
        raise UrlError("the host name could not be resolved")
    if not all(is_public_address(address) for address in addresses):
        raise UrlError("local and private network addresses are not allowed: the host name resolves to one")
