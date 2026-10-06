"""Tests for the connection guard: a ``requests`` session that refuses to talk to a peer that is not public.

The URL checks look at text and at DNS answers; this guard looks at the address the socket really connected to,
which is the only check a DNS-rebinding trick cannot get past. Nothing here leaves the machine: the fake sockets
are objects, and the one real connection goes to a throw-away server on the loopback interface.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator

import pytest
import urllib3.connection
import urllib3.util.connection

from gaia_agent.tools import _guarded_http as guarded
from gaia_agent.tools.web import WebToolError, fetch_page

PUBLIC_PEER = ("93.184.216.34", 443)


class FakeSocket:
    """Stands in for a connected socket: it only knows its peer and whether it was closed."""

    def __init__(self, peer: tuple[object, ...] | Exception) -> None:
        self._peer = peer
        self.closed = False

    def getpeername(self) -> tuple[object, ...]:
        if isinstance(self._peer, Exception):
            raise self._peer
        return self._peer

    def close(self) -> None:
        self.closed = True


def connecting_to(monkeypatch: pytest.MonkeyPatch, peer: tuple[object, ...] | Exception) -> FakeSocket:
    """Make the plain urllib3 connection "connect" to ``peer``; return the socket it will hand over."""
    sock = FakeSocket(peer)
    monkeypatch.setattr(urllib3.connection.HTTPConnection, "_new_conn", lambda self: sock)
    return sock


@pytest.mark.parametrize("connection_class", [guarded.GuardedHTTPConnection, guarded.GuardedHTTPSConnection])
def test_a_connection_to_a_public_address_goes_through(
    monkeypatch: pytest.MonkeyPatch, connection_class: type[urllib3.connection.HTTPConnection]
) -> None:
    sock = connecting_to(monkeypatch, PUBLIC_PEER)

    assert connection_class("example.org")._new_conn() is sock
    assert not sock.closed


@pytest.mark.parametrize("connection_class", [guarded.GuardedHTTPConnection, guarded.GuardedHTTPSConnection])
@pytest.mark.parametrize(
    "peer",
    [
        ("127.0.0.1", 80),
        ("10.1.2.3", 80),
        ("169.254.169.254", 80),
        ("192.168.0.9", 443),
        ("::1", 443, 0, 0),
        ("fe80::1", 443, 0, 0),
        ("::ffff:127.0.0.1", 443, 0, 0),
    ],
)
def test_a_connection_to_a_private_address_is_closed_and_refused(
    monkeypatch: pytest.MonkeyPatch, connection_class: type[urllib3.connection.HTTPConnection], peer: tuple[object, ...]
) -> None:
    sock = connecting_to(monkeypatch, peer)

    with pytest.raises(guarded.BlockedAddressError, match="private"):
        connection_class("example.org")._new_conn()

    assert sock.closed


def test_a_peer_that_cannot_be_read_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    sock = connecting_to(monkeypatch, OSError("not connected"))

    with pytest.raises(guarded.BlockedAddressError):
        guarded.GuardedHTTPConnection("example.org")._new_conn()

    assert sock.closed


def test_the_refusal_is_not_an_os_error_so_urllib3_cannot_retry_or_wrap_it() -> None:
    assert not issubclass(guarded.BlockedAddressError, OSError)


def test_the_session_uses_the_guard_for_both_schemes_and_ignores_proxy_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")

    with guarded.guarded_session() as session:
        assert isinstance(session.get_adapter("http://example.org/"), guarded.GuardedAdapter)
        assert isinstance(session.get_adapter("https://example.org/"), guarded.GuardedAdapter)
        assert session.trust_env is False  # a proxy would hide the real destination from the guard
        pools = session.get_adapter("https://example.org/").poolmanager.pool_classes_by_scheme
        assert pools["http"].ConnectionCls is guarded.GuardedHTTPConnection
        assert pools["https"].ConnectionCls is guarded.GuardedHTTPSConnection


# --- a real connection that DNS rebinding would send to the loopback interface -----------------------------------


class _Recorder(http.server.BaseHTTPRequestHandler):
    requests_seen: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - the name http.server expects
        type(self).requests_seen.append(self.path)
        body = b"internal secret"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the signature http.server expects
        pass


@pytest.fixture
def internal_server() -> Iterator[tuple[int, list[str]]]:
    """A server on the loopback interface that records the requests it gets."""
    handler = type("Recorder", (_Recorder,), {"requests_seen": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], handler.requests_seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_name_that_passes_the_dns_check_but_connects_to_loopback_is_stopped_before_any_request(
    monkeypatch: pytest.MonkeyPatch, internal_server: tuple[int, list[str]]
) -> None:
    port, requests_seen = internal_server
    real_create_connection = urllib3.util.connection.create_connection

    def rebind(address: tuple[str, int], *args: object, **kwargs: object):
        return real_create_connection(("127.0.0.1", address[1]), *args, **kwargs)

    # The conftest resolver says the name is public (the first lookup of a rebinding attack); the connection,
    # which resolves again, lands on the loopback interface (the second lookup).
    monkeypatch.setattr(urllib3.util.connection, "create_connection", rebind)

    with pytest.raises(WebToolError, match="private"):
        fetch_page(f"http://rebinding.example:{port}/admin")

    assert requests_seen == []
