"""A ``requests`` session that refuses to talk to any peer that is not a public address.

The URL checks of :mod:`gaia_agent.tools.web_urls` look at text and at what a host name resolved to a moment
earlier. This is the last layer and the one an attacker cannot talk around: right after the socket connects, the
address of the peer it connected to is checked, and the connection is closed before a single byte of the request
is sent when that address is local or private. A name that resolves to a public address for the first lookup and
to 127.0.0.1 for the second (DNS rebinding) is stopped here.

The check runs where urllib3 creates the socket (``_new_conn``), so it covers HTTP and HTTPS and every new
connection, redirect hops included. Proxies are not supported (the session ignores the proxy variables): through
a proxy the peer would be the proxy, and the real destination could no longer be seen.
"""

from __future__ import annotations

import socket
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from gaia_agent.tools.web_urls import is_public_address

__all__ = ["BlockedAddressError", "GuardedAdapter", "guarded_session"]


class BlockedAddressError(Exception):
    """The connection went to a local or private address and was closed. The message is safe to show an agent.

    Deliberately not an ``OSError``: urllib3 retries and wraps those, which would bury the reason.
    """


def _require_public_peer(sock: socket.socket) -> None:
    """Close ``sock`` and raise :class:`BlockedAddressError` unless it is connected to a public address."""
    try:
        peer = str(sock.getpeername()[0])
    except OSError:
        peer = ""  # a peer that cannot be read cannot be shown to be public
    if not is_public_address(peer):
        sock.close()
        raise BlockedAddressError("local and private network addresses are not allowed: the connection went to one")


class _PeerCheck:
    """Mixin for urllib3 connection classes: check the peer of every socket they create."""

    def _new_conn(self) -> socket.socket:
        sock = super()._new_conn()  # type: ignore[misc]
        _require_public_peer(sock)
        return sock


class GuardedHTTPConnection(_PeerCheck, HTTPConnection):
    """A plain HTTP connection that only talks to public peers."""


class GuardedHTTPSConnection(_PeerCheck, HTTPSConnection):
    """An HTTPS connection that only talks to public peers (checked before the TLS handshake)."""


class _GuardedHTTPPool(HTTPConnectionPool):
    ConnectionCls = GuardedHTTPConnection


class _GuardedHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = GuardedHTTPSConnection


class GuardedAdapter(HTTPAdapter):
    """An adapter whose pools create guarded connections."""

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False, **pool_kwargs: Any) -> None:
        """Set up the pool manager as usual, then have it create guarded connection pools."""
        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)
        self.poolmanager.pool_classes_by_scheme = {"http": _GuardedHTTPPool, "https": _GuardedHTTPSPool}


def guarded_session() -> requests.Session:
    """A session that only connects to public addresses and ignores the proxy settings of the environment."""
    session = requests.Session()
    session.trust_env = False  # a proxy would hide the real destination from the guard (and bypass it)
    adapter = GuardedAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
