"""Shared pytest fixtures.

Every test runs hermetically:

* the real network is blocked unless the test is marked ``@pytest.mark.network``
  (loopback connections stay allowed);
* ``GAIA_DATA_DIR`` points at a throw-away folder, so no test can touch the real
  data directory;
* secrets and ``GAIA_*`` variables of the developer's machine, including the
  ones stored in the Windows user environment, are hidden.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gaia_agent import config
from gaia_agent.tools import web_urls

DATA_DIR_NAME = "gaia-data"
HIDDEN_ENV_VARS = (
    "GROQ_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GEMINI_API_KEY",
    "GAIA_MODEL_ID",
    "GAIA_FALLBACK_MODELS",
    "GAIA_GEMINI_MODEL",
    "GAIA_ALLOW_RUN_PYTHON",
)
PUBLIC_TEST_ADDRESS = "93.184.216.34"  # a public address: where every invented host name resolves in the tests


class NetworkBlockedError(RuntimeError):
    """A test tried to reach a non-local host without ``@pytest.mark.network``.

    Deliberately not an ``OSError``: HTTP libraries would wrap those into their
    own connection errors, which code under test may catch and hide.
    """


def _is_local_host(host: object) -> bool:
    """Return True for ``localhost`` and loopback IP literals."""
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if not isinstance(host, str):
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check_host(host: object) -> None:
    if host is not None and not _is_local_host(host):
        raise NetworkBlockedError(
            f"Network access to {host!r} is blocked in tests; mark the test with @pytest.mark.network to allow it."
        )


def _check_address(address: object) -> None:
    """Check a socket address; non-tuple addresses (AF_UNIX paths) are local."""
    if isinstance(address, tuple) and address:
        _check_host(address[0])


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Block outgoing connections and DNS lookups unless the test opts in."""
    if request.node.get_closest_marker("network") is not None:
        return

    real_connect: Callable[..., Any] = socket.socket.connect
    real_connect_ex: Callable[..., Any] = socket.socket.connect_ex
    real_getaddrinfo: Callable[..., Any] = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        _check_address(address)
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        _check_address(address)
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        _check_host(host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every host name resolves to one public address, so that fetching a page never needs a real DNS lookup.

    A test that cares about what a name resolves to patches ``web_urls.resolve_host`` again.
    """
    monkeypatch.setattr(web_urls, "resolve_host", lambda host, port: [PUBLIC_TEST_ADDRESS])


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the machine's secrets and redirect ``GAIA_DATA_DIR`` to a temp folder."""
    for name in HIDDEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GAIA_DATA_DIR", str(tmp_path / DATA_DIR_NAME))
    monkeypatch.setattr(config, "_persistent_env_reader", lambda name: None)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """The temporary DATA folder that ``GAIA_DATA_DIR`` points at (not created yet)."""
    return tmp_path / DATA_DIR_NAME


@pytest.fixture
def settings() -> config.Settings:
    """Settings built by ``load_settings()`` on top of the temporary DATA folder."""
    return config.load_settings()
