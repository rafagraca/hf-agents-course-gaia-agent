"""Tests for the shared fixtures in conftest.py: blocked network and hermetic environment."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import requests

from gaia_agent import config

BLOCKED = "blocked in tests"
TEST_NET_ADDRESS = "203.0.113.1"  # RFC 5737 documentation range: never routable


def test_the_network_guard_is_installed_by_default() -> None:
    assert socket.socket.connect.__name__ == "guarded_connect"
    assert socket.getaddrinfo.__name__ == "guarded_getaddrinfo"


def test_remote_connections_are_blocked() -> None:
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.create_connection((TEST_NET_ADDRESS, 80), timeout=1)


def test_remote_connect_ex_is_blocked() -> None:
    with socket.socket() as sock, pytest.raises(RuntimeError, match=BLOCKED):
        sock.connect_ex((TEST_NET_ADDRESS, 80))


def test_dns_lookups_are_blocked() -> None:
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.getaddrinfo("example.com", 80)


def test_http_libraries_cannot_swallow_the_block() -> None:
    with pytest.raises(RuntimeError, match=BLOCKED):
        requests.get("https://example.com", timeout=1)


def test_loopback_connections_stay_allowed() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            accepted, _ = server.accept()
            with accepted:
                client.sendall(b"ping")
                assert accepted.recv(4) == b"ping"


@pytest.mark.network
def test_the_network_marker_removes_the_guard() -> None:
    # Nothing is sent: it only checks that the socket module is left untouched.
    assert socket.socket.connect.__name__ == "connect"
    assert socket.getaddrinfo.__name__ == "getaddrinfo"


@pytest.mark.parametrize(
    "name",
    ["GROQ_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GEMINI_API_KEY", "GAIA_MODEL_ID", "GAIA_FALLBACK_MODELS"],
)
def test_secrets_and_overrides_of_the_machine_are_hidden(name: str) -> None:
    assert os.environ.get(name) is None
    assert config.get_secret(name) is None


def test_gaia_data_dir_points_at_the_temporary_folder(data_dir: Path, tmp_path: Path) -> None:
    assert os.environ["GAIA_DATA_DIR"] == str(data_dir)
    assert data_dir.parent == tmp_path


def test_settings_fixture_uses_the_temporary_folder(settings: config.Settings, data_dir: Path) -> None:
    assert settings.data_dir == data_dir.resolve()
    assert settings.cache_dir.is_dir()
