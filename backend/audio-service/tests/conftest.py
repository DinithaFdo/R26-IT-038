from collections.abc import Generator
import os
import socket

from fastapi.testclient import TestClient
import pytest

os.environ.setdefault("MULTISCOPE_DISABLE_DOTENV", "1")

from app.api.dependencies import reset_dependency_caches
from app.auth.clerk_auth import reset_jwks_cache
from app.main import app


@pytest.fixture(autouse=True)
def isolate_process_state(monkeypatch: pytest.MonkeyPatch, request) -> Generator[None, None, None]:
    if not _allows_network(request):
        _block_network(monkeypatch)
    app.dependency_overrides.clear()
    reset_dependency_caches()
    reset_jwks_cache()
    yield
    app.dependency_overrides.clear()
    reset_dependency_caches()
    reset_jwks_cache()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def _allows_network(request) -> bool:
    return any(
        request.node.get_closest_marker(marker) is not None
        for marker in {"integration", "network", "mongodb", "cloudinary"}
    )


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def fail_connect(*_args, **_kwargs):
        raise AssertionError(
            "External network access is disabled in default tests. "
            "Mark the test with integration/network if it intentionally uses it."
        )

    def is_loopback(address: object) -> bool:
        return (
            isinstance(address, tuple)
            and bool(address)
            and address[0] in {"127.0.0.1", "::1"}
        )

    def guarded_connect(sock: socket.socket, address: object):
        if is_loopback(address):
            return original_connect(sock, address)
        return fail_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: object):
        if is_loopback(address):
            return original_connect_ex(sock, address)
        return fail_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", fail_connect)
