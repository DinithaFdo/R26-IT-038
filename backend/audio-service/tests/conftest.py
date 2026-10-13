from collections.abc import Generator
from hashlib import sha256
import json
import os
from pathlib import Path
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


@pytest.fixture
def fidelity_artifact_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use synthetic validation data without requiring a local research export."""
    from app.voice_xai.evaluation import offline_metrics

    source = tmp_path / "fixture_fidelity.csv"
    source.write_bytes(b"dataset,r2\nfixture_dataset,0.75\n")
    path = tmp_path / "surrogate_fidelity.json"
    path.write_text(
        json.dumps(
            {
                "artifact_version": "voice-xai-surrogate-fidelity-r2-v1",
                "metric_name": "surrogate_fidelity_r2",
                "value": 0.75,
                "scope": "offline_validation",
                "dataset_version": "fixture_dataset",
                "sample_count": 4,
                "semantic_model_version": "fixture-surrogate-v1",
                "source": {
                    "artifact_name": source.name,
                    "sha256": sha256(source.read_bytes()).hexdigest(),
                    "dataset_column": "dataset",
                    "dataset_value": "fixture_dataset",
                    "metric_column": "r2",
                },
            }
        ),
        encoding="utf-8",
    )
    loader = offline_metrics.load_surrogate_fidelity_artifact
    monkeypatch.setattr(
        offline_metrics, "load_surrogate_fidelity_artifact", lambda: loader(path)
    )
    return path


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
