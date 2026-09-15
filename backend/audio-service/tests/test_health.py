from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.main import ApplicationDependencies, LifespanDependencies, create_app


def test_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "MULTI-SCOPE Voice Classification API",
        "version": "0.1.0",
        "status": "running",
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_readiness_reports_ffmpeg_tool_availability(client: TestClient) -> None:
    response = client.get("/readiness")
    payload = response.json()

    assert response.status_code in {200, 503}
    assert payload["status"] in {"ready", "not_ready"}
    assert isinstance(payload["ffmpeg_available"], bool)
    assert isinstance(payload["ffprobe_available"], bool)
    assert isinstance(payload["mongodb_configured"], bool)
    assert isinstance(payload["mongodb_available"], bool)
    assert isinstance(payload["storage_enabled"], bool)
    assert isinstance(payload["storage_available"], bool)


def test_ready_reports_backend_dependency_readiness(client: TestClient) -> None:
    response = client.get("/ready")
    payload = response.json()

    assert response.status_code in {200, 503}
    assert payload["status"] in {"ready", "not_ready"}
    assert isinstance(payload["ffmpeg_available"], bool)
    assert isinstance(payload["ffprobe_available"], bool)
    assert isinstance(payload["mongodb_configured"], bool)
    assert isinstance(payload["mongodb_available"], bool)
    assert isinstance(payload["storage_enabled"], bool)
    assert isinstance(payload["storage_available"], bool)
    assert (response.status_code == 200) is (
        payload["ffmpeg_available"]
        and payload["ffprobe_available"]
        and payload["mongodb_configured"]
        and payload["mongodb_available"]
        and payload["storage_available"]
    )


def test_create_app_accepts_injected_lifecycle_and_readiness_dependencies(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.main.audio_tool_status",
        lambda: {"ffmpeg_available": True, "ffprobe_available": True},
    )
    runner = FakeRunner()
    app = create_app(
        Settings(
            _env_file=None,
            app_name="Injected Test API",
            app_env="test",
            cloudinary_storage_enabled=False,
        ),
        lifespan_dependencies=fake_lifespan_dependencies(),
        dependencies=ApplicationDependencies(
            job_runner=runner,
            storage=FakeStorage(),
            voice_service=FakeVoiceService(),
        ),
    )

    with TestClient(app) as client:
        root = client.get("/")
        ready = client.get("/ready")

    assert runner.shutdown_calls == [True]
    assert root.json()["name"] == "Injected Test API"
    assert ready.status_code == 200
    assert ready.json()["prediction_ready"] is True
    assert ready.json()["research_ready"] is False
    assert ready.json()["components"]["models"]["dummy_mode_allowed"] is True


def test_research_environment_reports_dummy_models_not_prediction_ready(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.main.audio_tool_status",
        lambda: {"ffmpeg_available": True, "ffprobe_available": True},
    )
    app = create_app(
        Settings(
            _env_file=None,
            app_env="research",
            debug=False,
            mongodb_uri="mongodb://localhost:27017",
            clerk_issuer="https://example.clerk.accounts.dev",
            clerk_jwks_url="https://example.clerk.accounts.dev/.well-known/jwks.json",
            clerk_audience="multi-scope",
            clerk_authorized_parties="https://dashboard.example",
            cloudinary_cloud_name="demo-cloud",
            cloudinary_api_key="demo-key",
            cloudinary_api_secret="demo-secret",
            allowed_origins="https://dashboard.example",
            trusted_hosts="testserver",
        ),
        lifespan_dependencies=fake_lifespan_dependencies(),
        dependencies=ApplicationDependencies(
            job_runner=FakeRunner(),
            storage=FakeStorage(),
            voice_service=FakeVoiceService(),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["prediction_ready"] is False
    assert response.json()["components"]["models"]["dummy_mode_allowed"] is False


def test_runner_shutdown_makes_readiness_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main.audio_tool_status",
        lambda: {"ffmpeg_available": True, "ffprobe_available": True},
    )
    runner = FakeRunner(shutting_down=True)
    app = create_app(
        Settings(_env_file=None, app_env="test", cloudinary_storage_enabled=False),
        lifespan_dependencies=fake_lifespan_dependencies(),
        dependencies=ApplicationDependencies(
            job_runner=runner,
            storage=FakeStorage(),
            voice_service=FakeVoiceService(),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["components"]["prediction_runner"]["shutting_down"] is True


def fake_lifespan_dependencies() -> LifespanDependencies:
    async def connect(_settings):
        return None

    async def close():
        return None

    async def mongodb(_settings):
        return {"mongodb_configured": True, "mongodb_available": True}

    async def storage(_settings):
        return {"storage_enabled": True, "storage_available": True}

    return LifespanDependencies(
        connect_mongodb=connect,
        close_mongodb=close,
        mongodb_readiness=mongodb,
        storage_readiness=storage,
    )


class FakeRunner:
    def __init__(self, *, shutting_down: bool = False) -> None:
        self.shutting_down = shutting_down
        self.shutdown_calls = []

    def health(self):
        return {
            "ready": not self.shutting_down,
            "accepting_jobs": not self.shutting_down,
            "shutting_down": self.shutting_down,
            "active_jobs": 0,
            "max_concurrent_jobs": 2,
            "available_capacity": 2,
            "executor_healthy": True,
        }

    async def shutdown(self, *, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


class FakeVoiceService:
    def model_health(self):
        return [
            {
                "model_name": "cnn_acoustic",
                "display_name": "CNN Acoustic",
                "mode": "dummy",
                "is_loaded": False,
            }
        ]


class FakeStorage:
    async def health(self):
        return {"storage_enabled": True, "storage_available": True}
