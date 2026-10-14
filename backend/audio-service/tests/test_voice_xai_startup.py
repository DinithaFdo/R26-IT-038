"""Real semantic initialization must not fall back to generated evidence."""

from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.main import ApplicationDependencies, LifespanDependencies, create_app, lifespan
from app.voice_xai.semantic.service import ProductionSemanticExplanationService


@pytest.mark.anyio
@pytest.mark.parametrize("artifact_failure", [False, True])
async def test_default_orchestrator_loads_real_service_or_rejects_startup(
    monkeypatch, tmp_path, artifact_failure
):
    service = object()
    calls = []

    async def no_connect(_settings):
        pass

    async def no_close():
        pass

    async def no_recovery(_app, _settings):
        pass

    def load_once(**paths):
        calls.append(paths)
        if artifact_failure:
            raise ValueError("Incompatible semantic artifacts")
        return service

    monkeypatch.setattr("app.main._recover_stale_xai_runs", no_recovery)
    monkeypatch.setattr(ProductionSemanticExplanationService, "load_once", load_once)
    app = create_app(
        Settings(
            _env_file=None, xai_enabled=True, xai_mode="real",
            xai_artifact_root=str(tmp_path / "private"),
        ),
        dependencies=ApplicationDependencies(voice_service=SimpleNamespace()),
        lifespan_dependencies=LifespanDependencies(
            connect_mongodb=no_connect, close_mongodb=no_close,
        ),
    )
    assert app.state.xai_orchestrator.readiness()["ready"] is False
    if artifact_failure:
        with pytest.raises(ValueError, match="Incompatible semantic artifacts"):
            async with lifespan(app):
                pytest.fail("Startup should reject invalid real artifacts")
        assert app.state.xai_orchestrator.readiness()["ready"] is False
    else:
        async with lifespan(app):
            assert app.state.xai_orchestrator._semantic_service is service
            assert app.state.xai_orchestrator.readiness()["ready"] is True
    assert len(calls) == 1
    assert calls[0]["manifest_path"].name == "xgboost-surrogate-v4.json"
