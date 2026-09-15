"""Freezes the REST contract the frontend is built against.

If a route path, method, or status code listed here changes, the frontend's
hand-maintained types (frontend/src/types/api.ts) and API client
(frontend/src/lib/api/*) must be updated in the same change. This test does
not validate response bodies (see the route-specific test modules for that);
it only guards the shape of the OpenAPI surface itself.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_openapi_schema_is_serialisable(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    # Round-trips cleanly, i.e. every field is JSON-serialisable.
    json.dumps(schema)
    assert schema["openapi"].startswith("3.")


def test_clerk_bearer_authentication_is_described_for_swagger(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["components"]["securitySchemes"]["ClerkBearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": (
            "Paste a Clerk session JWT in Swagger's Authorize dialog. Send it as "
            "`Authorization: Bearer <token>`."
        ),
    }
    assert schema["paths"]["/api/v1/predictions"]["post"]["security"] == [
        {"ClerkBearerAuth": []}
    ]
    assert "parameters" not in schema["paths"]["/api/v1/predictions"]["post"]


FROZEN_ROUTES: dict[str, set[str]] = {
    "/health": {"get"},
    "/ready": {"get"},
    "/api/v1/predictions": {"post"},
    "/api/v1/predictions/{prediction_id}/status": {"get"},
    "/api/v1/me/predictions": {"get"},
    "/api/v1/me/predictions/{prediction_id}": {"get", "delete"},
    "/api/v1/me/predictions/{prediction_id}/audio": {"get"},
    "/api/v1/me/predictions/{prediction_id}/rerun": {"post"},
    "/api/v1/voice/models/health": {"get"},
    "/api/v1/external/predictions": {"post"},
    "/api/v1/me/predictions/{prediction_id}/explanation": {"get", "post"},
    "/api/v1/me/predictions/{prediction_id}/explanation/temporal": {"get"},
    "/api/v1/me/predictions/{prediction_id}/explanation/semantic": {"get"},
    "/api/v1/me/predictions/{prediction_id}/explanation/report": {"get"},
    "/api/v1/me/predictions/{prediction_id}/explanation/retry": {"post"},
}


def test_frozen_routes_are_present(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    for path, methods in FROZEN_ROUTES.items():
        assert path in paths, f"Frozen route missing from OpenAPI schema: {path}"
        for method in methods:
            assert method in paths[path], f"Frozen method missing for {path}: {method}"


def test_readiness_alias_is_reachable_but_hidden_from_schema(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/readiness" not in schema["paths"]
    assert client.get("/readiness").status_code in {200, 503}


FROZEN_SCHEMA_FIELDS: dict[str, set[str]] = {
    "PredictionSubmissionResponse": {
        "prediction_id",
        "request_id",
        "status",
        "source_type",
        "audio",
        "branches",
        "fusion",
        "research_eligible",
        "created_at",
    },
    "PredictionDetailResponse": {
        "prediction_id",
        "request_id",
        "source_type",
        "status",
        "audio",
        "branches",
        "fusion",
        "preprocessing",
        "total_processing_time_ms",
        "warnings",
        "research_eligible",
        "created_at",
        "updated_at",
        "completed_at",
    },
    "PredictionHistoryListResponse": {"items", "page", "limit", "has_next"},
    "PredictionHistoryItem": {
        "prediction_id",
        "filename",
        "source_type",
        "status",
        "duration_seconds",
        "final_prediction",
        "confidence",
        "mode_summary",
        "created_at",
    },
    "BranchPrediction": {
        "model_name",
        "display_name",
        "status",
        "mode",
        "prediction",
        "confidence",
        "probabilities",
        "processing_time_ms",
        "error",
        "metadata",
    },
    "FusionResult": {
        "status",
        "prediction",
        "confidence",
        "probabilities",
        "method",
        "branch_weights",
        "contains_dummy_branches",
        "eligible_for_research_evaluation",
        "warning",
        "config_version",
        "minimum_successful_branches",
        "contributing_branches",
        "excluded_branches",
    },
    "ReadinessResponse": {
        "status",
        "prediction_ready",
        "research_ready",
        "ffmpeg_available",
        "ffprobe_available",
        "mongodb_configured",
        "mongodb_available",
        "storage_enabled",
        "storage_available",
        "components",
    },
    "ModelHealthResponse": {
        "branch_name",
        "model_name",
        "display_name",
        "mode",
        "is_loaded",
        "uses_dummy_mode",
        "warning",
    },
    "PredictionAudioPlaybackResponse": {"playback_url", "expires_in_seconds"},
    "PredictionDeleteResponse": {"prediction_id", "status"},
    "TemporalExplanation": {
        "status",
        "method_version",
        "attention_score_peak",
        "attention_threshold",
        "regions",
    },
    "ErrorResponse": {"request_id", "error"},
}


def test_frozen_schema_fields_are_present(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    for schema_name, required_fields in FROZEN_SCHEMA_FIELDS.items():
        assert schema_name in components, f"Frozen schema missing from OpenAPI components: {schema_name}"
        properties = set(components[schema_name].get("properties", {}).keys())
        missing = required_fields - properties
        assert not missing, f"{schema_name} is missing frozen fields: {missing}"


def test_prediction_status_enum_is_frozen(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    status_schema = components.get("PredictionStatus")
    assert status_schema is not None
    assert set(status_schema["enum"]) == {
        "queued",
        "validating",
        "storing",
        "processing",
        "completed",
        "failed",
        "deleting",
        "deleted",
    }


def test_canonical_branch_public_model_names_are_frozen() -> None:
    from app.models.runtime import PUBLIC_MODEL_NAMES

    assert PUBLIC_MODEL_NAMES == {
        "lfcc_cnn_tcn": "cnn_acoustic",
        "aasist": "aasist",
        "ssl_sequence": "ssl_wavlm_xlsr",
        "glottal": "glottal_features",
    }
