from __future__ import annotations

from fastapi.testclient import TestClient


EXPECTED_TAGS = {
    "Health & Readiness",
    "Predictions",
    "Voice Classification",
    "Explainability / XAI",
    "Users",
    "API Keys / External",
}

EXPECTED_ROUTE_METHODS = {
    "/": {"get"},
    "/health": {"get"},
    "/ready": {"get"},
    "/api/v1/external/predictions": {"post"},
    "/api/v1/me/api-keys": {"get", "post"},
    "/api/v1/me/api-keys/{api_key_id}": {"delete"},
    "/api/v1/me/predictions": {"get"},
    "/api/v1/me/predictions/{prediction_id}": {"get", "delete"},
    "/api/v1/me/predictions/{prediction_id}/audio": {"get"},
    "/api/v1/me/predictions/{prediction_id}/rerun": {"post"},
    "/api/v1/predictions": {"post"},
    "/api/v1/predictions/{prediction_id}/status": {"get"},
    "/api/v1/users/me": {"get"},
    "/api/v1/voice/predict": {"post"},
    "/api/v1/voice/models/health": {"get"},
    "/api/v1/me/predictions/{prediction_id}/explanation": {"get", "post"},
    "/api/v1/me/predictions/{prediction_id}/explanation/temporal": {"get"},
    "/api/v1/me/predictions/{prediction_id}/explanation/semantic": {"get"},
    "/api/v1/me/predictions/{prediction_id}/explanation/report": {"get"},
    "/api/v1/me/predictions/{prediction_id}/explanation/retry": {"post"},
    "/api/v1/me/predictions/{prediction_id}/explanation/artifacts/{artifact_id}": {
        "get"
    },
}


def test_swagger_and_redoc_are_enabled(client: TestClient) -> None:
    docs = client.get("/docs")
    redoc = client.get("/redoc")

    assert docs.status_code == 200
    assert "swagger-ui" in docs.text
    assert redoc.status_code == 200
    assert "redoc" in redoc.text.lower()


def test_swagger_openapi_schema_documents_active_routes(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    assert "/readiness" not in paths
    for path, methods in EXPECTED_ROUTE_METHODS.items():
        assert path in paths
        assert methods <= set(paths[path])


def test_swagger_openapi_has_clean_tags_and_no_duplicate_operation_ids(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()

    tag_names = {tag["name"] for tag in schema["tags"]}
    assert EXPECTED_TAGS <= tag_names
    operation_ids = [
        operation["operationId"]
        for path_item in schema["paths"].values()
        for method, operation in path_item.items()
        if method in {"get", "post", "put", "patch", "delete"}
    ]
    assert len(operation_ids) == len(set(operation_ids))


def test_swagger_openapi_documents_bearer_and_api_key_auth(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    security_schemes = schema["components"]["securitySchemes"]

    assert security_schemes["ClerkBearerAuth"]["scheme"] == "bearer"
    assert security_schemes["ClerkBearerAuth"]["bearerFormat"] == "JWT"
    assert security_schemes["ApiKeyBearerAuth"]["scheme"] == "bearer"
    assert "msk_live" in security_schemes["ApiKeyBearerAuth"]["bearerFormat"]
    assert schema["paths"]["/api/v1/predictions"]["post"]["security"] == [
        {"ClerkBearerAuth": []}
    ]
    assert schema["paths"]["/api/v1/external/predictions"]["post"]["security"] == [
        {"ApiKeyBearerAuth": []}
    ]


def test_swagger_openapi_documents_prediction_multipart_and_examples(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    prediction = schema["paths"]["/api/v1/predictions"]["post"]
    multipart = prediction["requestBody"]["content"]["multipart/form-data"]
    responses = prediction["responses"]

    assert prediction["summary"] == "Submit audio for deepfake voice classification"
    assert "upload -> validation -> audio preprocessing" in prediction["description"]
    assert "Authorize" in prediction["description"]
    assert multipart["schema"]["$ref"].startswith("#/components/schemas/")
    assert set(multipart["examples"]) == {
        "humanVoice",
        "syntheticVoice",
        "repoSample",
    }
    assert (
        responses["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/PredictionSubmissionResponse"
    )
    assert "completedPrediction" in responses["200"]["content"]["application/json"][
        "examples"
    ]
    assert "authentication_failed" in responses["401"]["content"][
        "application/json"
    ]["examples"]
    assert "model_unavailable" in responses["503"]["content"]["application/json"][
        "examples"
    ]


def test_swagger_openapi_documents_health_and_xai_routes(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["paths"]["/health"]["get"]["tags"] == ["Health & Readiness"]
    assert schema["paths"]["/ready"]["get"]["tags"] == ["Health & Readiness"]
    assert "research_ready" in schema["paths"]["/ready"]["get"]["description"]
    xai = schema["paths"]["/api/v1/me/predictions/{prediction_id}/explanation"][
        "post"
    ]
    assert xai["tags"] == ["Explainability / XAI"]
    assert "asynchronously" in xai["description"]
    assert "prediction_id" in xai["description"]
