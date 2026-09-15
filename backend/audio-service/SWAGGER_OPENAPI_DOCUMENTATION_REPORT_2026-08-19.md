# Swagger / OpenAPI Documentation Improvement

## Scope

This was a documentation/OpenAPI quality pass for the existing FastAPI backend.
It improved Swagger grouping, OpenAPI metadata, auth documentation, endpoint
descriptions, examples, and documentation tests without changing runtime route
behavior, persistence, model inference, fusion, XAI algorithms, or frontend code.

## Active Route Inventory

Generated from the active FastAPI OpenAPI schema. `/readiness` remains reachable
but intentionally hidden from OpenAPI.

| Method | Path | Router | Tag | Auth Required | Purpose | Request Type | Response Type |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GET | `/` | app | Health & Readiness | No | Service metadata | None | `dict[str, str]` |
| GET | `/health` | app | Health & Readiness | No | Liveness probe | None | `dict[str, str]` |
| GET | `/ready` | app | Health & Readiness | No | Dependency and prediction readiness | None | `ReadinessResponse` |
| POST | `/api/v1/predictions` | prediction_routes | Predictions | Clerk bearer JWT | Submit authenticated audio prediction | multipart/form-data | `PredictionSubmissionResponse` |
| GET | `/api/v1/predictions/{prediction_id}/status` | prediction_routes | Predictions | Clerk bearer JWT | Get prediction status | Path param | `PredictionJobStatusResponse` |
| POST | `/api/v1/external/predictions` | external_routes | API Keys / External | API-key bearer | Submit external prediction | multipart/form-data | `PredictionSubmissionResponse` |
| POST | `/api/v1/me/api-keys` | me_routes | Users, API Keys / External | Clerk bearer JWT | Create API key | JSON | `ApiKeyCreateResponse` |
| GET | `/api/v1/me/api-keys` | me_routes | Users, API Keys / External | Clerk bearer JWT | List API keys | None | `ApiKeyListResponse` |
| DELETE | `/api/v1/me/api-keys/{api_key_id}` | me_routes | Users, API Keys / External | Clerk bearer JWT | Revoke API key | Path param | `ApiKeyDeleteResponse` |
| GET | `/api/v1/me/predictions` | me_routes | Users, Predictions | Clerk bearer JWT | List prediction history | Query params | `PredictionHistoryListResponse` |
| GET | `/api/v1/me/predictions/{prediction_id}` | me_routes | Users, Predictions | Clerk bearer JWT | Get prediction detail | Path param | `PredictionDetailResponse` |
| DELETE | `/api/v1/me/predictions/{prediction_id}` | me_routes | Users, Predictions | Clerk bearer JWT | Delete prediction | Path param | `PredictionDeleteResponse` |
| GET | `/api/v1/me/predictions/{prediction_id}/audio` | me_routes | Users, Predictions | Clerk bearer JWT | Create signed playback URL | Path param | `PredictionAudioPlaybackResponse` |
| POST | `/api/v1/me/predictions/{prediction_id}/rerun` | me_routes | Users, Predictions | Clerk bearer JWT | Re-run prediction | Path + form | `PredictionSubmissionResponse` |
| GET | `/api/v1/users/me` | user_routes | Users | Clerk bearer JWT | Get authenticated principal | None | `AuthPrincipal` |
| POST | `/api/v1/voice/predict` | voice_routes | Voice Classification | Clerk bearer JWT and legacy flag | Deprecated compatibility prediction | multipart/form-data | `PredictionSubmissionResponse` |
| GET | `/api/v1/voice/models/health` | voice_routes | Voice Classification | No | Model branch health | None | `list[ModelHealthResponse]` |
| POST | `/api/v1/me/predictions/{prediction_id}/explanation` | xai_routes | Explainability / XAI | Clerk bearer JWT | Trigger XAI run | Path param | `XaiExplanationResponse` |
| GET | `/api/v1/me/predictions/{prediction_id}/explanation` | xai_routes | Explainability / XAI | Clerk bearer JWT | Get latest XAI run | Path param | `XaiExplanationResponse` |
| GET | `/api/v1/me/predictions/{prediction_id}/explanation/temporal` | xai_routes | Explainability / XAI | Clerk bearer JWT | Get temporal evidence | Path param | `TemporalExplanation` |
| GET | `/api/v1/me/predictions/{prediction_id}/explanation/semantic` | xai_routes | Explainability / XAI | Clerk bearer JWT | Get semantic evidence | Path param | `SemanticExplanation` |
| GET | `/api/v1/me/predictions/{prediction_id}/explanation/report` | xai_routes | Explainability / XAI | Clerk bearer JWT | Get combined report | Path param | `CombinedExplanationReport` |
| GET | `/api/v1/me/predictions/{prediction_id}/explanation/artifacts/{artifact_id}` | xai_routes | Explainability / XAI | Clerk bearer JWT | Download private artifact | Path params | binary response |
| POST | `/api/v1/me/predictions/{prediction_id}/explanation/retry` | xai_routes | Explainability / XAI | Clerk bearer JWT | Queue new XAI run | Path param | `XaiExplanationResponse` |

Total active documented operations: 24.

## Tags

Configured OpenAPI tags:

- Health & Readiness
- Predictions
- Voice Classification
- Explainability / XAI
- Users
- API Keys / External

Each tag has a description in FastAPI OpenAPI metadata.

## Authentication Schemes

Swagger now documents two existing bearer-header flows:

- `ClerkBearerAuth`: `Authorization: Bearer <Clerk JWT>`
- `ApiKeyBearerAuth`: `Authorization: Bearer msk_live_...`

No token is hardcoded, no authentication dependency was disabled, and no
`X-API-Key` header was invented.

## Prediction Endpoint Documentation

`POST /api/v1/predictions` now has a clear Swagger summary:

`Submit audio for deepfake voice classification`

Its description documents:

- authenticated audio upload
- validation
- audio preprocessing
- model branches
- score-level fusion
- persistence
- optional Voice XAI handoff
- accepted audio format categories
- configured upload size and duration limits
- common failure modes
- Clerk bearer JWT requirement

Multipart upload still renders through the existing `UploadFile` field.

## Health / Readiness Documentation

Health/readiness docs now distinguish:

- `/health`: HTTP process liveness only
- `/ready`: readiness for prediction work across audio tools, MongoDB, storage,
  prediction runner, models, and XAI components
- `/api/v1/voice/models/health`: per-branch model mode/load/readiness details

The docs clarify that `research_ready` is stricter than loaded/ready and that
AASIST reporting `research_ready: false` is not automatically a runtime failure.

## XAI Documentation

Existing XAI endpoints are documented only as existing behavior:

- trigger explanation
- get latest explanation
- get temporal evidence
- get semantic evidence
- get combined report
- download private artifact
- retry explanation

Descriptions document the required `prediction_id`, owner scoping, asynchronous
queue semantics, disabled/unavailable states, and the rule that XAI failures do
not change completed classifier predictions.

## Error Documentation

The existing centralized error envelope remains unchanged:

```json
{
  "request_id": "b34b2e2f-8fd2-4781-87cf-2c96e01fc2f5",
  "error": {
    "code": "authentication_failed",
    "message": "Authentication failed.",
    "details": null
  }
}
```

Prediction examples cover authentication failure, unsupported audio, and model
unavailable responses. The route still declares the existing status-code surface.

## Request / Response Examples

Added lightweight OpenAPI examples for:

- multipart upload cases: `human_voice.wav`, `synthetic_voice.wav`,
  `sample-voice.wav`
- completed prediction response using the existing schema fields
- authentication failure
- unsupported audio
- model unavailable

No binary example audio was embedded.

## OpenAPI Validation

Validated:

- `/docs` returns 200 and renders Swagger UI.
- `/redoc` returns 200.
- `/openapi.json` returns 200.
- `POST /api/v1/predictions` exists and remains POST.
- Prediction request remains multipart/form-data.
- Prediction response still references `PredictionSubmissionResponse`.
- Expected tags exist.
- Clerk bearer and API-key bearer schemes exist.
- API-key scheme documents the existing bearer API-key header.
- Health and XAI routes are documented.
- `/readiness` remains absent from OpenAPI.
- No duplicate operation IDs.
- Existing frozen route paths remain unchanged.

## Tests

Added:

- `backend/tests/test_swagger_openapi_documentation.py`

Commands run:

```text
.venv/bin/python -m pytest tests/test_swagger_openapi_documentation.py tests/test_openapi_contract.py -q
12 passed, 1 warning
```

```text
.venv/bin/python -m pytest tests/test_swagger_openapi_documentation.py tests/test_openapi_contract.py tests/test_voice_routes.py::test_legacy_prediction_route_is_marked_deprecated_in_openapi tests/test_voice_routes.py::test_canonical_prediction_route_remains_unchanged tests/test_aasist_light_v2_api_regression.py::test_phase6_public_prediction_openapi_contract_has_no_aasist_v2_fields -q
14 passed, 1 warning
```

```text
.venv/bin/python -m compileall app tests/test_swagger_openapi_documentation.py
PASS
```

```text
git diff --check
PASS
```

Ruff:

```text
.venv/bin/python -m ruff check ...
BLOCKED: No module named ruff
```

## Known Runtime Blockers

The pasted brief mentioned the AASIST startup error:

`AASIST-Light V2 checkpoint is missing model_state_dict`

This Swagger/OpenAPI task did not modify model loading or hide model startup
errors. In the current workspace, the endpoint schema can be generated and the
previous local configuration fix allows the app to start with CNN and AASIST V2
loaded. Remaining broader-suite environment blockers from earlier work still
apply where relevant, including missing `ruff` and Mongo/XAI test dependency
issues if those optional dependencies are absent.

## Swagger Test Guide

Created:

`backend/docs/setup/swagger-real-world-testing.md`

The guide covers backend venv activation, server startup, Swagger URLs,
authentication, health/readiness checks, prediction upload testing, model health,
optional XAI testing, and real-world test-data categories.

## Verdict

PASS WITH ISSUES

The OpenAPI/Swagger documentation is improved and covered by focused tests.
Issues are limited to existing environment/tooling blockers, not Swagger route
or schema failures.
