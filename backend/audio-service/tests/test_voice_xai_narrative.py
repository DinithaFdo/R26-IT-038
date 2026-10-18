from datetime import UTC, datetime
import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.xai import (
    CanonicalXaiBranch,
    ClassifierBranchSnapshot,
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationQuality,
    NarrativeExplanation,
    ReportDisposition,
    SemanticExplanation,
    SemanticFeatureContribution,
    SemanticTargetType,
    ShapDirection,
    TemporalExplanation,
    TemporalRegion,
)
from app.voice_xai.report.service import (
    AlibabaQwenNarrativeService,
    NarrativeGenerationError,
    PROMPT_VERSION,
)


def _settings(**overrides) -> Settings:
    values = {
        "xai_enabled": True,
        "xai_mode": "real",
        "xai_narrative_enabled": True,
        "xai_narrative_api_key": "test-model-studio-key",
        "xai_narrative_base_url": (
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        ),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_narrative_settings_require_complete_model_studio_configuration() -> None:
    with pytest.raises(ValidationError, match="XAI_NARRATIVE_API_KEY"):
        _settings(xai_narrative_api_key="")
    with pytest.raises(ValidationError, match="XAI_NARRATIVE_BASE_URL"):
        _settings(xai_narrative_base_url="http://example.test/v1")
    with pytest.raises(ValidationError, match="requires XAI_ENABLED"):
        Settings(
            _env_file=None,
            xai_enabled=False,
            xai_narrative_enabled=True,
            xai_narrative_api_key="test-model-studio-key",
            xai_narrative_base_url=(
                "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
            ),
        )


@pytest.mark.anyio
async def test_qwen_narrative_sends_only_bounded_evidence_and_validates_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["requests"] = captured.get("requests", 0) + 1
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3.6-flash-2026-04-16",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "The classifier flagged candidate spoof evidence requiring human review.",
                                    "detailed_explanation": "The fused classifier score and the selected temporal region support a cautious review. The semantic feature is auxiliary evidence and does not prove the result.",
                                    "evidence_references": [
                                        "classifier:verdict",
                                        "temporal:1",
                                        "semantic:1",
                                    ],
                                }
                            )
                        }
                    }
                ],
            },
        )

    service = AlibabaQwenNarrativeService(
        _settings(),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    narrative = await service.generate(
        classifier=_classifier(),
        temporal=_temporal(),
        semantic=_semantic(),
        quality=ExplanationQuality(),
        report=_report(),
        timeout_seconds=10,
    )

    assert narrative.model_id == "qwen3.6-flash-2026-04-16"
    assert narrative.prompt_version == PROMPT_VERSION
    assert captured["requests"] == 1
    assert narrative.status == ComponentStatus.completed
    assert captured["url"].endswith("/compatible-mode/v1/chat/completions")
    assert captured["authorization"] == "Bearer test-model-studio-key"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["enable_thinking"] is True
    schema = json.loads(captured["body"]["messages"][0]["content"].split(
        "The complete response must follow this JSON schema: ", 1
    )[1])
    assert set(schema["required"]) == {"summary", "detailed_explanation", "evidence_references"}
    reference_schema = schema["properties"]["evidence_references"]
    assert reference_schema["minItems"] == 1
    assert reference_schema["uniqueItems"] is True
    assert "semantic:1" in reference_schema["items"]["enum"]
    assert "semantic:999" not in reference_schema["items"]["enum"]
    user_evidence = captured["body"]["messages"][1]["content"]
    assert "owner_user_id" not in user_evidence
    assert "sample.wav" not in user_evidence
    assert "temporal:1" in user_evidence


@pytest.mark.anyio
async def test_qwen_narrative_rejects_invented_evidence_references() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Candidate evidence requires review.",
                                    "detailed_explanation": "The result requires qualified human review.",
                                    "evidence_references": ["temporal:999"],
                                }
                            )
                        }
                    }
                ]
            },
        )

    service = AlibabaQwenNarrativeService(
        _settings(),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(NarrativeGenerationError, match="invalid structured response"):
        await service.generate(
            classifier=_classifier(),
            temporal=_temporal(),
            semantic=_semantic(),
            quality=ExplanationQuality(),
            report=_report(),
            timeout_seconds=10,
        )


def test_narrative_schema_rejects_duplicate_evidence_references() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        NarrativeExplanation(
            provider="alibaba_model_studio",
            model_id="qwen3.6-flash",
            prompt_version="voice-xai-narrative-v1",
            input_sha256="0" * 64,
            summary="A concise evidence summary.",
            detailed_explanation="A detailed evidence summary requiring review.",
            evidence_references=["classifier:verdict", "classifier:verdict"],
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "invalid_content_type"),
        ("", "empty_content"),
        ("private-evidence-invalid-json", "invalid_json"),
        ("[]", "json_object_required"),
        (json.dumps({"summary": "private-evidence-summary"}), "missing_required_fields"),
        (
            json.dumps({
                "summary": "private-evidence-summary",
                "detailed_explanation": "Review the deterministic evidence.",
                "evidence_references": ["classifier:verdict", "classifier:verdict"],
            }),
            "duplicate_evidence_references",
        ),
        (
            json.dumps({
                "summary": {"private-evidence-summary": "invalid type"},
                "detailed_explanation": "Review the deterministic evidence.",
                "evidence_references": ["classifier:verdict"],
            }),
            "schema_validation_failed",
        ),
        (
            json.dumps({
                "summary": "private-evidence-summary " * 51,
                "detailed_explanation": "Review the deterministic evidence.",
                "evidence_references": ["classifier:verdict"],
            }),
            "summary_word_limit_exceeded",
        ),
        (
            json.dumps({
                "summary": "private-evidence-summary",
                "detailed_explanation": "private-evidence-detail " * 101,
                "evidence_references": ["classifier:verdict"],
            }),
            "detail_word_limit_exceeded",
        ),
        (
            json.dumps({
                "summary": "private-evidence-summary",
                "detailed_explanation": "Review the deterministic evidence.",
                "evidence_references": ["private-evidence-unknown-reference"],
            }),
            "unknown_evidence_references",
        ),
    ],
)
async def test_narrative_rejection_logs_specific_reason_without_evidence(
    content, reason, caplog
) -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    service = AlibabaQwenNarrativeService(
        _settings(xai_narrative_summary_max_words=50, xai_narrative_detailed_max_words=100),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(NarrativeGenerationError) as caught:
        await service.generate(
            classifier=_classifier(), temporal=_temporal(), semantic=_semantic(),
            quality=ExplanationQuality(), report=_report(), timeout_seconds=10,
        )

    assert caught.value.code == "narrative_invalid_response"
    assert caught.value.reason == reason
    assert request_count == 2
    assert str(caught.value) == "The AI narrative service returned an invalid structured response."
    assert f"reason={reason}" in caplog.text
    assert "private-evidence" not in caplog.text
    assert "test-model-studio-key" not in caplog.text
    assert "private-evidence" not in str(caught.value)
    if reason == "schema_validation_failed":
        assert "fields=summary" in caplog.text
    elif reason == "missing_required_fields":
        assert "fields=detailed_explanation,evidence_references" in caplog.text


@pytest.mark.anyio
async def test_narrative_rejection_identifies_malformed_completion_envelope(caplog) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    service = AlibabaQwenNarrativeService(
        _settings(),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(NarrativeGenerationError) as caught:
        await service.generate(
            classifier=_classifier(), temporal=_temporal(), semantic=_semantic(),
            quality=ExplanationQuality(), report=_report(), timeout_seconds=10,
        )

    assert caught.value.reason == "invalid_completion_envelope"
    assert "reason=invalid_completion_envelope" in caplog.text


def _completion(content) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _valid_content() -> str:
    return json.dumps({
        "summary": "Validated evidence requires human review.",
        "detailed_explanation": "Review the classifier result and deterministic report.",
        "evidence_references": ["classifier:verdict", "report:limitation"],
    })


def _generation_arguments(**overrides) -> dict:
    values = dict(
        classifier=_classifier(), temporal=_temporal(), semantic=_semantic(),
        quality=ExplanationQuality(), report=_report(), timeout_seconds=60,
    )
    values.update(overrides)
    return values


@pytest.mark.anyio
async def test_missing_references_are_corrected_once_with_the_same_evidence(caplog) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return _completion(json.dumps({
                "summary": "private-invalid-summary",
                "detailed_explanation": "private-invalid-detail",
            }))
        return _completion(_valid_content())

    service = AlibabaQwenNarrativeService(
        _settings(),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with caplog.at_level("INFO", logger="app.voice_xai.report.service"):
        narrative = await service.generate(**_generation_arguments())

    assert narrative.evidence_references == ["classifier:verdict", "report:limitation"]
    assert len(requests) == 2
    assert requests[1]["messages"][:2] == requests[0]["messages"]
    correction = requests[1]["messages"][2]["content"]
    assert "missing_required_fields" in correction
    assert "evidence_references" in correction
    assert "private-invalid" not in json.dumps(requests[1])
    assert "private-invalid" not in caplog.text
    assert "recovered on attempt 2" in caplog.text


@pytest.mark.anyio
async def test_second_invalid_response_is_rejected_without_fabricating_references() -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return _completion(json.dumps({"summary": "Summary", "detailed_explanation": "Detail"}))
        return _completion(json.dumps({
            "summary": "Summary", "detailed_explanation": "Detail",
            "evidence_references": ["temporal:999"],
        }))

    service = AlibabaQwenNarrativeService(
        _settings(),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(NarrativeGenerationError) as caught:
        await service.generate(**_generation_arguments())

    assert request_count == 2
    assert caught.value.code == "narrative_invalid_response"
    assert caught.value.reason == "unknown_evidence_references"


@pytest.mark.anyio
@pytest.mark.parametrize(("status_code", "error_code"), [
    (401, "narrative_upstream_rejected"),
    (429, "narrative_upstream_rejected"),
    (503, "narrative_upstream_unavailable"),
])
async def test_provider_errors_do_not_trigger_response_correction(status_code, error_code) -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(status_code, json={"error": "private-provider-detail"})

    service = AlibabaQwenNarrativeService(
        _settings(),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(NarrativeGenerationError) as caught:
        await service.generate(**_generation_arguments())

    assert request_count == 1
    assert caught.value.code == error_code
    assert "private-provider-detail" not in str(caught.value)


@pytest.mark.anyio
@pytest.mark.parametrize(("job_budget", "expected_timeouts"), [(40, [30, 15]), (600, [30, 30])])
async def test_correction_shares_a_bounded_budget(monkeypatch, job_budget, expected_timeouts) -> None:
    clock = [0.0]
    monkeypatch.setattr("app.voice_xai.report.service.monotonic", lambda: clock[0])
    timeouts = []
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        clock[0] = 25.0 if request_count == 1 else 30.0
        return _completion("{}" if request_count == 1 else _valid_content())

    def factory(**kwargs):
        timeouts.append(kwargs["timeout"])
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    service = AlibabaQwenNarrativeService(_settings(), http_client_factory=factory)
    narrative = await service.generate(**_generation_arguments(timeout_seconds=job_budget))

    assert narrative.status == ComponentStatus.completed
    assert timeouts == expected_timeouts


@pytest.mark.anyio
async def test_correction_is_not_sent_after_the_job_budget_expires(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("app.voice_xai.report.service.monotonic", lambda: clock[0])
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        clock[0] = 10.0
        return _completion("{}")

    service = AlibabaQwenNarrativeService(
        _settings(),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(NarrativeGenerationError) as caught:
        await service.generate(**_generation_arguments(timeout_seconds=10))

    assert caught.value.code == "narrative_timeout"
    assert request_count == 1


@pytest.mark.anyio
async def test_request_has_a_wall_clock_timeout_and_does_not_retry() -> None:
    request_count = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        await asyncio.Event().wait()
        return _completion(_valid_content())

    service = AlibabaQwenNarrativeService(
        _settings(xai_narrative_timeout_seconds=0.05),
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(NarrativeGenerationError) as caught:
        await asyncio.wait_for(service.generate(**_generation_arguments()), timeout=2)

    assert caught.value.code == "narrative_upstream_timeout"
    assert request_count == 1


def _classifier() -> ClassifierSnapshot:
    return ClassifierSnapshot(
        verdict=PredictionLabel.spoof,
        spoof_probability=0.8,
        bonafide_probability=0.2,
        confidence=0.8,
        decision_threshold=0.5,
        contains_dummy_branches=True,
        research_eligible=False,
        branches=[
            ClassifierBranchSnapshot(
                branch_name=CanonicalXaiBranch.lfcc_cnn_tcn,
                model_name="cnn_acoustic",
                status=BranchStatus.success,
                mode=ModelMode.dummy,
                spoof_probability=0.8,
            )
        ],
    )


def _temporal() -> TemporalExplanation:
    return TemporalExplanation(
        status=ComponentStatus.completed,
        method_version="attention-rollout-v1",
        high_attention_regions=[
            TemporalRegion(
                region_id=1,
                start_seconds=0.2,
                end_seconds=0.5,
                attention_score=0.9,
            )
        ],
    )


def _semantic() -> SemanticExplanation:
    return SemanticExplanation(
        status=ComponentStatus.completed,
        target_type=SemanticTargetType.classifier_surrogate,
        model_version="semantic-v1",
        feature_schema_version="features-v1",
        extractor_version="extractor-v1",
        output_space="raw_regression",
        feature_importance=[
            SemanticFeatureContribution(
                rank=1,
                feature_name="jitter",
                display_name="Pitch jitter",
                value=0.2,
                shap_value=0.3,
                direction=ShapDirection.toward_spoof,
            )
        ],
    )


def _report() -> CombinedExplanationReport:
    return CombinedExplanationReport(
        status=ComponentStatus.completed,
        disposition=ReportDisposition.spoof_suspected,
        finding="The classifier returned spoof candidate evidence.",
        primary_evidence="Validated classifier and temporal evidence are available.",
        quality_checks="No per-analysis quality metrics were supplied.",
        limitation="Evidence is correlational decision-support information.",
        recommendation="Review the source audio and provenance.",
        disclaimer="Explanation output is decision-support evidence only.",
    )
