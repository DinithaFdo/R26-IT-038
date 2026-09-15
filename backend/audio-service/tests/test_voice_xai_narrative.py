from datetime import UTC, datetime
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
from app.voice_xai.narrative.service import (
    AlibabaQwenNarrativeService,
    NarrativeGenerationError,
)


def _settings(**overrides) -> Settings:
    values = {
        "xai_enabled": True,
        "xai_mode": "mock",
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
    assert narrative.status == ComponentStatus.completed
    assert captured["url"].endswith("/compatible-mode/v1/chat/completions")
    assert captured["authorization"] == "Bearer test-model-studio-key"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["enable_thinking"] is True
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
        regions=[
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
        target_type=SemanticTargetType.independent_acoustic_evidence_model,
        model_version="semantic-v1",
        feature_schema_version="features-v1",
        extractor_version="extractor-v1",
        output_space="probability",
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
