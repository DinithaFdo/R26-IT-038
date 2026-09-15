"""Safe Alibaba Model Studio Qwen client for optional XAI narratives."""

from __future__ import annotations

from hashlib import sha256
import json
from time import monotonic
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from app.config.settings import Settings
from app.schemas.xai import (
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationQuality,
    NarrativeExplanation,
    SemanticExplanation,
    TemporalExplanation,
)

PROMPT_VERSION = "voice-xai-narrative-v1"


class NarrativeGenerationError(RuntimeError):
    """An upstream narrative failure represented by a safe public error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AlibabaQwenNarrativeService:
    """Generate an evidence-grounded JSON narrative through Model Studio.

    The service performs no audio handling.  It serializes a bounded subset of
    already validated XAI fields, requests JSON mode, then validates every
    returned evidence reference before it can enter persistence.
    """

    provider = "alibaba_model_studio"

    def __init__(
        self,
        app_settings: Settings,
        *,
        http_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        self._settings = app_settings
        self._http_client_factory = http_client_factory

    async def generate(
        self,
        *,
        classifier: ClassifierSnapshot,
        temporal: TemporalExplanation | None,
        semantic: SemanticExplanation | None,
        quality: ExplanationQuality,
        report: CombinedExplanationReport,
        timeout_seconds: float,
    ) -> NarrativeExplanation:
        if not self._settings.xai_narrative_enabled:
            raise NarrativeGenerationError(
                "narrative_disabled", "AI narrative generation is disabled."
            )
        if timeout_seconds <= 0:
            raise NarrativeGenerationError(
                "narrative_timeout", "The AI narrative time budget was exhausted."
            )

        evidence, allowed_references = _narrative_evidence(
            classifier=classifier,
            temporal=temporal,
            semantic=semantic,
            quality=quality,
            report=report,
            max_semantic_features=self._settings.xai_narrative_max_semantic_features,
        )
        input_sha256 = _hash_json(evidence)
        payload = {
            "model": self._settings.xai_narrative_model,
            "messages": [
                {
                    "role": "system",
                    "content": _system_prompt(
                        summary_max_words=self._settings.xai_narrative_summary_max_words,
                        detailed_max_words=self._settings.xai_narrative_detailed_max_words,
                        allowed_references=sorted(allowed_references),
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Use only this validated evidence and return JSON only:\n"
                        + json.dumps(evidence, separators=(",", ":"), ensure_ascii=False)
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "enable_thinking": self._settings.xai_narrative_enable_thinking,
            "thinking_budget": self._settings.xai_narrative_thinking_budget,
        }
        effective_timeout = min(
            timeout_seconds, self._settings.xai_narrative_timeout_seconds
        )
        started_at = monotonic()
        try:
            async with self._http_client_factory(timeout=effective_timeout) as client:
                response = await client.post(
                    self._endpoint(),
                    headers={
                        "Authorization": (
                            f"Bearer {self._settings.xai_narrative_api_key}"
                        ),
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                response_payload = response.json()
        except httpx.TimeoutException as error:
            raise NarrativeGenerationError(
                "narrative_upstream_timeout",
                "The AI narrative service timed out.",
            ) from error
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            code = (
                "narrative_upstream_rejected"
                if 400 <= status_code < 500
                else "narrative_upstream_unavailable"
            )
            raise NarrativeGenerationError(
                code, "The AI narrative service could not process this request."
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise NarrativeGenerationError(
                "narrative_upstream_unavailable",
                "The AI narrative service is unavailable.",
            ) from error

        if monotonic() - started_at > timeout_seconds:
            raise NarrativeGenerationError(
                "narrative_timeout", "The AI narrative time budget was exhausted."
            )
        content = _response_content(response_payload)
        try:
            generated = json.loads(content)
            narrative = NarrativeExplanation(
                status=ComponentStatus.completed,
                provider=self.provider,
                model_id=_response_model_id(
                    response_payload, self._settings.xai_narrative_model
                ),
                prompt_version=PROMPT_VERSION,
                input_sha256=input_sha256,
                summary=generated["summary"],
                detailed_explanation=generated["detailed_explanation"],
                evidence_references=generated["evidence_references"],
            )
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as error:
            raise NarrativeGenerationError(
                "narrative_invalid_response",
                "The AI narrative service returned an invalid structured response.",
            ) from error
        _validate_narrative(
            narrative,
            allowed_references=allowed_references,
            summary_max_words=self._settings.xai_narrative_summary_max_words,
            detailed_max_words=self._settings.xai_narrative_detailed_max_words,
        )
        return narrative

    def _endpoint(self) -> str:
        return self._settings.xai_narrative_base_url.rstrip("/") + "/chat/completions"


def _narrative_evidence(
    *,
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
    quality: ExplanationQuality,
    report: CombinedExplanationReport,
    max_semantic_features: int,
) -> tuple[dict[str, Any], set[str]]:
    allowed = {"classifier:verdict", "classifier:threshold", "report:limitation"}
    evidence: dict[str, Any] = {
        "classifier": {
            "reference": "classifier:verdict",
            "verdict": classifier.verdict.value if classifier.verdict else None,
            "spoof_probability": classifier.spoof_probability,
            "bonafide_probability": classifier.bonafide_probability,
            "confidence": classifier.confidence,
            "decision_threshold": classifier.decision_threshold,
            "research_eligible": classifier.research_eligible,
            "contains_dummy_branches": classifier.contains_dummy_branches,
            "successful_branches": [
                {
                    "branch_name": branch.branch_name.value,
                    "spoof_probability": branch.spoof_probability,
                }
                for branch in classifier.branches
                if branch.spoof_probability is not None
            ],
        },
        "deterministic_report": {
            "finding": report.finding,
            "primary_evidence": report.primary_evidence,
            "quality_checks": report.quality_checks,
            "limitation": report.limitation,
            "recommendation": report.recommendation,
            "disclaimer": report.disclaimer,
        },
        "temporal_regions": [],
        "semantic_features": [],
        "quality_metrics": [],
    }
    if temporal is not None and temporal.status == ComponentStatus.completed:
        for region in temporal.high_attention_regions:
            reference = f"temporal:{region.region_id}"
            allowed.add(reference)
            evidence["temporal_regions"].append(
                {
                    "reference": reference,
                    "start_seconds": region.start_seconds,
                    "end_seconds": region.end_seconds,
                    "attention_score": region.attention_score,
                }
            )
    if semantic is not None and semantic.status == ComponentStatus.completed:
        for feature in semantic.feature_importance[:max_semantic_features]:
            reference = f"semantic:{feature.rank}"
            allowed.add(reference)
            evidence["semantic_features"].append(
                {
                    "reference": reference,
                    "display_name": feature.display_name,
                    "value": feature.value,
                    "unit": feature.unit,
                    "reference_summary": feature.reference_summary,
                    "shap_value": feature.shap_value,
                    "direction": feature.direction.value,
                    "start_seconds": feature.start_seconds,
                    "end_seconds": feature.end_seconds,
                }
            )
    for metric_name, metric in quality:
        if metric is None:
            continue
        reference = f"quality:{metric_name}"
        allowed.add(reference)
        evidence["quality_metrics"].append(
            {
                "reference": reference,
                "status": metric.status.value,
                "value": metric.value,
                "reason": metric.reason,
                "scope": metric.scope,
            }
        )
    return evidence, allowed


def _system_prompt(
    *, summary_max_words: int, detailed_max_words: int, allowed_references: list[str]
) -> str:
    return (
        "Return JSON only with exactly these keys: summary, detailed_explanation, "
        "evidence_references. Explain validated voice-classification XAI evidence "
        "in plain language. Do not diagnose people, claim proof, invent values, "
        "or override the classifier. Temporal regions are candidate evidence, and "
        "semantic contributions are auxiliary evidence. Preserve limitations and "
        "human-review language. summary must have no more than "
        f"{summary_max_words} words; detailed_explanation no more than "
        f"{detailed_max_words} words. evidence_references must be a non-empty "
        "JSON array containing only these exact strings: "
        + json.dumps(allowed_references)
    )


def _response_content(response_payload: Any) -> str:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise NarrativeGenerationError(
            "narrative_invalid_response",
            "The AI narrative service returned an invalid structured response.",
        ) from error
    if not isinstance(content, str) or not content.strip():
        raise NarrativeGenerationError(
            "narrative_invalid_response",
            "The AI narrative service returned an invalid structured response.",
        )
    return content


def _response_model_id(response_payload: Any, configured_model: str) -> str:
    model = response_payload.get("model") if isinstance(response_payload, dict) else None
    return model if isinstance(model, str) and model.strip() else configured_model


def _validate_narrative(
    narrative: NarrativeExplanation,
    *,
    allowed_references: set[str],
    summary_max_words: int,
    detailed_max_words: int,
) -> None:
    if len(_words(narrative.summary)) > summary_max_words:
        raise NarrativeGenerationError(
            "narrative_invalid_response",
            "The AI narrative service returned an invalid structured response.",
        )
    if len(_words(narrative.detailed_explanation)) > detailed_max_words:
        raise NarrativeGenerationError(
            "narrative_invalid_response",
            "The AI narrative service returned an invalid structured response.",
        )
    if not set(narrative.evidence_references).issubset(allowed_references):
        raise NarrativeGenerationError(
            "narrative_invalid_response",
            "The AI narrative service returned an invalid structured response.",
        )


def _words(value: str) -> list[str]:
    return value.split()


def _hash_json(value: dict[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
