"""Deterministic Voice XAI reports and optional validated Qwen narration."""

from __future__ import annotations

import asyncio
from hashlib import sha256
import json
import logging
from time import monotonic
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from app.config.settings import Settings
from app.schemas.common import PredictionLabel
from app.schemas.xai import (
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationQuality,
    MetricStatus,
    NarrativeExplanation,
    ReportDisposition,
    SemanticExplanation,
    TemporalExplanation,
)


class DeterministicReportComposer:
    """Create a repeatable human-review report without generative inference."""

    _DISCLAIMER = (
        "This explanation is decision-support evidence only. It does not prove "
        "that audio is spoofed or bonafide and requires qualified human review."
    )

    def compose(
        self,
        *,
        classifier: ClassifierSnapshot,
        temporal: TemporalExplanation | None,
        semantic: SemanticExplanation | None,
        quality: ExplanationQuality | None = None,
    ) -> CombinedExplanationReport:
        """Return the same report for the same validated inputs."""

        disposition = _disposition(classifier)
        finding = _finding(classifier, temporal, semantic)
        return CombinedExplanationReport(
            status=ComponentStatus.completed,
            disposition=disposition,
            requires_human_review=True,
            finding=finding,
            primary_evidence=_primary_evidence(classifier, temporal, semantic),
            quality_checks=_quality_checks(quality),
            limitation=_limitations(classifier, temporal, semantic),
            recommendation=(
                "Review the original audio, classifier snapshot, evidence intervals, "
                "and provenance before making a decision or recording sign-off."
            ),
            disclaimer=self._DISCLAIMER,
        )


def _disposition(classifier: ClassifierSnapshot) -> ReportDisposition:
    if classifier.verdict == PredictionLabel.spoof:
        return ReportDisposition.spoof_suspected
    if classifier.verdict == PredictionLabel.bonafide:
        return ReportDisposition.bonafide_suspected
    return ReportDisposition.inconclusive


def _finding(
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> str:
    if classifier.verdict is None or classifier.spoof_probability is None:
        decision = "The classifier verdict and score are unavailable"
    else:
        decision = (
            f"The classifier returned {classifier.verdict.value} with a spoof score "
            f"of {classifier.spoof_probability:.2f}."
        )
    return " ".join((decision, _temporal_sentence(temporal), _semantic_sentence(semantic)))


def _primary_evidence(
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> str:
    parts = [
        "Classifier decision threshold " f"was {classifier.decision_threshold:.2f}."
    ]
    if temporal is not None and temporal.status == ComponentStatus.completed:
        parts.append(_temporal_sentence(temporal))
    if semantic is not None and semantic.status == ComponentStatus.completed:
        parts.append(_semantic_sentence(semantic))
    if len(parts) == 1:
        parts.append("No completed temporal or semantic evidence is available.")
    return " ".join(parts)


def _temporal_sentence(temporal: TemporalExplanation | None) -> str:
    if temporal is None or temporal.status != ComponentStatus.completed:
        return "Temporal high-attention evidence is unavailable."
    if not temporal.high_attention_regions:
        return "No high-attention regions were selected by the temporal configuration."
    regions = ", ".join(
        f"{region.start_seconds:.2f}-{region.end_seconds:.2f} seconds"
        for region in temporal.high_attention_regions
    )
    return f"High-attention activity occurred at {regions}."


def _semantic_sentence(semantic: SemanticExplanation | None) -> str:
    if semantic is None or semantic.status != ComponentStatus.completed:
        return "Semantic acoustic evidence is unavailable."
    if not semantic.feature_importance:
        return "No ranked semantic feature contributions are available."
    features = ", ".join(
        contribution.display_name for contribution in semantic.feature_importance[:2]
    )
    sentence = f"The largest semantic {semantic.target_type.value} contributions were {features}."
    localization = semantic.localization
    if localization is not None:
        sentence += (f" Window SHAP activity has {localization.coverage_percentage:.1f}% valid coverage "
                     f"of the temporal evidence spans; window model validation is {localization.model_validation_status}.")
        if localization.regions:
            regions = ', '.join(f'{region.start_seconds:.2f}-{region.end_seconds:.2f}s' for region in localization.regions)
            sentence += f' High semantic activity occurred at {regions}.'
        sentence += ' Window activity and temporal overlap describe agreement, not verified fake-audio boundaries.'
    return sentence


def _quality_checks(quality: ExplanationQuality | None) -> str:
    if quality is None:
        return "No per-analysis quality metrics were supplied."
    available = []
    unavailable = []
    for name, metric in quality:
        if metric is None:
            continue
        if metric.status == MetricStatus.available and metric.value is not None:
            # Keep the full-precision value in ``quality`` itself. The report is
            # a human-facing summary, so render a stable, readable precision
            # together with the metric scope.
            available.append(f"{name}={metric.value:.4f} ({metric.scope})")
            continue
        reason = f" ({metric.reason})" if metric.reason else ""
        unavailable.append(f"{name}{reason}")
    parts = []
    if available:
        parts.append("Available quality metrics: " + ", ".join(available) + ".")
    if unavailable:
        parts.append("Unavailable quality metrics: " + ", ".join(unavailable) + ".")
    return " ".join(parts) or "No per-analysis quality metrics were supplied."


def _limitations(
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> str:
    limitations = []
    if classifier.contains_dummy_branches:
        limitations.append(
            "The classifier contains development/dummy branches and is not research eligible."
        )
    for component in (temporal, semantic):
        if component is not None and component.warning:
            limitations.append(component.warning)
    if not limitations:
        limitations.append(
            "Temporal attention and semantic contributions are correlational decision-support evidence."
        )
    return " ".join(limitations)


PROMPT_VERSION = "voice-xai-narrative-v2"
NARRATIVE_MAX_RESPONSE_ATTEMPTS = 2
_NARRATIVE_RESPONSE_FIELDS = ("summary", "detailed_explanation", "evidence_references")


def narrative_generation_timeout_seconds(app_settings: Settings) -> float:
    """Allow one request timeout per attempt, bounded further by the job deadline."""
    return app_settings.xai_narrative_timeout_seconds * NARRATIVE_MAX_RESPONSE_ATTEMPTS


class NarrativeGenerationError(RuntimeError):
    """An upstream narrative failure represented by a safe public error code."""

    def __init__(self, code: str, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason


def _invalid_narrative_response(
    reason: str, *, fields: tuple[str, ...] = ()
) -> NarrativeGenerationError:
    # Log only fixed reason codes and schema field names, never generated text,
    # evidence values, credentials, or Pydantic errors containing input data.
    logging.getLogger(__name__).warning(
        "Voice XAI narrative response rejected: reason=%s; fields=%s",
        reason,
        ",".join(fields) or "-",
    )
    return NarrativeGenerationError(
        "narrative_invalid_response",
        "The AI narrative service returned an invalid structured response.",
        reason=reason,
    )


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
        deadline = monotonic() + min(
            timeout_seconds, narrative_generation_timeout_seconds(self._settings)
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
        attempt = 1
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise NarrativeGenerationError(
                    "narrative_timeout", "The AI narrative time budget was exhausted."
                )
            response_payload = await self._request(
                payload,
                timeout_seconds=min(remaining, self._settings.xai_narrative_timeout_seconds),
            )
            if monotonic() >= deadline:
                raise NarrativeGenerationError(
                    "narrative_timeout", "The AI narrative time budget was exhausted."
                )
            try:
                narrative = _parse_narrative_response(
                    response_payload,
                    configured_model=self._settings.xai_narrative_model,
                    input_sha256=input_sha256,
                    allowed_references=allowed_references,
                    summary_max_words=self._settings.xai_narrative_summary_max_words,
                    detailed_max_words=self._settings.xai_narrative_detailed_max_words,
                )
            except NarrativeGenerationError as error:
                if (
                    error.code != "narrative_invalid_response"
                    or attempt >= NARRATIVE_MAX_RESPONSE_ATTEMPTS
                ):
                    raise
                if monotonic() >= deadline:
                    raise NarrativeGenerationError(
                        "narrative_timeout", "The AI narrative time budget was exhausted."
                    ) from error
                attempt += 1
                logging.getLogger(__name__).info(
                    "Voice XAI narrative response retrying: attempt=%s; reason=%s",
                    attempt,
                    error.reason,
                )
                # Keep the same validated evidence. Never echo untrusted model
                # output into a correction prompt or fabricate missing references.
                payload["messages"].append({
                    "role": "user",
                    "content": (
                        f"The previous response failed validation: {error.reason}. "
                        "Generate a complete replacement JSON object from the same evidence. "
                        "All three keys summary, detailed_explanation, evidence_references "
                        "are mandatory. Include a non-empty, unique evidence_references "
                        "array using only allowed references supporting your explanation. "
                        "Follow the response schema and word limits in the system message."
                    ),
                })
                continue
            if monotonic() >= deadline:
                raise NarrativeGenerationError(
                    "narrative_timeout", "The AI narrative time budget was exhausted."
                )
            if attempt > 1:
                logging.getLogger(__name__).info(
                    "Voice XAI narrative response recovered on attempt %s", attempt
                )
            return narrative

    async def _request(self, payload: dict[str, Any], *, timeout_seconds: float) -> Any:
        try:
            # HTTPX phase timeouts alone do not bound the total request duration.
            async with asyncio.timeout(timeout_seconds):
                async with self._http_client_factory(timeout=timeout_seconds) as client:
                    response = await client.post(
                        self._endpoint(),
                        headers={
                            "Authorization": f"Bearer {self._settings.xai_narrative_api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                    return response.json()
        except (httpx.TimeoutException, TimeoutError) as error:
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

    def _endpoint(self) -> str:
        return self._settings.xai_narrative_base_url.rstrip("/") + "/chat/completions"


def _parse_narrative_response(
    response_payload: Any,
    *,
    configured_model: str,
    input_sha256: str,
    allowed_references: set[str],
    summary_max_words: int,
    detailed_max_words: int,
) -> NarrativeExplanation:
    content = _response_content(response_payload)
    try:
        generated = json.loads(content)
    except json.JSONDecodeError as error:
        raise _invalid_narrative_response("invalid_json") from error
    if not isinstance(generated, dict):
        raise _invalid_narrative_response("json_object_required")
    missing_fields = tuple(field for field in _NARRATIVE_RESPONSE_FIELDS if field not in generated)
    if missing_fields:
        raise _invalid_narrative_response("missing_required_fields", fields=missing_fields)
    references = generated["evidence_references"]
    if isinstance(references, list) and all(isinstance(ref, str) for ref in references):
        if len(set(references)) != len(references):
            raise _invalid_narrative_response("duplicate_evidence_references")
    try:
        narrative = NarrativeExplanation(
            status=ComponentStatus.completed,
            provider=AlibabaQwenNarrativeService.provider,
            model_id=_response_model_id(response_payload, configured_model),
            prompt_version=PROMPT_VERSION,
            input_sha256=input_sha256,
            summary=generated["summary"],
            detailed_explanation=generated["detailed_explanation"],
            evidence_references=generated["evidence_references"],
        )
    except ValidationError as error:
        fields = tuple(sorted({
            str(item["loc"][0]) if item["loc"] else "narrative"
            for item in error.errors(include_input=False, include_context=False)
        }))
        raise _invalid_narrative_response("schema_validation_failed", fields=fields) from error
    _validate_narrative(
        narrative,
        allowed_references=allowed_references,
        summary_max_words=summary_max_words,
        detailed_max_words=detailed_max_words,
    )
    return narrative


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
        if semantic.localization is not None:
            allowed.add('semantic:localization')
            evidence['semantic_localization'] = {
                'reference': 'semantic:localization',
                'status': semantic.localization.status,
                'validation_status': semantic.localization.model_validation_status,
                'coverage_percentage': semantic.localization.coverage_percentage,
                'activity_definition': semantic.localization.activity_definition,
                'regions': [region.model_dump(mode='json') for region in semantic.localization.regions],
                'limitation': 'Window-surrogate activity is distinct from the full-clip prediction and is not ground-truth localization.',
            }
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
    properties = NarrativeExplanation.model_json_schema()["properties"]
    response_properties = {field: properties[field] for field in _NARRATIVE_RESPONSE_FIELDS}
    response_properties["evidence_references"]["uniqueItems"] = True
    response_properties["evidence_references"]["items"]["enum"] = allowed_references
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(_NARRATIVE_RESPONSE_FIELDS),
        "properties": response_properties,
    }
    return (
        "Return JSON only with exactly these keys: summary, detailed_explanation, "
        "evidence_references. All three keys are mandatory, including evidence_references. "
        "Explain validated voice-classification XAI evidence "
        "in plain language. Do not diagnose people, claim proof, invent values, "
        "or override the classifier. Temporal regions are candidate evidence, and "
        "semantic contributions are auxiliary evidence. Preserve limitations and "
        "human-review language. summary must have no more than "
        f"{summary_max_words} words; detailed_explanation no more than "
        f"{detailed_max_words} words. evidence_references must be a non-empty "
        "JSON array of unique references actually used in your explanation. "
        "Copy allowed reference strings exactly; do not invent references. "
        "The complete response must follow this JSON schema: "
        + json.dumps(response_schema, separators=(",", ":"))
    )


def _response_content(response_payload: Any) -> str:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise _invalid_narrative_response("invalid_completion_envelope") from error
    if not isinstance(content, str):
        raise _invalid_narrative_response("invalid_content_type")
    if not content.strip():
        raise _invalid_narrative_response("empty_content")
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
        raise _invalid_narrative_response("summary_word_limit_exceeded")
    if len(_words(narrative.detailed_explanation)) > detailed_max_words:
        raise _invalid_narrative_response("detail_word_limit_exceeded")
    if not set(narrative.evidence_references).issubset(allowed_references):
        raise _invalid_narrative_response("unknown_evidence_references")


def _words(value: str) -> list[str]:
    return value.split()


def _hash_json(value: dict[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
