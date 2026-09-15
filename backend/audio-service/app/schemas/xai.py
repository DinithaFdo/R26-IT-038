"""Versioned public and persistence schemas for Voice XAI explanations."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import BranchStatus, ModelMode, PredictionLabel

XAI_SCHEMA_VERSION = "voice-xai-api-v1"


class XaiSchemaModel(BaseModel):
    """Strict base model used to prevent silent XAI contract drift."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CanonicalXaiBranch(str, Enum):
    lfcc_cnn_tcn = "lfcc_cnn_tcn"
    aasist = "aasist"
    ssl_sequence = "ssl_sequence"
    glottal = "glottal"


class ExplanationStatus(str, Enum):
    queued = "queued"
    running = "running"
    partial = "partial"
    completed = "completed"
    failed = "failed"
    blocked = "blocked"


class ExplanationComponent(str, Enum):
    temporal = "temporal"
    semantic = "semantic"
    report = "report"
    narrative = "narrative"


class ComponentStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    blocked = "blocked"
    not_available = "not_available"


class MetricStatus(str, Enum):
    available = "available"
    not_computed = "not_computed"
    not_applicable = "not_applicable"


class ShapDirection(str, Enum):
    toward_spoof = "toward_spoof"
    toward_bonafide = "toward_bonafide"


class SemanticTargetType(str, Enum):
    classifier_surrogate = "classifier_surrogate"
    independent_acoustic_evidence_model = "independent_acoustic_evidence_model"


class ReportDisposition(str, Enum):
    spoof_suspected = "spoof_suspected"
    bonafide_suspected = "bonafide_suspected"
    inconclusive = "inconclusive"


class ClassifierBranchSnapshot(XaiSchemaModel):
    branch_name: CanonicalXaiBranch
    model_name: str = Field(min_length=1)
    status: BranchStatus
    mode: ModelMode
    spoof_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_success_probability(self) -> "ClassifierBranchSnapshot":
        if self.status == BranchStatus.success and self.spoof_probability is None:
            raise ValueError("Successful branch snapshots require spoof_probability.")
        if self.status != BranchStatus.success and self.spoof_probability is not None:
            raise ValueError("Non-successful branch snapshots cannot include a score.")
        return self


class ClassifierAuxiliaryEvidence(XaiSchemaModel):
    """Evidence that informs explanations but never the primary fusion decision.

    Currently just Glottal: per the final research detector contract
    (``model_artifacts/fusion/final_detector_v1.json``), it is physiological/
    XAI evidence only. ``glottal_spoof_probability`` is ``None`` whenever the
    branch did not produce a score (disabled, failed, or not yet
    implemented) -- absence here must never be read as a bonafide-leaning
    score of 0. ``used_for_primary_decision`` is always ``False`` and exists
    so a consumer never has to infer that fact from field absence.
    """

    glottal_spoof_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    used_for_primary_decision: Literal[False] = False


class ClassifierSnapshot(XaiSchemaModel):
    verdict: PredictionLabel | None = None
    spoof_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    bonafide_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    contains_dummy_branches: bool
    research_eligible: bool
    branches: list[ClassifierBranchSnapshot] = Field(default_factory=list)
    auxiliary_evidence: ClassifierAuxiliaryEvidence = Field(
        default_factory=ClassifierAuxiliaryEvidence
    )

    @model_validator(mode="after")
    def validate_classifier_snapshot(self) -> "ClassifierSnapshot":
        probabilities = (self.spoof_probability, self.bonafide_probability)
        if (probabilities[0] is None) != (probabilities[1] is None):
            raise ValueError("Classifier probabilities must be supplied together.")
        if probabilities[0] is not None and probabilities[1] is not None:
            if abs(probabilities[0] + probabilities[1] - 1.0) > 0.01:
                raise ValueError(
                    "Classifier probabilities must approximately sum to 1."
                )
            winning_label = (
                PredictionLabel.spoof
                if probabilities[0] >= probabilities[1]
                else PredictionLabel.bonafide
            )
            winning_probability = max(probabilities)
            if self.verdict is not None and self.verdict != winning_label:
                raise ValueError(
                    "Classifier verdict must match the winning probability."
                )
            if (
                self.confidence is not None
                and abs(self.confidence - winning_probability) > 0.01
            ):
                raise ValueError(
                    "Classifier confidence must match the winning probability."
                )
        actual_dummy = any(branch.mode == ModelMode.dummy for branch in self.branches)
        if actual_dummy != self.contains_dummy_branches:
            raise ValueError("Classifier dummy-branch summary is inconsistent.")
        if self.contains_dummy_branches and self.research_eligible:
            raise ValueError("Dummy classifier results are not research eligible.")
        return self


class ExplanationArtifactReference(XaiSchemaModel):
    artifact_id: str = Field(min_length=1)
    kind: Literal[
        "temporal_evidence",
        "attention_visualization",
        "attention_spectrogram",
        "attention_tensor",
        "intermediate_representations",
        "semantic_values",
        "report",
        "other",
    ]
    content_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at: datetime | None = None
    download_path: str | None = None


class TemporalRegion(XaiSchemaModel):
    region_id: int = Field(ge=1)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    # XLS-R rollout density is normalized around a uniform-attention baseline
    # of 1.0, rather than being a probability.  A local concentration can
    # therefore legitimately exceed 1.0.
    attention_score: float = Field(ge=0.0)
    @model_validator(mode="after")
    def validate_time_order(self) -> "TemporalRegion":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Temporal region end must be after its start.")
        return self


class TemporalExplanation(XaiSchemaModel):
    status: ComponentStatus
    method_version: str = Field(min_length=1)
    model_version: str | None = None
    development_placeholder: bool = True
    research_eligible: bool = False
    score_kind: Literal["high_attention"] = "high_attention"
    attention_score_peak: float | None = Field(default=None, ge=0.0)
    threshold_percentile: float | None = Field(default=None, ge=0.0, le=100.0)
    attention_threshold: float | None = Field(default=None, ge=0.0)
    high_attention_region_count: int | None = Field(default=None, ge=0)
    high_attention_regions: list[TemporalRegion] = Field(default_factory=list)
    high_attention_combined_duration_seconds: float | None = Field(
        default=None, ge=0.0
    )
    # The fields above are the fixed PartialSpoof DEV-calibrated evaluation
    # result. These fields are a separate per-clip display rule for the UI.
    visualization_threshold_percentile: float | None = Field(
        default=None, ge=50.0, le=99.0
    )
    visualization_attention_threshold: float | None = Field(default=None, ge=0.0)
    visualization_high_attention_region_count: int | None = Field(
        default=None, ge=0
    )
    visualization_high_attention_regions: list[TemporalRegion] | None = None
    visualization_high_attention_combined_duration_seconds: float | None = Field(
        default=None, ge=0.0
    )
    artifacts: list[ExplanationArtifactReference] = Field(default_factory=list)
    warning: str | None = None

    @model_validator(mode="after")
    def fill_and_validate_high_attention_region_count(self) -> "TemporalExplanation":
        actual_high_attention_count = len(self.high_attention_regions)
        if self.high_attention_region_count is None:
            self.high_attention_region_count = actual_high_attention_count
        elif self.high_attention_region_count != actual_high_attention_count:
            raise ValueError(
                "high_attention_region_count must match the high_attention_regions list."
            )
        if self.visualization_high_attention_regions is not None:
            actual_visualization_count = len(
                self.visualization_high_attention_regions
            )
            if self.visualization_high_attention_region_count is None:
                self.visualization_high_attention_region_count = actual_visualization_count
            elif self.visualization_high_attention_region_count != actual_visualization_count:
                raise ValueError(
                    "visualization_high_attention_region_count must match the "
                    "visualization_high_attention_regions list."
                )
        return self


class SemanticFeatureContribution(XaiSchemaModel):
    rank: int = Field(ge=1)
    feature_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    value: float
    unit: str | None = None
    reference_summary: str | None = None
    shap_value: float
    direction: ShapDirection
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_optional_window(self) -> "SemanticFeatureContribution":
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("Semantic window start and end must be supplied together.")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds <= self.start_seconds
        ):
            raise ValueError("Semantic window end must be after its start.")
        return self


class SemanticEvidenceWindow(XaiSchemaModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    feature_importance: list[SemanticFeatureContribution] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window(self) -> "SemanticEvidenceWindow":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Semantic window end must be after its start.")
        return self


class SemanticClipSummary(XaiSchemaModel):
    """One clip-level decision from the independent semantic evidence model.

    This is intentionally separate from ``ClassifierSnapshot``: semantic XAI
    explains an auxiliary acoustic model and must never be presented as the
    primary fused classifier decision.
    """

    spoof_probability: float = Field(ge=0.0, le=1.0)
    bonafide_probability: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(gt=0.0, lt=1.0)
    predicted_label: PredictionLabel
    aggregation_method: Literal[
        "coverage_weighted_mean_window_probability"
    ] = "coverage_weighted_mean_window_probability"
    window_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_summary(self) -> "SemanticClipSummary":
        if abs(self.spoof_probability + self.bonafide_probability - 1.0) > 0.01:
            raise ValueError("Semantic clip probabilities must sum to one.")
        expected_label = (
            PredictionLabel.spoof
            if self.spoof_probability >= self.decision_threshold
            else PredictionLabel.bonafide
        )
        if self.predicted_label != expected_label:
            raise ValueError(
                "Semantic clip predicted_label must match the decision threshold."
            )
        return self


class SemanticExplanation(XaiSchemaModel):
    status: ComponentStatus
    target_type: SemanticTargetType
    model_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    output_space: Literal["raw_margin", "log_odds", "probability"]
    development_placeholder: bool = True
    research_eligible: bool = False
    base_value: float | None = None
    predicted_value: float | None = None
    clip_summary: SemanticClipSummary | None = None
    feature_importance: list[SemanticFeatureContribution] = Field(default_factory=list)
    windows: list[SemanticEvidenceWindow] = Field(default_factory=list)
    warning: str | None = None


class MetricEvidence(XaiSchemaModel):
    status: MetricStatus
    value: float | None = None
    scope: Literal["per_analysis", "offline_validation"]
    reason: str | None = None
    dataset_version: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "MetricEvidence":
        if self.status == MetricStatus.available and self.value is None:
            raise ValueError("Available metrics require a value.")
        if self.status != MetricStatus.available and not self.reason:
            raise ValueError("Unavailable metrics require a reason.")
        return self


class ExplanationQuality(XaiSchemaModel):
    surrogate_fidelity_r2: MetricEvidence | None = None
    temporal_semantic_iou: MetricEvidence | None = None

    @model_validator(mode="before")
    @classmethod
    def discard_uncomputed_legacy_metrics(cls, value):
        """Keep immutable records written before the quality contract narrowed."""

        if not isinstance(value, dict):
            return value
        cleaned = dict(value)
        for field_name in ("temporal_localization_iou", "robustness_stability"):
            if cleaned.get(field_name) is not None:
                raise ValueError(
                    f"Legacy metric {field_name} cannot contain a measured value."
                )
            cleaned.pop(field_name, None)
        return cleaned


class CombinedExplanationReport(XaiSchemaModel):
    status: ComponentStatus
    disposition: ReportDisposition
    requires_human_review: bool = True
    finding: str = Field(min_length=1)
    primary_evidence: str = Field(min_length=1)
    quality_checks: str = Field(min_length=1)
    limitation: str | None = None
    recommendation: str = Field(min_length=1)
    disclaimer: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_completed_report(self) -> "CombinedExplanationReport":
        if self.status != ComponentStatus.completed:
            raise ValueError("A persisted combined report must be completed.")
        return self


class NarrativeExplanation(XaiSchemaModel):
    """Validated optional natural-language rendering of existing XAI evidence.

    The deterministic report remains authoritative.  This object exists only
    after an upstream model returned JSON that passed local bounds and
    evidence-reference validation.
    """

    status: Literal[ComponentStatus.completed] = ComponentStatus.completed
    provider: Literal["alibaba_model_studio"]
    model_id: str = Field(min_length=1, max_length=160)
    prompt_version: str = Field(min_length=1, max_length=80)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: str = Field(min_length=1, max_length=2500)
    detailed_explanation: str = Field(min_length=1, max_length=6000)
    evidence_references: list[str] = Field(min_length=1, max_length=32)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_evidence_references(self) -> "NarrativeExplanation":
        if len(set(self.evidence_references)) != len(self.evidence_references):
            raise ValueError("Narrative evidence references must be unique.")
        for reference in self.evidence_references:
            if not reference or len(reference) > 80:
                raise ValueError("Narrative evidence reference is invalid.")
        return self


class ExplanationProvenance(XaiSchemaModel):
    schema_version: str = XAI_SCHEMA_VERSION
    pipeline_version: str = Field(min_length=1)
    classifier_contract_version: str = Field(min_length=1)
    feature_extractor_version: str | None = None
    semantic_model_version: str | None = None
    semantic_model_hash_short: str | None = None
    temporal_model_version: str | None = None
    temporal_model_hash_short: str | None = None
    configuration_hash: str | None = None
    configuration_hash_inputs: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExplanationError(XaiSchemaModel):
    component: ExplanationComponent | None = None
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=500)


class ExplanationComponentStatuses(XaiSchemaModel):
    temporal: ComponentStatus = ComponentStatus.queued
    semantic: ComponentStatus = ComponentStatus.queued
    report: ComponentStatus = ComponentStatus.queued
    # Legacy persisted runs predate the optional narrative component.  The
    # terminal default keeps those immutable documents readable.
    narrative: ComponentStatus = ComponentStatus.not_available


class ExplanationComponentErrors(XaiSchemaModel):
    temporal: ExplanationError | None = None
    semantic: ExplanationError | None = None
    report: ExplanationError | None = None
    narrative: ExplanationError | None = None


class XaiExplanationResponse(XaiSchemaModel):
    schema_version: str = XAI_SCHEMA_VERSION
    explanation_id: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    development_placeholder: bool = True
    research_eligible: bool = False
    status: ExplanationStatus
    component_statuses: ExplanationComponentStatuses
    component_errors: ExplanationComponentErrors = Field(
        default_factory=ExplanationComponentErrors
    )
    classifier_snapshot: ClassifierSnapshot
    temporal: TemporalExplanation | None = None
    semantic: SemanticExplanation | None = None
    combined_report: CombinedExplanationReport | None = None
    narrative: NarrativeExplanation | None = None
    quality: ExplanationQuality = Field(default_factory=ExplanationQuality)
    provenance: ExplanationProvenance
    artifacts: list[ExplanationArtifactReference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: ExplanationError | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_lifecycle_consistency(self) -> "XaiExplanationResponse":
        if self.development_placeholder and self.research_eligible:
            raise ValueError("Development placeholder results are not research eligible.")
        if self.research_eligible and not self.classifier_snapshot.research_eligible:
            raise ValueError("XAI research eligibility requires a research-eligible classifier.")
        terminal = self.status in {
            ExplanationStatus.completed,
            ExplanationStatus.failed,
        }
        if terminal and self.completed_at is None:
            raise ValueError("Terminal XAI responses require completed_at.")
        if not terminal and self.completed_at is not None:
            raise ValueError("Non-terminal XAI responses cannot include completed_at.")
        if self.status == ExplanationStatus.failed and self.error is None:
            raise ValueError("Failed XAI responses require a safe error.")
        if self.status == ExplanationStatus.completed and self.error is not None:
            raise ValueError("Completed XAI responses cannot include an error.")
        if self.status == ExplanationStatus.completed:
            component_values = (
                self.component_statuses.temporal,
                self.component_statuses.semantic,
                self.component_statuses.report,
            )
            if any(
                value in {ComponentStatus.queued, ComponentStatus.running}
                for value in component_values
            ):
                raise ValueError("Completed XAI responses require terminal components.")
            if self.component_statuses.report != ComponentStatus.completed:
                raise ValueError("Completed XAI responses require a completed report.")
        component_values = {
            ExplanationComponent.temporal: (
                self.component_statuses.temporal,
                self.component_errors.temporal,
                self.temporal,
            ),
            ExplanationComponent.semantic: (
                self.component_statuses.semantic,
                self.component_errors.semantic,
                self.semantic,
            ),
            ExplanationComponent.report: (
                self.component_statuses.report,
                self.component_errors.report,
                self.combined_report,
            ),
            ExplanationComponent.narrative: (
                self.component_statuses.narrative,
                self.component_errors.narrative,
                self.narrative,
            ),
        }
        failure_like = {
            ComponentStatus.failed,
            ComponentStatus.blocked,
            ComponentStatus.not_available,
        }
        for component, (component_status, component_error, result) in (
            component_values.items()
        ):
            # Backward-compatible read of terminal runs persisted before the
            # optional narrative component existed. New runs always carry an
            # explicit status/error for this component.
            if (
                component == ExplanationComponent.narrative
                and component_status == ComponentStatus.not_available
                and component_error is None
                and result is None
            ):
                continue
            if component_status in failure_like and component_error is None:
                raise ValueError(
                    f"{component.value} status requires a safe component error."
                )
            if component_status not in failure_like and component_error is not None:
                raise ValueError(
                    f"{component.value} status cannot include a component error."
                )
            if (
                component_error is not None
                and component_error.component is not None
                and component_error.component != component
            ):
                raise ValueError(
                    f"{component.value} error identifies a different component."
                )
            if component_status == ComponentStatus.completed and result is None:
                raise ValueError(
                    f"Completed {component.value} status requires its result."
                )
            if result is not None and result.status != component_status:
                raise ValueError(
                    f"{component.value} result status does not match component status."
                )
        return self
