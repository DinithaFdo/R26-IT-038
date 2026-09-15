from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.provenance import PredictionProvenance

PROBABILITY_SUM_TOLERANCE = 0.01


class ProbabilityScores(BaseModel):
    bonafide: float = Field(ge=0.0, le=1.0)
    spoof: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probability_sum(self) -> "ProbabilityScores":
        if abs((self.bonafide + self.spoof) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                "Bonafide and spoof probabilities must approximately sum to 1."
            )
        return self


class BranchPrediction(BaseModel):
    model_name: str
    display_name: str
    status: BranchStatus
    mode: ModelMode
    prediction: PredictionLabel | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: ProbabilityScores | None = None
    processing_time_ms: float = Field(ge=0.0)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_branch_consistency(self) -> "BranchPrediction":
        if self.status == BranchStatus.failed and self.prediction is not None:
            raise ValueError("A failed branch must not include a prediction.")
        _validate_prediction_confidence(
            prediction=self.prediction,
            confidence=self.confidence,
            probabilities=self.probabilities,
        )
        return self


class AudioStorageMetadata(BaseModel):
    status: BranchStatus
    asset_id: str | None = None
    public_id: str | None = None
    resource_type: str | None = None
    version: int | None = None
    format: str | None = None
    bytes: int | None = Field(default=None, ge=0)
    duration: float | None = Field(default=None, ge=0.0)
    created_at: datetime | None = None
    error: str | None = None


class AudioMetadata(BaseModel):
    original_filename: str
    content_type: str
    original_extension: str | None = None
    detected_container: str | None = None
    detected_format: str | None = None
    detected_codec: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    file_size_bytes: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    storage: AudioStorageMetadata | None = None

    @model_validator(mode="after")
    def fill_size_aliases(self) -> "AudioMetadata":
        if self.size_bytes is None:
            self.size_bytes = self.file_size_bytes
        if self.detected_container is None and self.detected_format is not None:
            self.detected_container = self.detected_format
        if self.detected_format is None and self.detected_container is not None:
            self.detected_format = self.detected_container
        return self


class FusionResult(BaseModel):
    status: BranchStatus
    prediction: PredictionLabel | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: ProbabilityScores | None = None
    method: str
    #: The decision threshold actually applied to `probabilities.spoof` to
    #: produce `prediction` (`spoof` iff `probabilities.spoof >= decision_threshold`).
    #: Always populated -- including on a failed fusion result -- so a client
    #: never has to read internal provenance to learn what threshold this
    #: deployment is configured with.
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    branch_weights: dict[str, float] = Field(default_factory=dict)
    contains_dummy_branches: bool
    eligible_for_research_evaluation: bool
    warning: str | None = None
    config_version: str | None = None
    minimum_successful_branches: int | None = Field(default=None, ge=1)
    contributing_branches: list[str] = Field(default_factory=list)
    excluded_branches: dict[str, str] = Field(default_factory=dict)
    #: Which stage of the MULTI-SCOPE architecture actually produced this
    #: result. ``partial`` means fewer than the full designed branch set
    #: contributed, so the score must not be read as full-system performance.
    system_stage: str = "partial"
    #: Machine-readable reasons this result is not research eligible. Empty
    #: when it is. Lets clients explain the gap instead of just flagging it.
    research_blockers: list[str] = Field(default_factory=list)
    #: Which fusion contract actually produced this result, e.g.
    #: ``"fusion-convex-4branch-v3"`` or ``"legacy-3branch-frozen-v1"``.
    #: Defaults preserve the identity every pre-Fusion-V3 result already had.
    fusion_version: str = "score-level-fusion-v1"
    #: Coarse fusion strategy label, e.g. ``"learned_constrained"``,
    #: ``"legacy_average"``, ``"development_average"``.
    fusion_mode: str = "development_average"
    #: True only when Fusion V3 was configured/available but this specific
    #: request's branches did not satisfy it (see ``VoiceService._fuse``), so
    #: the legacy 3-branch detector produced this result instead. Never true
    #: when the legacy detector is simply the deliberately configured primary.
    fallback_used: bool = False

    @model_validator(mode="after")
    def validate_fusion_consistency(self) -> "FusionResult":
        if self.status == BranchStatus.failed and self.prediction is not None:
            raise ValueError("A failed fusion result must not include a prediction.")
        if self.contains_dummy_branches and self.eligible_for_research_evaluation:
            raise ValueError(
                "Fusion with dummy branches is not eligible for research evaluation."
            )
        if self.research_blockers and self.eligible_for_research_evaluation:
            raise ValueError(
                "A fusion result with research blockers cannot be research eligible."
            )
        _validate_prediction_confidence(
            prediction=self.prediction,
            confidence=self.confidence,
            probabilities=self.probabilities,
        )
        return self


class VoicePredictionResponse(BaseModel):
    request_id: str
    audio: AudioMetadata
    branches: list[BranchPrediction]
    fusion: FusionResult
    provenance: PredictionProvenance | None = None
    total_processing_time_ms: float = Field(ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_dummy_fusion_consistency(self) -> "VoicePredictionResponse":
        contains_dummy_branch = any(
            branch.mode == ModelMode.dummy for branch in self.branches
        )
        if contains_dummy_branch and not self.fusion.contains_dummy_branches:
            raise ValueError("Fusion must identify when it contains dummy branches.")
        if contains_dummy_branch and self.fusion.eligible_for_research_evaluation:
            raise ValueError(
                "Responses with dummy branches are not eligible for research evaluation."
            )
        return self


def _validate_prediction_confidence(
    *,
    prediction: PredictionLabel | None,
    confidence: float | None,
    probabilities: ProbabilityScores | None,
) -> None:
    if prediction is None or confidence is None or probabilities is None:
        return

    scores = {
        PredictionLabel.bonafide: probabilities.bonafide,
        PredictionLabel.spoof: probabilities.spoof,
    }
    winning_label = max(scores, key=scores.get)
    winning_probability = scores[winning_label]

    if prediction != winning_label:
        raise ValueError("Prediction must match the winning probability label.")
    if abs(confidence - winning_probability) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError("Confidence must match the winning probability.")
