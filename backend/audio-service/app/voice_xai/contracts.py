"""Frozen classifier-to-XAI integration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.models.runtime import (
    CANONICAL_BRANCH_ORDER,
    PUBLIC_MODEL_NAMES,
    CanonicalBranchName,
)
from app.schemas.common import SourceType
from app.schemas.prediction import VoicePredictionResponse
from app.voice_xai.capture.contracts import ExtractionBundle
from app.voice_xai.temporal.contracts import TemporalAttentionWindowInput

if TYPE_CHECKING:
    from app.ingestion.audio import ProcessedAudio

CLASSIFIER_XAI_CONTRACT_VERSION = "classifier-xai-v1"

CANONICAL_XAI_BRANCHES: tuple[CanonicalBranchName, ...] = CANONICAL_BRANCH_ORDER
PUBLIC_MODEL_NAME_BY_XAI_BRANCH: dict[CanonicalBranchName, str] = dict(
    PUBLIC_MODEL_NAMES
)


@dataclass(frozen=True)
class ClassifierInferenceBundle:
    """Internal handoff created after classifier inference and before XAI work."""

    prediction_id: str
    request_id: str
    owner_user_id: str
    source_type: SourceType
    prediction: VoicePredictionResponse
    extraction: ExtractionBundle | None = None
    # Generated inside the original XLS-R classifier forward pass. It is a
    # reduced, private representation, never a raw attention tensor artifact.
    temporal_evidence: tuple[TemporalAttentionWindowInput, ...] | None = None
    # Ephemeral only: it is never serialized or reconstructed from a prediction
    # document. The classifier integration can supply it for windowed XAI.
    processed_audio: "ProcessedAudio | None" = None
    contract_version: str = CLASSIFIER_XAI_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.prediction_id.strip():
            raise ValueError("prediction_id is required.")
        if not self.request_id.strip():
            raise ValueError("request_id is required.")
        if not self.owner_user_id.strip():
            raise ValueError("owner_user_id is required.")
        if self.contract_version != CLASSIFIER_XAI_CONTRACT_VERSION:
            raise ValueError("Unsupported classifier-to-XAI contract version.")
        if self.prediction.request_id != self.request_id:
            raise ValueError(
                "Classifier prediction request_id must match the handoff bundle."
            )
        extraction_request_mismatch = (
            self.extraction is not None
            and self.extraction.request_id != self.request_id
        )
        if extraction_request_mismatch:
            raise ValueError(
                "Extraction bundle request_id must match the handoff bundle."
            )
