from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import PredictionStatus, SourceType
from app.schemas.prediction import BranchPrediction, FusionResult
from app.schemas.prediction_history import PredictionHistoryAudioMetadata


class PredictionSubmissionResponse(BaseModel):
    prediction_id: str = Field(description="Stable prediction identifier.")
    request_id: str = Field(description="Request identifier assigned by the API.")
    status: PredictionStatus = Field(description="Current prediction lifecycle status.")
    source_type: SourceType = Field(description="How the audio was submitted.")
    audio: PredictionHistoryAudioMetadata = Field(
        description="Validated audio metadata and playback availability."
    )
    branches: list[BranchPrediction] = Field(
        description="Per-branch classifier outputs exposed by the existing API."
    )
    fusion: FusionResult | None = Field(
        description="Score-level fusion result when branch outputs are usable."
    )
    research_eligible: bool = Field(
        description=(
            "Whether this prediction can be used as a research result under the "
            "current verification policy."
        )
    )
    created_at: datetime = Field(description="Prediction creation timestamp.")


class PredictionJobStatusResponse(BaseModel):
    prediction_id: str = Field(description="Stable prediction identifier.")
    request_id: str = Field(description="Request identifier assigned by the API.")
    status: PredictionStatus = Field(description="Current prediction lifecycle status.")
    source_type: SourceType = Field(description="How the audio was submitted.")
    created_at: datetime = Field(description="Prediction creation timestamp.")
    updated_at: datetime = Field(description="Last status update timestamp.")
    completed_at: datetime | None = Field(
        default=None,
        description="Terminal completion/failure timestamp when available.",
    )
    error_summary: list[dict] = Field(
        default_factory=list,
        description="Sanitized stage/code summaries for failed prediction work.",
    )
