from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import ModelMode, PredictionLabel, PredictionStatus, SourceType
from app.schemas.prediction import BranchPrediction, FusionResult


class PredictionModeSummary(BaseModel):
    dummy: int = Field(ge=0)
    real: int = Field(ge=0)
    contains_dummy: bool


class PredictionHistoryItem(BaseModel):
    prediction_id: str
    filename: str | None = None
    source_type: SourceType
    status: PredictionStatus
    duration_seconds: float | None = Field(default=None, ge=0.0)
    final_prediction: PredictionLabel | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mode_summary: PredictionModeSummary
    created_at: datetime


class PredictionHistoryListResponse(BaseModel):
    items: list[PredictionHistoryItem]
    page: int = Field(ge=1)
    limit: int = Field(ge=1, le=50)
    has_next: bool


class PredictionHistoryAudioMetadata(BaseModel):
    original_filename: str | None = None
    original_extension: str | None = None
    detected_container: str | None = None
    detected_codec: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    size_bytes: int | None = Field(default=None, ge=0)
    storage_status: str | None = None
    playback_available: bool = False


class PredictionDetailResponse(BaseModel):
    prediction_id: str
    request_id: str
    source_type: SourceType
    status: PredictionStatus
    audio: PredictionHistoryAudioMetadata
    branches: list[BranchPrediction]
    fusion: FusionResult | None = None
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    total_processing_time_ms: float | None = Field(default=None, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    research_eligible: bool
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class PredictionAudioPlaybackResponse(BaseModel):
    playback_url: str
    expires_in_seconds: int = Field(gt=0)


class PredictionDeleteResponse(BaseModel):
    prediction_id: str
    status: PredictionStatus
