from enum import Enum

from pydantic import BaseModel, Field
from typing import Any


class PredictionLabel(str, Enum):
    bonafide = "bonafide"
    spoof = "spoof"


class BranchStatus(str, Enum):
    success = "success"
    failed = "failed"
    skipped = "skipped"


class ModelMode(str, Enum):
    dummy = "dummy"
    real = "real"
    #: Branch is intentionally not part of this deployment -- no trained model
    #: exists yet. It runs nothing and contributes nothing to fusion. Distinct
    #: from ``failed``, which is a branch that should have worked.
    disabled = "disabled"


class SourceType(str, Enum):
    dashboard_upload = "dashboard_upload"
    live_recording = "live_recording"
    public_api = "public_api"
    mcp = "mcp"


class PredictionStatus(str, Enum):
    queued = "queued"
    validating = "validating"
    storing = "storing"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    deleting = "deleting"
    deleted = "deleted"


VALID_PREDICTION_STATUS_TRANSITIONS: dict[
    PredictionStatus,
    set[PredictionStatus],
] = {
    PredictionStatus.queued: {
        PredictionStatus.validating,
        PredictionStatus.failed,
        PredictionStatus.deleting,
    },
    PredictionStatus.validating: {
        PredictionStatus.storing,
        PredictionStatus.failed,
        PredictionStatus.deleting,
    },
    PredictionStatus.storing: {
        PredictionStatus.processing,
        PredictionStatus.failed,
        PredictionStatus.deleting,
    },
    PredictionStatus.processing: {
        PredictionStatus.completed,
        PredictionStatus.failed,
        PredictionStatus.deleting,
    },
    PredictionStatus.completed: {
        PredictionStatus.deleting,
    },
    PredictionStatus.failed: {
        PredictionStatus.deleting,
    },
    PredictionStatus.deleting: {
        PredictionStatus.deleted,
        PredictionStatus.failed,
    },
    PredictionStatus.deleted: set(),
}


def is_valid_prediction_status_transition(
    current_status: PredictionStatus,
    next_status: PredictionStatus,
) -> bool:
    return next_status in VALID_PREDICTION_STATUS_TRANSITIONS[current_status]


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str = Field(description="Overall readiness status: ready or not_ready.")
    prediction_ready: bool = Field(
        default=False,
        description="True when dependencies required for prediction requests are ready.",
    )
    research_ready: bool = Field(
        default=False,
        description=(
            "True only when runtime readiness and research verification criteria "
            "are satisfied."
        ),
    )
    ffmpeg_available: bool = Field(description="Whether FFmpeg is available.")
    ffprobe_available: bool = Field(description="Whether ffprobe is available.")
    mongodb_configured: bool = Field(
        default=False,
        description="Whether MongoDB configuration is present.",
    )
    mongodb_available: bool = Field(
        default=False,
        description="Whether MongoDB responded to the readiness check.",
    )
    storage_enabled: bool = Field(
        default=False,
        description="Whether remote audio storage is enabled by configuration.",
    )
    storage_available: bool = Field(
        default=False,
        description="Whether configured audio storage is currently available.",
    )
    components: dict[str, Any] = Field(
        default_factory=dict,
        description="Component-level readiness details for operators and Swagger users.",
    )


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Sanitized user-facing error message.")
    details: Any = Field(
        default=None,
        description="Optional safe structured details when an endpoint provides them.",
    )


class ErrorResponse(BaseModel):
    request_id: str = Field(description="Server request identifier for support/debugging.")
    error: ErrorDetail = Field(description="Standard API error envelope.")
