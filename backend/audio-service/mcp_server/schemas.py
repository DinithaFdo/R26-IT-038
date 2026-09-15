from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import PredictionLabel, PredictionStatus, SourceType


class MCPErrorDetail(BaseModel):
    code: str
    message: str
    details: Any = None


class MCPToolResponse(BaseModel):
    ok: bool
    request_id: str
    data: dict[str, Any] | None = None
    error: MCPErrorDetail | None = None


class MCPCreatePredictionInput(BaseModel):
    api_token: str = Field(min_length=1)
    filename: str | None = Field(default=None, max_length=180)
    content_type: str | None = Field(default=None, max_length=120)
    client_filename: str | None = Field(default=None, max_length=180)
    idempotency_key: str | None = Field(default=None, max_length=120)
    secure_audio_asset_id: str | None = Field(default=None, max_length=180)
    signed_upload_reference: str | None = Field(default=None, max_length=400)
    small_audio_payload_base64: str | None = None

    @model_validator(mode="after")
    def validate_one_audio_source(self) -> "MCPCreatePredictionInput":
        sources = [
            bool(self.secure_audio_asset_id),
            bool(self.signed_upload_reference),
            bool(self.small_audio_payload_base64),
        ]
        if sum(sources) != 1:
            raise ValueError(
                "Provide exactly one audio source: secure_audio_asset_id, "
                "signed_upload_reference, or small_audio_payload_base64."
            )
        return self


class MCPGetPredictionInput(BaseModel):
    api_token: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1, max_length=160)


class MCPListPredictionsInput(BaseModel):
    api_token: str = Field(min_length=1)
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=50)
    status: PredictionStatus | None = None
    source_type: SourceType | None = None
    prediction_label: PredictionLabel | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None


class MCPDeletePredictionInput(BaseModel):
    api_token: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1, max_length=160)


class MCPModelStatusInput(BaseModel):
    api_token: str = Field(min_length=1)
