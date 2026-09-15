from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import ModelMode


class ModelHealthResponse(BaseModel):
    branch_name: str | None = None
    model_name: str
    display_name: str
    adapter_type: Literal["dummy", "real"] | str | None = None
    mode: ModelMode
    is_loaded: bool
    lifecycle_state: str | None = None
    ready: bool | None = None
    research_ready: bool | None = None
    requested_device: str | None = None
    resolved_device: str | None = None
    fallback_used: bool | None = None
    precision: str | None = None
    checkpoint_configured: bool | None = None
    checkpoint_valid: bool | None = None
    checkpoint_hash_short: str | None = None
    model_version: str | None = None
    architecture: str | None = None
    preprocessing_version: str | None = None
    last_error_code: str | None = None
    last_load_error_code: str | None = None
    last_inference_error_code: str | None = None
    last_transition_at: str | None = None
    branch_timeout_seconds: float | None = Field(default=None, gt=0)
    uses_dummy_mode: bool
    warning: str | None = None
