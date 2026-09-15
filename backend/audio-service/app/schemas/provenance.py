from pydantic import BaseModel, Field

from app.schemas.common import ModelMode


class PreprocessingProvenance(BaseModel):
    input_sample_rate: int = Field(gt=0)
    input_channels: int = Field(gt=0)
    input_duration_seconds: float = Field(ge=0.0)
    target_sample_rate: int = Field(gt=0)
    target_channels: int = Field(gt=0)
    resampled: bool
    mono_conversion_applied: bool
    normalisation_applied: bool
    preprocessing_version: str
    minimum_duration_seconds: float = Field(default=1.0, gt=0)
    model_window_duration_seconds: float = Field(default=6.0, gt=0)
    model_window_overlap_seconds: float = Field(default=1.0, ge=0)
    segment_count: int = Field(default=1, ge=1)
    padding_policy: str = "right_zero"
    trim_policy: str = "none"
    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None


class BranchModelProvenance(BaseModel):
    model_name: str
    mode: ModelMode
    model_version: str
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None
    architecture_version: str
    class_mapping_version: str
    preprocessing_compatibility_version: str
    framework_version: str
    device_type: str
    research_result: bool


class FusionProvenance(BaseModel):
    fusion_method: str
    configured_weights: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    threshold: float = Field(ge=0.0, le=1.0)
    fusion_version: str
    fusion_mode: str | None = None
    fallback_used: bool = False
    contains_dummy_branches: bool
    eligible_for_research_evaluation: bool
    config_version: str | None = None
    minimum_successful_branches: int | None = Field(default=None, ge=1)
    contributing_branches: list[str] = Field(default_factory=list)
    excluded_branches: dict[str, str] = Field(default_factory=dict)


class PredictionProvenance(BaseModel):
    preprocessing: PreprocessingProvenance
    models: list[BranchModelProvenance]
    fusion: FusionProvenance
