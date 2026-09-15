from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class XAIRequest(BaseModel):
    text: str = Field(..., description="The input text to be analyzed and explained.")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata from previous pipeline stages.",
    )


class TokenHeatmap(BaseModel):
    original_token: str = Field(...)
    display_token: str = Field(...)
    is_noise: bool = Field(...)
    raw_score: float = Field(...)
    normalized_score: float = Field(...)


class XAIResponse(BaseModel):
    status: str = Field(default="success")
    predicted_class: str = Field(
        ...,
        description="The classification result (e.g., AI-Generated, Human-Written).",
    )
    confidence: float = Field(
        ..., description="The confidence percentage of the classification."
    )

    # --- Component 3 Data Pass-Through ---
    model_version: str = Field(
        default="Unknown", description="Model version identifier."
    )
    processing_time_ms: float = Field(
        default=0.0, description="Total pipeline processing time."
    )
    sanitization: Dict[str, Any] = Field(
        default_factory=dict, description="Attack detection and sanitization results."
    )
    signal_analysis: Optional[Dict[str, Any]] = Field(
        default=None, description="Stylometric XGBoost conflict signals."
    )

    # --- New Telemetry Field for Performance Monitoring ---
    telemetry: Optional[Dict[str, Any]] = Field(
        default=None, description="Granular latency breakdown in ms."
    )
    # -------------------------------------

    heatmap_data: List[TokenHeatmap] = Field(
        ..., description="List of token metrics for frontend heatmap generation."
    )
    interpretability_report: str = Field(
        ..., description="The LLM-translated, human-readable explanation."
    )
    final_label: str = Field(
        description=(
            "User-facing label. "
            "'AI-Generated' | 'Human-Written' | 'Mixed'. "
            "Passed through from classify()."
        )
    )
    leans_toward: Optional[str] = Field(
        default=None,
        description=(
            "DeBERTa direction when Mixed. "
            "'AI-Generated' | 'Human-Written' | None."
        )
    )
