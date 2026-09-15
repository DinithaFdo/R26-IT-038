from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClassifyRequest(BaseModel):
    """Request payload for the text classification endpoint."""

    text: str = Field(..., min_length=1, description="Text to classify. Must not be empty.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"text": "The quick brown fox jumps over the lazy dog."}
            ]
        }
    )


class TokenHighlight(BaseModel):
    """A single token/word with its explainability score for highlighting."""

    word: str = Field(..., description="The word or token being scored.")
    score: float = Field(..., ge=0.0, le=1.0, description="Importance score between 0.0 and 1.0.")
    level: Literal["high", "mid", "low"] = Field(
        ..., description="Discretized importance bucket for UI highlighting."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"word": "quick", "score": 0.87, "level": "high"}
            ]
        }
    )


class SanitizationResult(BaseModel):
    """Result of running input sanitization/attack-detection on the request text."""

    was_attacked: bool = Field(..., description="Whether an adversarial attack was detected in the input.")
    attack_report: list[str] = Field(
        default_factory=list, description="Human-readable descriptions of detected attacks, if any."
    )
    clean_text: str = Field(..., description="The sanitized text used for classification.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "was_attacked": False,
                    "attack_report": [],
                    "clean_text": "The quick brown fox jumps over the lazy dog.",
                }
            ]
        }
    )


class SignalAnalysis(BaseModel):
    """CSS-v2 stylometric conflict signal analysis result."""

    css_conflict_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Signed margin conflict score (0=agree, 1=max conflict). "
                    "Null if text is too short for stylometric analysis."
    )
    conflict_level: str = Field(
        default="N/A",
        description="Conflict severity: LOW | MODERATE | HIGH | N/A"
    )
    conflict_detected: bool = Field(
        default=False,
        description="True if CSS conflict score exceeds 0.30 threshold."
    )
    deberta_direction: str = Field(
        ...,
        description="Direction Model 2 is pointing: AI or Human."
    )
    xgboost_direction: Optional[str] = Field(
        default=None,
        description="Direction Branch 2 XGBoost is pointing: AI or Human. "
                    "Null if text is too short."
    )
    shap_ai_signals: list[str] = Field(
        default_factory=list,
        description="Top stylometric features pushing toward AI classification."
    )
    shap_human_signals: list[str] = Field(
        default_factory=list,
        description="Top stylometric features pushing toward Human classification."
    )
    word_count: int = Field(
        ...,
        description="Word count of the input text."
    )
    note: Optional[str] = Field(
        default=None,
        description="Informational note, e.g. when text is too short for CSS."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "css_conflict_score": 0.492,
                    "conflict_level": "MODERATE",
                    "conflict_detected": True,
                    "deberta_direction": "AI",
                    "xgboost_direction": "Human",
                    "shap_ai_signals": ["Phrase repetition density", "Formal vocabulary density"],
                    "shap_human_signals": ["Contraction usage", "Personal pronoun usage"],
                    "word_count": 203,
                    "note": None
                }
            ]
        }
    )


class ClassifyResponse(BaseModel):
    """Response payload for the text classification endpoint."""

    label: Literal["AI-Generated", "Human-Written"] = Field(
        ..., description="The predicted class label."
    )
    prob_ai: float = Field(..., ge=0.0, le=1.0, description="Predicted probability that the text is AI-generated.")
    prob_human: float = Field(..., ge=0.0, le=1.0, description="Predicted probability that the text is human-written.")
    sanitization: SanitizationResult = Field(..., description="Details of the input sanitization step.")
    token_highlights: list[TokenHighlight] = Field(
        default_factory=list, description="Per-token explainability highlights."
    )
    model_version: str = Field(..., description="Version identifier of the model used for prediction.")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds.")
    signal_analysis: Optional[SignalAnalysis] = Field(
        default=None,
        description="CSS-v2 stylometric conflict signal analysis. "
                    "Null if text is below minimum word count."
    )
    final_label: str = Field(
        description=(
            "User-facing classification label. "
            "'AI-Generated' | 'Human-Written' | 'Mixed'. "
            "Set to 'Mixed' when CSS conflict_level is HIGH."
        )
    )
    leans_toward: Optional[str] = Field(
        default=None,
        description=(
            "Only populated when final_label is 'Mixed'. "
            "Shows DeBERTa original direction. "
            "'AI-Generated' or 'Human-Written'."
        )
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "label": "AI-Generated",
                    "prob_ai": 0.93,
                    "prob_human": 0.07,
                    "sanitization": {
                        "was_attacked": False,
                        "attack_report": [],
                        "clean_text": "The quick brown fox jumps over the lazy dog.",
                    },
                    "token_highlights": [
                        {"word": "quick", "score": 0.87, "level": "high"},
                        {"word": "the", "score": 0.12, "level": "low"},
                    ],
                    "model_version": "v1.0.0",
                    "processing_time_ms": 42.5,
                }
            ]
        }
    )
