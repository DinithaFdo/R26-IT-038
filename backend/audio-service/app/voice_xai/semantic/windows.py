"""Deterministic sliding-window inputs and outputs for semantic SHAP."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from app.ingestion.audio import ProcessedAudio
from app.schemas.xai import SemanticFeatureContribution
from app.voice_xai.semantic.features import (
    AcousticFeatureExtractor,
    FeatureExtractionError,
    FeatureExtractionResult,
)


class SemanticWindowError(ValueError):
    """Raised when processed audio cannot produce semantic analysis windows."""


@dataclass(frozen=True)
class SemanticWindowConfig:
    """Versioned, unpadded analysis-window configuration.

    The extractor is intentionally run over each clipped waveform directly.  We
    never right-pad a short final interval because doing so would manufacture
    acoustic evidence and make its timestamps misleading.
    """

    duration_seconds: float = 1.0
    overlap_seconds: float = 0.5
    version: str = "semantic-sliding-window-v1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0.0
        ):
            raise ValueError("duration_seconds must be finite and positive.")
        if not math.isfinite(self.overlap_seconds):
            raise ValueError("overlap_seconds must be finite.")
        if not 0.0 <= self.overlap_seconds < self.duration_seconds:
            raise ValueError(
                "overlap_seconds must be non-negative and shorter than duration."
            )
        if not self.version.strip():
            raise ValueError("Semantic window configuration version is required.")

    @property
    def step_seconds(self) -> float:
        return self.duration_seconds - self.overlap_seconds


@dataclass(frozen=True)
class SemanticFeatureWindow:
    """One exact audio interval and its frozen v4 feature vector."""

    start_seconds: float
    end_seconds: float
    start_sample: int
    end_sample: int
    features: FeatureExtractionResult

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise SemanticWindowError("Semantic window sample bounds are invalid.")
        if self.start_seconds < 0.0 or self.end_seconds <= self.start_seconds:
            raise SemanticWindowError("Semantic window timestamps are invalid.")


@dataclass(frozen=True)
class SemanticWindow:
    """Per-window SHAP evidence with timestamps safe for temporal comparison."""

    start_seconds: float
    end_seconds: float
    contributions: tuple[SemanticFeatureContribution, ...]
    spoof_probability: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0.0 or self.end_seconds <= self.start_seconds:
            raise SemanticWindowError("Semantic window timestamps are invalid.")
        if not self.contributions:
            raise SemanticWindowError(
                "Semantic windows require at least one contribution."
            )
        if not (
            math.isfinite(self.spoof_probability)
            and 0.0 <= self.spoof_probability <= 1.0
        ):
            raise SemanticWindowError("Semantic window model values are invalid.")
        for contribution in self.contributions:
            if (
                contribution.start_seconds != self.start_seconds
                or contribution.end_seconds != self.end_seconds
            ):
                raise SemanticWindowError(
                    "Semantic contribution timestamps must match their window."
                )


def extract_semantic_feature_windows(
    audio: ProcessedAudio,
    *,
    extractor: AcousticFeatureExtractor | None = None,
    config: SemanticWindowConfig | None = None,
) -> tuple[SemanticFeatureWindow, ...]:
    """Extract the approved v4 features over deterministic overlapping windows."""

    feature_extractor = extractor or AcousticFeatureExtractor()
    window_config = config or SemanticWindowConfig()
    # Match the notebook training recipe. The classifier's normalised waveform
    # remains reserved for classifier/temporal logic; semantic XGBoost uses the
    # bounded pre-normalisation copy retained by ingestion when available.
    source_waveform = (
        audio.unnormalised_waveform
        if audio.unnormalised_waveform is not None
        else audio.waveform
    )
    waveform = np.asarray(source_waveform, dtype=np.float32).squeeze()
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise SemanticWindowError(
            "Processed audio must contain a finite mono waveform."
        )
    if audio.sample_rate != feature_extractor.config.sample_rate:
        raise SemanticWindowError(
            "Processed audio sample rate does not match the semantic feature extractor."
        )

    duration_samples = _seconds_to_samples(
        window_config.duration_seconds, audio.sample_rate
    )
    step_samples = _seconds_to_samples(window_config.step_seconds, audio.sample_rate)
    minimum_samples = feature_extractor.config.frame_length_samples
    if duration_samples < minimum_samples:
        raise SemanticWindowError(
            "Semantic window duration is shorter than the feature extractor frame length."
        )
    if waveform.size < minimum_samples:
        raise SemanticWindowError(
            "Processed audio is too short for semantic window analysis."
        )

    windows: list[SemanticFeatureWindow] = []
    for start_sample in _window_start_samples(
        waveform_size=waveform.size,
        duration_samples=duration_samples,
        step_samples=step_samples,
        minimum_samples=minimum_samples,
    ):
        end_sample = min(start_sample + duration_samples, waveform.size)
        segment = waveform[start_sample:end_sample]
        start_seconds = start_sample / audio.sample_rate
        end_seconds = end_sample / audio.sample_rate
        try:
            features = feature_extractor.extract_waveform(
                segment,
                sample_rate=audio.sample_rate,
                metadata={
                    "preprocessing_version": audio.preprocessing_version,
                    "semantic_window_version": window_config.version,
                    "semantic_window_start_seconds": start_seconds,
                    "semantic_window_end_seconds": end_seconds,
                    "semantic_window_start_sample": start_sample,
                    "semantic_window_end_sample": end_sample,
                },
            )
        except FeatureExtractionError as error:
            raise SemanticWindowError(
                "Semantic feature extraction failed for a window."
            ) from error
        windows.append(
            SemanticFeatureWindow(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                start_sample=start_sample,
                end_sample=end_sample,
                features=features,
            )
        )
    return tuple(windows)


def _window_start_samples(
    *,
    waveform_size: int,
    duration_samples: int,
    step_samples: int,
    minimum_samples: int,
) -> tuple[int, ...]:
    if waveform_size < duration_samples:
        return (0,) if waveform_size >= minimum_samples else ()
    last_full_window_start = waveform_size - duration_samples
    starts = list(range(0, last_full_window_start + 1, step_samples))
    if starts[-1] != last_full_window_start:
        starts.append(last_full_window_start)
    return tuple(starts)


def _seconds_to_samples(seconds: float, sample_rate: int) -> int:
    samples = int(round(seconds * sample_rate))
    if samples < 1:
        raise SemanticWindowError(
            "Semantic window configuration resolves to zero samples."
        )
    return samples


def unavailable_window_reason() -> str:
    """Prevent clip-level values from being misrepresented as local evidence."""

    return (
        "No timestamp-level localization is available because sliding-window "
        "acoustic extraction and per-window SHAP results are absent; "
        "temporal-semantic IoU must remain not computed."
    )
