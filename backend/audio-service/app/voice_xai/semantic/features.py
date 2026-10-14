"""Extract the ordered 165-feature input from original source audio.

Combine cropped acoustic measurements with full-source Glottal measurements,
preserve base values for shared names, and mark non-finite values for imputation.
"""
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import tempfile
import numpy as np
from numpy.typing import NDArray
from app.ingestion.audio import ProcessedAudio
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION

_LOCK = Lock()

class FeatureExtractionError(ValueError):
    pass

@dataclass(frozen=True)
class FeatureExtractionResult:
    feature_names: tuple[str, ...]
    values: NDArray[np.float64]
    sample_rate: int
    duration_seconds: float
    extractor_version: str
    metadata: dict

    def as_feature_mapping(self):
        return dict(zip(self.feature_names, map(float, self.values), strict=True))

class AcousticFeatureExtractor:
    """Combine the acoustic and Glottal paths while preserving their preprocessing."""
    def extract_file(self, path: Path) -> FeatureExtractionResult:
        with _LOCK:
            from app.voice_xai.semantic import recipe
            y, sr = recipe.load_audio_path(path)
            combined = recipe.extract_acoustic_features(y, sr)
            selected = recipe.extract_selected20_features_exact(path)
            for name, value in selected.items():
                if name not in combined:
                    combined[name] = value
            values = np.asarray([combined[name] for name in FEATURE_NAMES], dtype=np.float64)
            values[~np.isfinite(values)] = np.nan
            return FeatureExtractionResult(
                FEATURE_NAMES, values, sr, len(y) / sr, FEATURE_EXTRACTION_VERSION,
                {"feature_count": len(FEATURE_NAMES), "analysis_scope": "whole_clip",
                 "acoustic_crop_seconds": 6.0, "glottal_scope": "full_original_audio"},
            )

    def extract_processed_audio(self, audio: ProcessedAudio) -> FeatureExtractionResult:
        if not audio.source_audio_bytes:
            raise FeatureExtractionError("Semantic feature extraction requires original source audio bytes.")
        extension = audio.source_audio_extension or ".wav"
        if extension not in {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus", ".webm"}:
            raise FeatureExtractionError("Unsupported source audio extension.")
        with tempfile.TemporaryDirectory(prefix="semantic-audio-") as directory:
            path = Path(directory) / ("source" + extension)
            path.write_bytes(audio.source_audio_bytes)
            return self.extract_file(path)
