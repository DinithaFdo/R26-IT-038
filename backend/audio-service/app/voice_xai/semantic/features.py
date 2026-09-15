"""Notebook-compatible acoustic features for the v4 semantic XAI model.

The order and parameters here mirror ``XGBoost_Surrogate_v4`` exactly. Any
change requires retraining and a new hashed semantic artifact manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.ingestion.audio import ProcessedAudio


FEATURE_EXTRACTION_VERSION = "voice-xai-acoustic-v4"
FEATURE_NAMES = (
    "jitter", "shimmer", "hnr", "gci_count", "gci_mean_interval",
    "gci_std_interval", "gci_irregularity", "gci_jitter_gci",
    *(f"formant_f{formant}_{statistic}" for formant in range(1, 4)
      for statistic in ("mean", "std")),
    *(f"lfcc_{index}" for index in range(1, 21)),
    *(f"{family}_{index}_{statistic}" for family in ("mfcc", "dmfcc", "ddmfcc")
      for index in range(1, 14) for statistic in ("mean", "std")),
    "f0_mean", "f0_std", "f0_median", "f0_range", "voiced_ratio",
    "rms_mean", "rms_std", "rms_dynamic_range_db",
    *(f"{name}_{statistic}" for name in (
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "spectral_flatness", "spectral_flux",
    ) for statistic in ("mean", "std")),
    *(f"spectral_contrast_band{band}_{statistic}" for band in range(1, 7)
      for statistic in ("mean", "std")),
    "cpp_mean", "cpp_std", "pause_ratio", "pause_count",
    "pause_mean_duration", "pause_max_duration",
)


class FeatureExtractionError(ValueError):
    """Raised when an audio clip cannot satisfy the v4 feature contract."""


@dataclass(frozen=True)
class FeatureExtractionConfig:
    sample_rate: int = 16_000
    max_audio_seconds: float = 6.0
    minimum_audio_seconds: float = 0.20
    frame_length_samples: int = 400
    hop_length_samples: int = 160
    fft_size: int = 512


@dataclass(frozen=True)
class FeatureExtractionResult:
    feature_names: tuple[str, ...]
    values: NDArray[np.float32]
    sample_rate: int
    duration_seconds: float
    extractor_version: str
    metadata: dict[str, Any]

    def as_feature_mapping(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.feature_names, self.values, strict=True)}

    def as_json_dict(self) -> dict[str, Any]:
        return {"extractor_version": self.extractor_version, "sample_rate": self.sample_rate,
                "duration_seconds": self.duration_seconds, "features": self.as_feature_mapping(),
                "metadata": self.metadata}


class AcousticFeatureExtractor:
    """Extract the notebook's 148 feature values from classifier-preprocessed audio."""

    def __init__(self, config: FeatureExtractionConfig | None = None) -> None:
        self.config = config or FeatureExtractionConfig()

    def extract_processed_audio(self, audio: ProcessedAudio) -> FeatureExtractionResult:
        # The v4 notebook trained on decoded PCM after DC removal, not the
        # classifier's peak-normalised branch input.  Ingestion retains this
        # bounded, same-request waveform exactly for model-specific consumers.
        waveform = audio.unnormalised_waveform if audio.unnormalised_waveform is not None else audio.waveform
        return self.extract_waveform(waveform, sample_rate=audio.sample_rate, metadata={
            "preprocessing_version": audio.preprocessing_version,
            "classifier_window_count": len(audio.segments),
            "normalisation_applied": audio.normalisation_applied,
            "waveform_source": "unnormalised_pre_classifier" if audio.unnormalised_waveform is not None else "classifier_waveform_fallback",
        })

    def extract_waveform(self, waveform: NDArray[np.floating[Any]], *, sample_rate: int,
                         metadata: dict[str, Any] | None = None) -> FeatureExtractionResult:
        input_values = np.asarray(waveform, dtype=np.float32).squeeze()
        if input_values.ndim != 1 or not input_values.size:
            raise FeatureExtractionError("Waveform must contain a non-empty mono signal.")
        if not np.isfinite(input_values).all():
            raise FeatureExtractionError("Waveform contains non-finite values.")
        librosa, parselmouth, fft, signal = _load_feature_runtime()
        y = self._prepare_waveform(input_values, sample_rate, librosa)
        mapping = _extract_feature_mapping(y, self.config.sample_rate, librosa, parselmouth, fft, signal)
        values = np.asarray([mapping[name] for name in FEATURE_NAMES], dtype=np.float32)
        if np.isinf(values).any():
            invalid = [name for name, value in zip(FEATURE_NAMES, values, strict=True) if np.isinf(value)]
            raise FeatureExtractionError(
                "The v4 feature recipe produced infinite values "
                f"({invalid[:5]})."
            )
        missing = [name for name, value in zip(FEATURE_NAMES, values, strict=True) if np.isnan(value)]
        return FeatureExtractionResult(FEATURE_NAMES, values, self.config.sample_rate,
            float(y.size / self.config.sample_rate), FEATURE_EXTRACTION_VERSION,
            {"feature_count": len(FEATURE_NAMES), "notebook_recipe": "xgboost-surrogate-v4",
             "missing_feature_names": missing, **(metadata or {})})

    def _prepare_waveform(self, waveform: NDArray[np.floating[Any]], sample_rate: int,
                          librosa: Any) -> NDArray[np.float32]:
        y = np.asarray(waveform, dtype=np.float32).squeeze()
        if y.ndim != 1 or not y.size:
            raise FeatureExtractionError("Waveform must contain a non-empty mono signal.")
        if not np.isfinite(y).all():
            raise FeatureExtractionError("Waveform contains non-finite values.")
        if sample_rate != self.config.sample_rate:
            try:
                import resampy  # noqa: F401 - librosa selects this backend below.
            except ImportError as error:
                raise FeatureExtractionError(
                    "Semantic v4 resampling requires the optional xai dependency: "
                    "resampy."
                ) from error
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=self.config.sample_rate,
                                 res_type="kaiser_fast")
        y = y - np.mean(y)
        maximum = int(self.config.max_audio_seconds * self.config.sample_rate)
        if y.size > maximum:
            start = (y.size - maximum) // 2
            y = y[start : start + maximum]
        if y.size < int(self.config.minimum_audio_seconds * self.config.sample_rate):
            raise FeatureExtractionError("Waveform is too short for the v4 feature recipe.")
        return np.asarray(y, dtype=np.float32)


def _load_feature_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import librosa
        import parselmouth
        from scipy import fft, signal
    except ImportError as error:
        raise FeatureExtractionError(
            "Semantic v4 extraction requires optional xai dependencies: librosa, "
            "praat-parselmouth, and scipy."
        ) from error
    return librosa, parselmouth, fft, signal


def _extract_feature_mapping(y: NDArray[np.float32], sr: int, librosa: Any,
                             parselmouth: Any, fft: Any, signal: Any) -> dict[str, float]:
    out = _praat_voice_features(y, sr, parselmouth)
    hop, n_fft, win = 160, 512, 400
    lfcc = _compute_lfcc(y, sr, librosa, fft.dct)
    out.update({f"lfcc_{index + 1}": _mean_or_nan(lfcc[index]) for index in range(20)})
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14, n_fft=n_fft, hop_length=hop,
                                win_length=win, n_mels=40, fmin=20.0,
                                fmax=min(7600.0, sr / 2 - 1))[1:14]
    width = 9 if mfcc.shape[1] >= 9 else max(3, mfcc.shape[1] // 2 * 2 + 1)
    if width > mfcc.shape[1]:
        width = mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1
    d1, d2 = (np.zeros_like(mfcc), np.zeros_like(mfcc)) if width < 3 else (
        librosa.feature.delta(mfcc, width=width, order=1, mode="nearest"),
        librosa.feature.delta(mfcc, width=width, order=2, mode="nearest"))
    for family, matrix in (("mfcc", mfcc), ("dmfcc", d1), ("ddmfcc", d2)):
        for index in range(13):
            _summarize_track(matrix[index], f"{family}_{index + 1}", out)
    _pitch_features(y, sr, hop, librosa, out)
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-12), ref=1.0)
    out.update({"rms_mean": float(np.mean(rms)), "rms_std": float(np.std(rms)),
                "rms_dynamic_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))})
    magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window="hann"))
    for name, track in {"spectral_centroid": librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0],
                        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0],
                        "spectral_rolloff": librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.85)[0],
                        "spectral_flatness": librosa.feature.spectral_flatness(S=magnitude)[0]}.items():
        _summarize_track(track, name, out)
    normalized = magnitude / (np.linalg.norm(magnitude, axis=0, keepdims=True) + 1e-12)
    flux = np.sqrt(np.sum(np.diff(normalized, axis=1) ** 2, axis=0)) if normalized.shape[1] > 1 else np.asarray([0.0])
    _summarize_track(flux, "spectral_flux", out)
    try:
        contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr, n_bands=5, fmin=200.0)
        for band in range(contrast.shape[0]):
            _summarize_track(contrast[band], f"spectral_contrast_band{band + 1}", out)
    except Exception:
        for band in range(1, 7):
            out.update({f"spectral_contrast_band{band}_mean": np.nan,
                        f"spectral_contrast_band{band}_std": np.nan})
    cpp = _compute_framewise_cpp(y, sr, librosa, fft, signal)
    out.update({"cpp_mean": _mean_or_nan(cpp), "cpp_std": _std_or_nan(cpp)})
    out.update(_pause_features_from_rms(rms, hop, sr, librosa))
    return out


def _compute_lfcc(y: NDArray[np.float32], sr: int, librosa: Any, dct: Any) -> NDArray[np.float64]:
    power = np.abs(librosa.stft(y, n_fft=512, hop_length=160, win_length=400, window="hann", center=True)) ** 2
    frequencies, edges = np.linspace(0.0, sr / 2.0, power.shape[0]), np.linspace(0.0, sr / 2.0, 42)
    bank = np.zeros((40, frequencies.size), dtype=np.float64)
    for index in range(40):
        left, center, right = edges[index : index + 3]
        left_mask, right_mask = ((frequencies >= left) & (frequencies <= center),
                                 (frequencies >= center) & (frequencies <= right))
        bank[index, left_mask] = (frequencies[left_mask] - left) / (center - left)
        bank[index, right_mask] = (right - frequencies[right_mask]) / (right - center)
    return dct(np.log(np.maximum(bank @ power, 1e-12)), type=2, axis=0, norm="ortho")[1:21]


def _praat_voice_features(y: NDArray[np.float32], sr: int, parselmouth: Any) -> dict[str, float]:
    from parselmouth.praat import call
    names = ("jitter", "shimmer", "hnr", "gci_count", "gci_mean_interval", "gci_std_interval",
             "gci_irregularity", "gci_jitter_gci", *(f"formant_f{i}_{stat}" for i in range(1, 4) for stat in ("mean", "std")))
    out = {name: np.nan for name in names}
    try:
        sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        points = call(sound, "To PointProcess (periodic, cc)", 75.0, 500.0)
        count = int(call(points, "Get number of points")); out["gci_count"] = float(count)
        if count >= 2:
            intervals = np.diff(np.asarray([call(points, "Get time from index", i) for i in range(1, count + 1)], dtype=float))
            intervals = intervals[(intervals > 1 / 500) & (intervals < 1 / 50)]
            if intervals.size:
                mean, std = float(np.mean(intervals)), float(np.std(intervals))
                out.update({"gci_mean_interval": mean, "gci_std_interval": std,
                            "gci_jitter_gci": std / (mean + 1e-12),
                            "gci_irregularity": float(np.mean(np.abs(np.diff(intervals))) / (mean + 1e-12)) if intervals.size > 1 else np.nan})
        out["jitter"] = _finite_or_nan(call(points, "Get jitter (local)", 0.0, 0.0, 1 / 500, 1 / 75, 1.3))
        out["shimmer"] = _finite_or_nan(call([sound, points], "Get shimmer (local)", 0.0, 0.0, 1 / 500, 1 / 75, 1.3, 1.6))
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
        out["hnr"] = _finite_or_nan(call(harmonicity, "Get mean", 0.0, 0.0))
        formant = call(sound, "To Formant (burg)", 0.0, 5.0, 5500.0, 0.025, 50.0)
        for index in range(1, 4):
            out[f"formant_f{index}_mean"] = _finite_or_nan(call(formant, "Get mean", index, 0.0, 0.0, "Hertz"))
            out[f"formant_f{index}_std"] = _finite_or_nan(call(formant, "Get standard deviation", index, 0.0, 0.0, "Hertz"))
    except Exception:
        pass
    return out


def _pitch_features(y: NDArray[np.float32], sr: int, hop: int, librosa: Any, out: dict[str, float]) -> None:
    try:
        f0, voiced_flag, _ = librosa.pyin(y, sr=sr, fmin=50.0, fmax=500.0, frame_length=2048, hop_length=hop)
        voiced = f0[np.isfinite(f0)]
        out.update({"f0_mean": _mean_or_nan(voiced), "f0_std": _std_or_nan(voiced),
                    "f0_median": float(np.median(voiced)) if voiced.size else np.nan,
                    "f0_range": float(np.ptp(voiced)) if voiced.size else np.nan,
                    "voiced_ratio": float(np.mean(voiced_flag.astype(float))) if voiced_flag is not None else np.nan})
    except Exception:
        out.update({name: np.nan for name in ("f0_mean", "f0_std", "f0_median", "f0_range", "voiced_ratio")})


def _compute_framewise_cpp(y: NDArray[np.float32], sr: int, librosa: Any, fft: Any, signal: Any) -> NDArray[np.float64]:
    frame_length, hop = int(round(sr * .04)), int(round(sr * .01))
    if y.size < frame_length: y = np.pad(y, (0, frame_length - y.size))
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop).T[:1000]
    window, q = signal.get_window("hann", frame_length, fftbins=True), np.arange(frame_length) / sr
    search, trend = ((q >= 1 / 500.0) & (q <= 1 / 60.0), (q >= .001) & (q <= min(.05, q[-1])))
    if search.sum() < 2 or trend.sum() < 3: return np.asarray([], dtype=float)
    values: list[float] = []
    for frame in frames:
        if np.sqrt(np.mean(frame ** 2) + 1e-12) < 1e-5: continue
        cepstrum = fft.irfft(10 * np.log10(np.maximum(np.abs(fft.rfft(frame * window)) ** 2, 1e-12)), n=frame_length)
        if not np.isfinite(cepstrum[trend]).all(): continue
        slope, intercept = np.polyfit(q[trend], cepstrum[trend], 1)
        peak = np.where(search)[0][np.argmax(cepstrum[search])]
        values.append(float(cepstrum[peak] - (slope * q[peak] + intercept)))
    return np.asarray(values, dtype=float)


def _pause_features_from_rms(rms: NDArray[np.floating[Any]], hop: int, sr: int, librosa: Any) -> dict[str, float]:
    if not rms.size or np.max(rms) <= 1e-12:
        duration = rms.size * hop / sr
        return {"pause_ratio": 1., "pause_count": 1., "pause_mean_duration": duration, "pause_max_duration": duration}
    silent = librosa.amplitude_to_db(np.maximum(rms, 1e-12), ref=np.max) <= -40.
    minimum, durations, start = max(1, int(round(.15 * sr / hop))), [], None
    for index, silent_frame in enumerate(silent):
        if silent_frame and start is None: start = index
        if (not silent_frame or index == len(silent) - 1) and start is not None:
            end = index if not silent_frame else index + 1
            if end - start >= minimum: durations.append((end - start) * hop / sr)
            start = None
    return {"pause_ratio": float(np.mean(silent)), "pause_count": float(len(durations)),
            "pause_mean_duration": float(np.mean(durations)) if durations else 0.,
            "pause_max_duration": float(np.max(durations)) if durations else 0.}


def _summarize_track(values: NDArray[np.floating[Any]], prefix: str, out: dict[str, float]) -> None:
    finite = np.asarray(values, dtype=float); finite = finite[np.isfinite(finite)]
    out[f"{prefix}_mean"], out[f"{prefix}_std"] = _mean_or_nan(finite), _std_or_nan(finite)


def _mean_or_nan(values: NDArray[np.floating[Any]]) -> float: return float(np.mean(values)) if np.asarray(values).size else np.nan
def _std_or_nan(values: NDArray[np.floating[Any]]) -> float: return float(np.std(values)) if np.asarray(values).size else np.nan
def _finite_or_nan(value: Any) -> float:
    try: parsed = float(value)
    except (TypeError, ValueError): return np.nan
    return parsed if np.isfinite(parsed) else np.nan
