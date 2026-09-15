"""Human-readable labels for the frozen v4 feature schema."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.voice_xai.semantic.features import FEATURE_NAMES


@dataclass(frozen=True)
class FeatureLabel:
    display_name: str
    unit: str | None


_DIRECT = {
    "jitter": FeatureLabel("Pitch jitter", "ratio"),
    "shimmer": FeatureLabel("Amplitude shimmer", "ratio"),
    "hnr": FeatureLabel("Harmonics-to-noise ratio", "dB"),
    "gci_count": FeatureLabel("Estimated glottal-closure count", "count"),
    "gci_mean_interval": FeatureLabel("Mean glottal-closure interval", "s"),
    "gci_std_interval": FeatureLabel("Glottal-closure interval standard deviation", "s"),
    "gci_irregularity": FeatureLabel("Glottal-closure irregularity", "ratio"),
    "gci_jitter_gci": FeatureLabel("Glottal-closure jitter", "ratio"),
    "f0_mean": FeatureLabel("Mean fundamental frequency", "Hz"),
    "f0_std": FeatureLabel("Fundamental-frequency standard deviation", "Hz"),
    "f0_median": FeatureLabel("Median fundamental frequency", "Hz"),
    "f0_range": FeatureLabel("Fundamental-frequency range", "Hz"),
    "voiced_ratio": FeatureLabel("Voiced-frame ratio", "ratio"),
    "rms_mean": FeatureLabel("Mean RMS energy", None),
    "rms_std": FeatureLabel("RMS energy standard deviation", None),
    "rms_dynamic_range_db": FeatureLabel("RMS dynamic range", "dB"),
    "cpp_mean": FeatureLabel("Mean cepstral peak prominence", None),
    "cpp_std": FeatureLabel("Cepstral peak-prominence standard deviation", None),
    "pause_ratio": FeatureLabel("Silent-frame ratio", "ratio"),
    "pause_count": FeatureLabel("Pause count", "count"),
    "pause_mean_duration": FeatureLabel("Mean pause duration", "s"),
    "pause_max_duration": FeatureLabel("Maximum pause duration", "s"),
}


def label_for_feature(feature_name: str) -> FeatureLabel:
    if feature_name in _DIRECT:
        return _DIRECT[feature_name]
    if match := re.fullmatch(r"formant_f([123])_(mean|std)", feature_name):
        return FeatureLabel(f"Formant F{match.group(1)} {'mean' if match.group(2) == 'mean' else 'standard deviation'}", "Hz")
    if match := re.fullmatch(r"lfcc_(\d+)", feature_name):
        return FeatureLabel(f"Linear-frequency cepstral coefficient {match.group(1)}", None)
    if match := re.fullmatch(r"(d{0,2}mfcc)_(\d+)_(mean|std)", feature_name):
        prefix = {"mfcc": "MFCC", "dmfcc": "Delta MFCC", "ddmfcc": "Delta-delta MFCC"}[match.group(1)]
        return FeatureLabel(f"{prefix} {match.group(2)} {'mean' if match.group(3) == 'mean' else 'standard deviation'}", None)
    if match := re.fullmatch(r"spectral_(centroid|bandwidth|rolloff|flatness|flux)_(mean|std)", feature_name):
        return FeatureLabel(f"Spectral {match.group(1)} {'mean' if match.group(2) == 'mean' else 'standard deviation'}", "Hz" if match.group(1) in {"centroid", "bandwidth", "rolloff"} else None)
    if match := re.fullmatch(r"spectral_contrast_band([1-6])_(mean|std)", feature_name):
        return FeatureLabel(f"Spectral contrast band {match.group(1)} {'mean' if match.group(2) == 'mean' else 'standard deviation'}", "dB")
    raise ValueError(f"Unknown semantic feature: {feature_name}")


def validate_feature_labels() -> None:
    for feature_name in FEATURE_NAMES:
        label_for_feature(feature_name)


validate_feature_labels()
