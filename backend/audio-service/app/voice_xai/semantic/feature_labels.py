"""Map supported acoustic feature identifiers to readable display labels.

Validate each identifier against the feature schema before formatting its label.
"""
from dataclasses import dataclass
from app.voice_xai.semantic.schema import FEATURE_NAMES
@dataclass(frozen=True)
class FeatureLabel:
    display_name: str
    unit: str | None = None

def label_for_feature(name: str) -> FeatureLabel:
    if name not in FEATURE_NAMES:
        raise ValueError(f"Unknown semantic feature: {name}")
    return FeatureLabel(name.replace("_", " ").capitalize())
