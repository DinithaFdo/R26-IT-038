"""Map supported acoustic feature identifiers to readable display labels.

Validate each identifier against the feature schema before formatting its label.
"""
from app.voice_xai.semantic.schema import FEATURE_NAMES


def label_for_feature(name: str) -> str:
    if name not in FEATURE_NAMES:
        raise ValueError(f"Unknown semantic feature: {name}")
    return name.replace("_", " ").capitalize()
