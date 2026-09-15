from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.voice_xai.evaluation.offline_metrics import (
    OfflineMetricArtifactError,
    SURROGATE_FIDELITY_ARTIFACT_PATH,
    load_surrogate_fidelity_artifact,
    surrogate_fidelity_for_model,
)


def test_versioned_surrogate_fidelity_artifact_matches_validation_export() -> None:
    artifact = load_surrogate_fidelity_artifact(SURROGATE_FIDELITY_ARTIFACT_PATH)

    assert artifact.artifact_version == "voice-xai-surrogate-fidelity-r2-v1"
    assert artifact.semantic_model_version == "xgboost-surrogate-v4-2026-08-19"
    assert artifact.dataset_version == "ASVspoof2019_LA"
    assert artifact.sample_count == 449
    assert artifact.value == pytest.approx(0.6342774342328266)
    assert artifact.source.artifact_name == "fidelity_by_dataset.csv"
    assert artifact.source.sha256 == (
        "2277b752e84057fe4ecd9aaf6c733cd1c965c820a824b6fec895a3e2982b50ff"
    )
    assert artifact.source.metric_column == "r2"


def test_fidelity_is_available_only_for_the_validated_semantic_model() -> None:
    available = surrogate_fidelity_for_model("xgboost-surrogate-v4-2026-08-19")
    mismatched = surrogate_fidelity_for_model("another-semantic-model")

    assert available.status.value == "available"
    assert available.value == pytest.approx(0.6342774342328266)
    assert available.scope == "offline_validation"
    assert available.dataset_version == "ASVspoof2019_LA"
    assert mismatched.status.value == "not_applicable"
    assert mismatched.value is None


def test_invalid_offline_metric_artifact_fails_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "surrogate_fidelity.json"
    invalid.write_text(
        json.dumps(
            {
                "artifact_version": "voice-xai-surrogate-fidelity-r2-v1",
                "metric_name": "surrogate_fidelity_r2",
                "value": 0.5,
                "scope": "offline_validation",
                "dataset_version": "ASVspoof2019_LA",
                "sample_count": 0,
                "semantic_model_version": "xgboost-surrogate-v4-2026-08-19",
                "source": {
                    "artifact_name": "fidelity_by_dataset.csv",
                    "sha256": (
                        "2277b752e84057fe4ecd9aaf6c733cd1c965c820a824b6fec895a3e2982b50ff"
                    ),
                    "dataset_column": "dataset_source",
                    "dataset_value": "ASVspoof2019_LA",
                    "metric_column": "r2",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(OfflineMetricArtifactError):
        load_surrogate_fidelity_artifact(invalid)
