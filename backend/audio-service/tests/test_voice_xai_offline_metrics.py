from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from app.voice_xai.evaluation.offline_metrics import (
    OfflineMetricArtifactError,
    load_surrogate_fidelity_artifact,
    surrogate_fidelity_for_model,
)


def test_versioned_fidelity_artifact_loads_validated_metadata(
    fidelity_artifact_path: Path,
) -> None:
    artifact = load_surrogate_fidelity_artifact(fidelity_artifact_path)

    assert artifact.artifact_version == "voice-xai-surrogate-fidelity-r2-v1"
    assert artifact.semantic_model_version == "fixture-surrogate-v1"
    assert artifact.dataset_version == "fixture_dataset"
    assert artifact.sample_count == 4
    assert artifact.value == pytest.approx(0.75)
    assert artifact.source.artifact_name == "fixture_fidelity.csv"
    source = fidelity_artifact_path.parent / artifact.source.artifact_name
    assert artifact.source.sha256 == sha256(source.read_bytes()).hexdigest()
    assert artifact.source.metric_column == "r2"


def test_fidelity_is_available_only_for_the_validated_semantic_model(
    fidelity_artifact_path: Path,
) -> None:
    available = surrogate_fidelity_for_model("fixture-surrogate-v1")
    mismatched = surrogate_fidelity_for_model("another-semantic-model")

    assert available.status.value == "available"
    assert available.value == pytest.approx(0.75)
    assert available.scope == "offline_validation"
    assert available.dataset_version == "fixture_dataset"
    assert mismatched.status.value == "not_applicable"
    assert mismatched.value is None


def test_invalid_offline_metric_artifact_fails_closed(
    tmp_path: Path, fidelity_artifact_path: Path
) -> None:
    invalid = tmp_path / "surrogate_fidelity.json"
    payload = json.loads(fidelity_artifact_path.read_text(encoding="utf-8"))
    payload["sample_count"] = 0
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OfflineMetricArtifactError):
        load_surrogate_fidelity_artifact(invalid)


def test_missing_fidelity_artifact_reports_not_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(
        "app.voice_xai.evaluation.offline_metrics.load_surrogate_fidelity_artifact",
        lambda: load_surrogate_fidelity_artifact(missing),
    )

    with pytest.raises(OfflineMetricArtifactError):
        load_surrogate_fidelity_artifact(missing)
    metric = surrogate_fidelity_for_model("fixture-surrogate-v1")
    assert metric.status.value == "not_computed"
    assert metric.value is None
    assert metric.scope == "offline_validation"
    assert "unavailable" in metric.reason
