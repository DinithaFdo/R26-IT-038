from io import BytesIO
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.config.settings import Settings
from app.voice_xai.artifacts.service import LocalExplanationArtifactStore
from app.voice_xai.capture.serialization import (
    EXTRACTION_ARTIFACT_FORMAT,
    serialize_extraction_bundle,
)
from app.voice_xai.capture.contracts import (
    CapturedRepresentation,
    ExtractionBundle,
)
from app.voice_xai.orchestrator import VoiceXaiOrchestrator


def test_extraction_bundle_is_encoded_as_binary_npz_with_a_manifest() -> None:
    bundle = ExtractionBundle(
        request_id="request-123",
        branches={
            "aasist": {
                2: CapturedRepresentation(
                    branch_name="aasist",
                    layer_index=2,
                    category="attention",
                    values=np.asarray([[[0.25], [0.75]]], dtype=np.float32),
                    normalization="minmax",
                    status="captured",
                )
            }
        },
        metadata={"captured_elements": 2},
    )

    encoded = serialize_extraction_bundle(bundle)

    with np.load(BytesIO(encoded), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"].item()))
        np.testing.assert_allclose(archive["capture_0"], [[[0.25], [0.75]]])

    assert manifest["format"] == EXTRACTION_ARTIFACT_FORMAT
    assert manifest["request_id"] == "request-123"
    assert manifest["branches"]["aasist"]["2"]["array_key"] == "capture_0"


@pytest.mark.anyio
async def test_orchestrator_writes_capture_as_a_private_binary_artifact(tmp_path) -> None:
    bundle = ExtractionBundle(
        request_id="request-123",
        branches={
            "lfcc_cnn_tcn": {
                0: CapturedRepresentation(
                    branch_name="lfcc_cnn_tcn",
                    layer_index=0,
                    category="feature_map",
                    values=np.ones((1, 2, 3), dtype=np.float32),
                    normalization="minmax",
                    status="captured",
                )
            }
        },
    )
    repository = _ArtifactRepository()
    store = LocalExplanationArtifactStore(tmp_path / "private")
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=object(),
        queue=object(),
        app_settings=Settings(_env_file=None),
        artifact_store=store,
        temporal_service=object(),
    )

    await orchestrator._store_extraction_artifact(
        repository=repository,
        explanation_id="explanation-123",
        bundle=SimpleNamespace(extraction=bundle, request_id=bundle.request_id),
    )

    assert len(repository.artifacts) == 1
    reference = repository.artifacts[0]
    assert reference.kind == "intermediate_representations"
    assert reference.content_type == "application/x-npz"
    stored = store.read(reference)
    assert stored is not None
    with np.load(BytesIO(stored.content), allow_pickle=False) as archive:
        assert "capture_0" in archive.files


class _ArtifactRepository:
    def __init__(self) -> None:
        self.artifacts = []

    async def append_artifact(self, explanation_id, artifact) -> None:
        assert explanation_id == "explanation-123"
        self.artifacts.append(artifact)
