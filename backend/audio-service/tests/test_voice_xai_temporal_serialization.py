import numpy as np
import pytest

from app.voice_xai.temporal.contracts import (
    TemporalAttentionError,
    TemporalAttentionWindowInput,
)
from app.voice_xai.temporal.serialization import (
    TEMPORAL_EVIDENCE_ARTIFACT_CONTENT_TYPE,
    deserialize_temporal_evidence,
    serialize_temporal_evidence,
)


def test_compact_temporal_evidence_serializes_without_raw_attention_tensors() -> None:
    evidence = (
        TemporalAttentionWindowInput(
            start_seconds=0.0,
            end_seconds=0.06,
            token_times_seconds=np.array([0.01, 0.03, 0.05], dtype=np.float32),
            attention_density=np.array([0.9, 1.3, 0.8], dtype=np.float32),
            spoof_probability=0.75,
        ),
    )

    content = serialize_temporal_evidence(evidence)
    recovered = deserialize_temporal_evidence(content)

    assert TEMPORAL_EVIDENCE_ARTIFACT_CONTENT_TYPE.startswith("application/")
    assert b"attention_density" in content
    assert b"raw_attention" not in content
    assert len(content) < 2_000
    np.testing.assert_allclose(
        recovered[0].token_times_seconds, evidence[0].token_times_seconds
    )
    np.testing.assert_allclose(
        recovered[0].attention_density, evidence[0].attention_density
    )


def test_temporal_evidence_deserialization_rejects_noncontract_payload() -> None:
    with pytest.raises(TemporalAttentionError, match="Stored original-pass"):
        deserialize_temporal_evidence(b'{"version":"bad","windows":[]}')
