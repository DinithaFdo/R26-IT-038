import numpy as np
import pytest
import json

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


def test_source_offsets_survive_private_evidence_roundtrip() -> None:
    window = TemporalAttentionWindowInput(
        start_seconds=7.0, end_seconds=13.0,
        token_times_seconds=np.array([0.01, 2.01, 5.99], dtype=np.float32),
        attention_density=np.array([0.8, 1.3, 0.7], dtype=np.float32),
    )
    content = serialize_temporal_evidence((window,))
    recovered = deserialize_temporal_evidence(content)[0]
    assert recovered.start_seconds == 7.0 and recovered.end_seconds == 13.0
    np.testing.assert_array_equal(recovered.token_times_seconds, window.token_times_seconds)
    old = json.loads(content)
    old["version"] = "xlsr-temporal-evidence-v2"
    with pytest.raises(TemporalAttentionError, match="Stored original-pass"):
        deserialize_temporal_evidence(json.dumps(old).encode())
