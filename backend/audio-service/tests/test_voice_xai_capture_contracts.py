import numpy as np
import pytest

from app.voice_xai.capture.contracts import (
    CaptureTarget,
    CapturedRepresentation,
    ExtractionBundle,
)


def test_capture_target_preserves_tensor_selection_contract() -> None:
    target = CaptureTarget(
        branch_name="ssl_sequence",
        layer_index=3,
        module_path="encoder.layers.3.attention",
        category="attention",
        output_path=(1,),
        normalization="zscore",
        max_elements=500,
    )

    assert target.output_path == (1,)
    assert target.normalization == "zscore"
    assert target.max_elements == 500


@pytest.mark.parametrize(
    "target_kwargs",
    [
        {"branch_name": "", "module_path": "encoder"},
        {"branch_name": "aasist", "module_path": "", "layer_index": 0},
        {"branch_name": "aasist", "module_path": "encoder", "layer_index": -1},
        {"branch_name": "aasist", "module_path": "encoder", "max_elements": 0},
    ],
)
def test_capture_target_rejects_invalid_configuration(target_kwargs) -> None:
    values = {
        "branch_name": "aasist",
        "layer_index": 0,
        "module_path": "encoder",
        "category": "embedding",
        **target_kwargs,
    }

    with pytest.raises(ValueError):
        CaptureTarget(**values)


def test_extraction_bundle_serializes_branch_and_layer_keys() -> None:
    capture = CapturedRepresentation(
        branch_name="aasist",
        layer_index=2,
        category="embedding",
        values=np.asarray([[0.25, 0.75]], dtype=np.float32),
        normalization="minmax",
        status="captured",
    )
    bundle = ExtractionBundle(
        request_id="request-123",
        branches={"aasist": {2: capture}},
    )

    payload = bundle.as_json_dict()

    assert payload["request_id"] == "request-123"
    assert payload["branches"]["aasist"]["2"]["shape"] == [1, 2]
    assert payload["branches"]["aasist"]["2"]["values"] == [[0.25, 0.75]]
