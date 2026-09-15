import numpy as np
import pytest

from app.voice_xai.capture.torch_hooks import CaptureTarget, PyTorchExtractionInterface


def test_hook_captures_request_scoped_layer_output() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        module[0].weight.fill_(1.0)
    interface = PyTorchExtractionInterface()
    interface.register_branch(
        module,
        [
            CaptureTarget(
                branch_name="aasist",
                layer_index=0,
                module_path="0",
                category="embedding",
                normalization="none",
            )
        ],
    )
    try:
        with interface.capture("request-123") as session:
            module(torch.tensor([[1.0, 2.0, 3.0]]))
        bundle = session.build_bundle()

        capture = bundle.branches["aasist"][0]
        np.testing.assert_allclose(capture.values, [[6.0, 6.0]])
        assert capture.status == "captured"
    finally:
        interface.close()


def test_hook_marks_unselected_tensor_output_as_skipped() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Sequential(torch.nn.Identity())
    interface = PyTorchExtractionInterface()
    interface.register_branch(
        module,
        [
            CaptureTarget(
                branch_name="ssl_sequence",
                layer_index=0,
                module_path="0",
                category="attention",
                output_path=(1,),
            )
        ],
    )
    try:
        with interface.capture("request-123") as session:
            module(torch.ones((1, 3)))
        capture = session.build_bundle().branches["ssl_sequence"][0]

        assert capture.status == "skipped"
        assert capture.reason == "selected_output_is_not_a_tensor"
    finally:
        interface.close()


def test_hook_marks_oversized_tensor_as_skipped() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Sequential(torch.nn.Identity())
    interface = PyTorchExtractionInterface()
    interface.register_branch(
        module,
        [
            CaptureTarget(
                branch_name="lfcc_cnn_tcn",
                layer_index=0,
                module_path="0",
                category="feature_map",
                max_elements=2,
            )
        ],
    )
    try:
        with interface.capture("request-123") as session:
            module(torch.ones((1, 3)))
        capture = session.build_bundle().branches["lfcc_cnn_tcn"][0]

        assert capture.status == "skipped"
        assert capture.reason == "capture_exceeds_max_elements:2"
    finally:
        interface.close()


def test_hook_enforces_the_request_total_capture_budget() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
    interface = PyTorchExtractionInterface(max_total_elements=3)
    interface.register_branch(
        module,
        [
            CaptureTarget(
                branch_name="aasist",
                layer_index=0,
                module_path="0",
                category="feature_map",
            ),
            CaptureTarget(
                branch_name="aasist",
                layer_index=1,
                module_path="1",
                category="feature_map",
            ),
        ],
    )
    try:
        with interface.capture("request-123") as session:
            module(torch.ones((1, 2)))
        captures = session.build_bundle().branches["aasist"]

        assert captures[0].status == "captured"
        assert captures[1].status == "skipped"
        assert captures[1].reason == "capture_exceeds_total_elements:3"
    finally:
        interface.close()
