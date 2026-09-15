from app.voice_xai.capture import torch_hooks
from app.voice_xai.capture.session import PyTorchExtractionInterface


def test_public_interface_exposes_only_capture_types() -> None:
    assert torch_hooks.PyTorchExtractionInterface is PyTorchExtractionInterface
    assert "CaptureTarget" in torch_hooks.__all__
    assert "AsynchronousExplanationQueue" not in torch_hooks.__all__
