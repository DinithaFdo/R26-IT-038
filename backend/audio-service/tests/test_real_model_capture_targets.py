"""Voice XAI capture targets against the real loaded checkpoints.

The point of these tests is that the module paths in
``app/models/capture_targets.py`` are *verified*, not copied from a diagram.
``PyTorchExtractionInterface.register_branch`` raises on an unresolvable path,
so a stale path would break branch registration the moment XAI capture is
switched on. These tests catch that now instead of then.
"""

from __future__ import annotations

import pytest

from tests.real_model_helpers import (
    AASIST_LIGHT_V1_CHECKPOINT,
    CNN_CHECKPOINT,
    requires_aasist_light_v1_checkpoint,
    requires_cnn_checkpoint,
    requires_torch,
)

pytestmark = requires_torch


def _loaded(build, checkpoint):
    from app.models.torch_support import load_state_dict_strict

    module = build()
    load_state_dict_strict(module, checkpoint)
    module.eval()
    return module


@requires_cnn_checkpoint
def test_every_cnn_capture_path_resolves_on_the_loaded_model() -> None:
    from app.models.architectures.cnn_v2 import build_cnn_v2_net
    from app.models.capture_targets import cnn_capture_targets

    module = _loaded(build_cnn_v2_net, CNN_CHECKPOINT)
    available = dict(module.named_modules())

    for target in cnn_capture_targets():
        assert target.module_path in available, target.module_path


@requires_aasist_light_v1_checkpoint
def test_every_aasist_capture_path_resolves_on_the_loaded_model() -> None:
    """Tests the superseded V1 architecture against V1's own checkpoint.

    `app/models/capture_targets.py::aasist_capture_targets` names V1 module
    paths (``temporal``, ``pooling.attention``); AASIST-Light V2 -- the
    deployed architecture -- uses different module names entirely
    (``frontend``, ``gru``, ``attention``, ``classifier``, per
    ``app/models/real/inference.py::build_aasist_loader``'s inline capture
    targets) and is not exercised by this legacy path.
    """

    from app.models.architectures.aasist_light import build_aasist_light_net
    from app.models.capture_targets import aasist_capture_targets

    module = _loaded(build_aasist_light_net, AASIST_LIGHT_V1_CHECKPOINT)
    available = dict(module.named_modules())

    for target in aasist_capture_targets():
        assert target.module_path in available, target.module_path


@requires_cnn_checkpoint
def test_cnn_capture_produces_tensors_within_the_element_budget() -> None:
    import torch

    from app.models.architectures.cnn_v2 import build_cnn_v2_net
    from app.models.capture_targets import cnn_capture_targets
    from app.voice_xai.capture.session import PyTorchExtractionInterface

    module = _loaded(build_cnn_v2_net, CNN_CHECKPOINT)
    interface = PyTorchExtractionInterface()
    interface.register_branch(module, cnn_capture_targets())

    try:
        with interface.capture("request-1") as session:
            with torch.inference_mode():
                module(torch.zeros(1, 1, 120, 401))
        bundle = session.build_bundle()
    finally:
        interface.close()

    captures = bundle.branches["lfcc_cnn_tcn"]
    assert len(captures) == 3
    for capture in captures.values():
        assert capture.status == "captured", capture.reason


@requires_aasist_light_v1_checkpoint
def test_aasist_capture_selects_the_gru_sequence_not_the_hidden_state() -> None:
    """``temporal`` returns ``(output, h_n)``; ``output_path=(0,)`` picks output.

    Superseded V1 architecture test; see the module docstring above.
    """

    import torch

    from app.models.architectures.aasist_light import build_aasist_light_net
    from app.models.capture_targets import aasist_capture_targets
    from app.voice_xai.capture.session import PyTorchExtractionInterface

    module = _loaded(build_aasist_light_net, AASIST_LIGHT_V1_CHECKPOINT)
    interface = PyTorchExtractionInterface()
    interface.register_branch(module, aasist_capture_targets())

    try:
        with interface.capture("request-2") as session:
            with torch.inference_mode():
                module(torch.zeros(1, 1, 64000))
        bundle = session.build_bundle()
    finally:
        interface.close()

    captures = bundle.branches["aasist"]
    for capture in captures.values():
        assert capture.status == "captured", capture.reason

    # layer 1 is the BiGRU sequence: (batch, time, 2 * hidden)
    assert captures[1].values.shape == (1, 1000, 256)
    # layer 2 is the per-frame attention score used by a future rollout consumer
    assert captures[2].values.shape == (1, 1000, 1)
    assert captures[2].category == "attention"


def test_branches_without_a_trained_model_have_no_capture_targets() -> None:
    from app.models.capture_targets import register_real_branch_capture

    class ExplodingInterface:
        def register_branch(self, *_args, **_kwargs):
            raise AssertionError("must not register an unintegrated branch")

    for branch in ("ssl_sequence", "glottal"):
        assert register_real_branch_capture(ExplodingInterface(), branch, None) is False


@requires_cnn_checkpoint
def test_capture_is_not_enabled_during_normal_prediction() -> None:
    """XAI capture stays off until a consumer exists.

    Real model inference must not be blocked by, or pay for, an explanation
    pipeline that cannot yet read these tensors.
    """

    from app.models.factory import ModelFactory
    from tests.real_model_helpers import processed_audio, real_model_settings

    model = ModelFactory(real_model_settings()).create("lfcc_cnn_tcn")
    model.predict_safe(processed_audio())

    module = model._runtime_model.module
    hooked = [
        name
        for name, submodule in module.named_modules()
        if getattr(submodule, "_forward_hooks", None)
    ]
    assert hooked == []
