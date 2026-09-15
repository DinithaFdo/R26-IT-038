"""Architecture reconstruction and checkpoint compatibility.

These are the tests that would catch the reconstruction drifting away from the
trained checkpoints. Because no training notebook shipped with the models, the
checkpoints themselves are the only specification, so `strict=True` loading is
the contract under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.real_model_helpers import (
    AASIST_CHECKPOINT,
    AASIST_LIGHT_V1_CHECKPOINT,
    CNN_CHECKPOINT,
    CNN_LEGACY_CHECKPOINT,
    requires_aasist_checkpoint,
    requires_aasist_light_v1_checkpoint,
    requires_cnn_checkpoint,
    requires_cnn_legacy_checkpoint,
    requires_torch,
)

pytestmark = requires_torch


@requires_cnn_legacy_checkpoint
def test_cnn_legacy_checkpoint_loads_strictly_with_no_missing_or_unexpected_keys() -> None:
    """Tests the superseded log-mel/3-block architecture against its own
    checkpoint. Production now deploys CNN-V2; see
    ``test_cnn_v2_checkpoint_integration`` below for its strict-load contract."""

    from app.models.architectures.cnn import build_cnn_acoustic_net
    from app.models.torch_support import load_state_dict_strict

    module = build_cnn_acoustic_net()
    summary = load_state_dict_strict(module, CNN_LEGACY_CHECKPOINT)

    assert summary["state_dict_keys"] == 23
    assert summary["parameter_count"] == 23_650


@requires_cnn_checkpoint
def test_cnn_v2_checkpoint_integration() -> None:
    """CNN-V2 (self-describing: config/feature_pipeline/label_map are
    embedded and validated at load time -- see
    ``load_cnn_v2_checkpoint_strict``, ``app/models/real/inference.py``).
    Combines strict load, declared feature-shape acceptance, and eval-mode
    determinism in one test, mirroring the AASIST-Light V2 integration test's
    structure rather than the older per-assertion CNN test split above."""

    import torch

    from app.models.architectures.cnn_v2 import (
        EXPECTED_PARAMETER_COUNT,
        EXPECTED_STATE_DICT_TENSORS,
        build_cnn_v2_net,
    )
    from app.models.torch_support import load_state_dict_strict

    module = build_cnn_v2_net()
    summary = load_state_dict_strict(module, CNN_CHECKPOINT)
    assert summary["state_dict_keys"] == EXPECTED_STATE_DICT_TENSORS
    assert summary["parameter_count"] == EXPECTED_PARAMETER_COUNT

    module.eval()
    with torch.inference_mode():
        logits = module(torch.zeros(1, 1, 120, 401))
    assert tuple(logits.shape) == (1, 2)
    assert bool(torch.isfinite(logits).all())

    features = torch.randn(1, 1, 120, 401, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        first = module(features)
        second = module(features)
    assert torch.equal(first, second)


@requires_aasist_light_v1_checkpoint
def test_aasist_checkpoint_loads_strictly_with_no_missing_or_unexpected_keys() -> None:
    """Tests the superseded V1 architecture against V1's own checkpoint.

    Production now deploys AASIST-Light V2; see
    ``test_aasist_light_v2_checkpoint_integration.py`` for its strict-load
    contract test.
    """

    from app.models.architectures.aasist_light import build_aasist_light_net
    from app.models.torch_support import load_state_dict_strict

    module = build_aasist_light_net()
    summary = load_state_dict_strict(module, AASIST_LIGHT_V1_CHECKPOINT)

    assert summary["state_dict_keys"] == 37
    assert summary["parameter_count"] == 300_035


@requires_cnn_checkpoint
def test_loading_the_aasist_checkpoint_into_the_cnn_fails_loudly() -> None:
    """A mismatched checkpoint must raise, never be masked by strict=False.

    strict=False would leave every layer randomly initialised and still return
    confident-looking probabilities.
    """

    from app.models.architectures.cnn import build_cnn_acoustic_net
    from app.models.torch_support import (
        CheckpointCompatibilityError,
        load_state_dict_strict,
    )

    if not AASIST_CHECKPOINT.is_file():
        pytest.skip("AASIST checkpoint not present.")

    with pytest.raises(CheckpointCompatibilityError):
        load_state_dict_strict(build_cnn_acoustic_net(), AASIST_CHECKPOINT)


def test_a_corrupt_checkpoint_raises_a_domain_error(tmp_path: Path) -> None:
    from app.models.architectures.cnn import build_cnn_acoustic_net
    from app.models.torch_support import (
        CheckpointCompatibilityError,
        load_state_dict_strict,
    )

    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"this is not a torch archive")

    with pytest.raises(CheckpointCompatibilityError):
        load_state_dict_strict(build_cnn_acoustic_net(), corrupt)


@requires_cnn_legacy_checkpoint
def test_cnn_legacy_accepts_the_declared_feature_shape_and_emits_two_logits() -> None:
    import torch

    from app.models.architectures.cnn import build_cnn_acoustic_net
    from app.models.torch_support import load_state_dict_strict

    module = build_cnn_acoustic_net()
    load_state_dict_strict(module, CNN_LEGACY_CHECKPOINT)
    module.eval()

    with torch.inference_mode():
        logits = module(torch.zeros(1, 1, 40, 400))

    assert tuple(logits.shape) == (1, 2)
    assert bool(torch.isfinite(logits).all())


@requires_cnn_legacy_checkpoint
def test_cnn_legacy_is_input_size_agnostic_which_is_why_features_need_verifying() -> None:
    """Documents the risk the verification flags exist to manage.

    The global average pool means a wrong coefficient count or frame count is
    accepted silently. Nothing in the architecture can detect a wrong feature
    pipeline, so it has to be attested by configuration.
    """

    import torch

    from app.models.architectures.cnn import build_cnn_acoustic_net
    from app.models.torch_support import load_state_dict_strict

    module = build_cnn_acoustic_net()
    load_state_dict_strict(module, CNN_LEGACY_CHECKPOINT)
    module.eval()

    with torch.inference_mode():
        for shape in [(1, 1, 20, 100), (1, 1, 40, 400), (1, 1, 80, 800)]:
            assert tuple(module(torch.zeros(*shape)).shape) == (1, 2)


@requires_aasist_light_v1_checkpoint
def test_aasist_accepts_waveform_input_and_emits_two_logits() -> None:
    import torch

    from app.models.architectures.aasist_light import build_aasist_light_net
    from app.models.torch_support import load_state_dict_strict

    module = build_aasist_light_net()
    load_state_dict_strict(module, AASIST_LIGHT_V1_CHECKPOINT)
    module.eval()

    with torch.inference_mode():
        logits = module(torch.zeros(1, 1, 64000))

    assert tuple(logits.shape) == (1, 2)
    assert bool(torch.isfinite(logits).all())


@requires_cnn_legacy_checkpoint
def test_cnn_legacy_inference_is_deterministic_in_eval_mode() -> None:
    import torch

    from app.models.architectures.cnn import build_cnn_acoustic_net
    from app.models.torch_support import load_state_dict_strict

    module = build_cnn_acoustic_net()
    load_state_dict_strict(module, CNN_LEGACY_CHECKPOINT)
    module.eval()

    features = torch.randn(1, 1, 40, 400, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        first = module(features)
        second = module(features)

    assert torch.equal(first, second)


@requires_aasist_light_v1_checkpoint
def test_aasist_inference_is_deterministic_in_eval_mode() -> None:
    import torch

    from app.models.architectures.aasist_light import build_aasist_light_net
    from app.models.torch_support import load_state_dict_strict

    module = build_aasist_light_net()
    load_state_dict_strict(module, AASIST_LIGHT_V1_CHECKPOINT)
    module.eval()

    waveform = torch.randn(1, 1, 64000, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        first = module(waveform)
        second = module(waveform)

    assert torch.equal(first, second)


@requires_aasist_light_v1_checkpoint
def test_aasist_light_is_not_a_graph_attention_network() -> None:
    """Guards against someone 'fixing' this by dropping in upstream AASIST.

    The trained checkpoint has no SincConv front end, no graph construction and
    no graph attention layers. Substituting the reference implementation would
    report a different model under the same branch name.
    """

    import torch

    state_dict = torch.load(
        AASIST_LIGHT_V1_CHECKPOINT, map_location="cpu", weights_only=True
    )
    keys = " ".join(state_dict.keys()).lower()

    for graph_marker in ("sinc", "graph", "gat", "hs_gal", "master"):
        assert graph_marker not in keys

    assert "temporal.weight_ih_l0" in state_dict
    # 3 gates x 128 hidden == GRU. An LSTM would be 4 x 128 == 512.
    assert tuple(state_dict["temporal.weight_ih_l0"].shape) == (384, 128)
    # Attention-weighted mean pooling: 256 in, not 512 (which would be
    # mean+std statistics pooling).
    assert tuple(state_dict["pooling.attention.0.weight"].shape) == (128, 256)


def test_aasist_light_v2_instantiates_with_finalized_parameter_count() -> None:
    from app.models.architectures.aasist_light_v2 import build_aasist_light_v2_net

    module = build_aasist_light_v2_net()

    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

    assert parameter_count == 641_795
    assert trainable_count == 641_795


def test_aasist_light_v2_accepts_batched_waveforms_and_returns_logits() -> None:
    import torch

    from app.models.architectures.aasist_light_v2 import build_aasist_light_v2_net

    module = build_aasist_light_v2_net()
    module.eval()

    with torch.inference_mode():
        logits = module(torch.zeros(2, 64_600))

    assert tuple(logits.shape) == (2, 2)
    assert bool(torch.isfinite(logits).all())


def test_aasist_light_v2_forward_does_not_apply_softmax() -> None:
    import torch

    from app.models.architectures.aasist_light_v2 import build_aasist_light_v2_net

    module = build_aasist_light_v2_net()
    module.eval()
    waveform = torch.randn(2, 64_600, generator=torch.Generator().manual_seed(7))

    with torch.inference_mode():
        logits = module(waveform)

    row_sums = logits.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_aasist_light_v2_label_contract_is_bonafide_then_spoof() -> None:
    from app.models.architectures.aasist_light_v2 import LABEL_MAPPING

    assert LABEL_MAPPING == {"bonafide": 0, "spoof": 1}
