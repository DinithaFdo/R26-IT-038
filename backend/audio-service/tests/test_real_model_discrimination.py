"""Guards against shipping a feature configuration the checkpoint cannot use.

Because both networks accept any input shape, a wrong front end does not raise
— it saturates. The first configuration tried here returned spoof=1.000000 for
every input: a detector that always says the same thing, with no error anywhere
to reveal it. Nothing in the type system, the schema, or `strict=True` loading
catches that.

These tests assert the *shipped defaults* still produce a model that responds to
its input. They deliberately assert nothing about which verdict is right —
that needs labelled evaluation audio, which this repository does not contain.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config.settings import Settings
from app.models.factory import ModelFactory
from tests.real_model_helpers import (
    CNN_LEGACY_CHECKPOINT,
    processed_audio,
    real_model_settings,
    requires_aasist_checkpoint,
    requires_cnn_checkpoint,
    requires_cnn_legacy_checkpoint,
    requires_torch,
)

pytestmark = requires_torch

SAMPLE_RATE = 16000


def _legacy_cnn_spoof_probability(waveform, *, feature_config) -> float:
    """Run the superseded log-mel/3-block CNN directly (not through
    ModelFactory, which now always builds CNN-V2 for ``lfcc_cnn_tcn``).

    Mirrors the production forward pass (build -> strict load -> softmax)
    without the adapter/schema machinery, since this file's tests are only
    about the front end's discrimination behaviour, not the full contract.
    """

    import torch

    from app.models.architectures.cnn import build_cnn_acoustic_net
    from app.models.preprocessing.spectral import SpectralFeatureExtractor
    from app.models.torch_support import load_state_dict_strict

    module = build_cnn_acoustic_net()
    load_state_dict_strict(module, CNN_LEGACY_CHECKPOINT)
    module.eval()

    features = SpectralFeatureExtractor(feature_config).extract(
        torch.from_numpy(waveform.copy())
    )
    with torch.inference_mode():
        logits = module(features)
    return float(torch.softmax(logits, dim=-1)[0, 1])


def probe_signals() -> dict[str, np.ndarray]:
    """Acoustically diverse probes. Not speech, and not labelled — they exist
    only to show the model's output moves when its input does."""

    time = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    generator = np.random.RandomState(0)
    return {
        "harmonic": (
            0.30 * np.sin(2 * np.pi * 140 * time)
            + 0.15 * np.sin(2 * np.pi * 280 * time)
            + 0.02 * generator.randn(time.size)
        ).astype(np.float32),
        "broadband_noise": (0.20 * generator.randn(time.size)).astype(np.float32),
        "pure_tone": (0.50 * np.sin(2 * np.pi * 440 * time)).astype(np.float32),
        "chirp": (0.40 * np.sin(2 * np.pi * (100 + 300 * time / 3) * time)).astype(np.float32),
    }


def spoof_probabilities(model) -> list[float]:
    return [
        model.predict_safe(processed_audio(signal)).probabilities.spoof
        for signal in probe_signals().values()
    ]


@requires_cnn_checkpoint
def test_the_shipped_cnn_feature_defaults_are_not_degenerate() -> None:
    """The CNN must not return an identical probability for every input."""

    model = ModelFactory(real_model_settings()).create("lfcc_cnn_tcn")
    probabilities = spoof_probabilities(model)

    assert max(probabilities) - min(probabilities) > 0.01, (
        "The default CNN feature configuration drives the checkpoint into "
        f"saturation: {probabilities}. Fix CNN_FEATURE_* rather than relaxing "
        "this test — a constant output is an undetectable failure in production."
    )


@requires_aasist_checkpoint
def test_the_shipped_aasist_waveform_defaults_are_not_degenerate() -> None:
    model = ModelFactory(real_model_settings()).create("aasist")
    probabilities = spoof_probabilities(model)

    assert max(probabilities) - min(probabilities) > 0.01, (
        f"The default AASIST waveform configuration is degenerate: {probabilities}."
    )


@requires_cnn_legacy_checkpoint
def test_the_known_degenerate_configuration_is_still_detected() -> None:
    """Pins the failure mode this guard exists for, against the SUPERSEDED
    log-mel/3-block CNN checkpoint (``models/best_cnn_full_weighted.pth``).

    CNN-V2 (the live default for ``lfcc_cnn_tcn``) hardcodes its own
    LFCC-40+delta+delta-delta front end and ignores ``CNN_FEATURE_*``
    entirely (see ``build_cnn_loader``), so this configuration-sensitivity
    guard no longer applies to it -- only to the legacy checkpoint/front-end
    pairing this test now exercises directly.

    If this configuration ever stops being degenerate for the legacy
    checkpoint, the detector above needs revisiting — it would mean the
    front end changed materially.
    """

    from app.models.preprocessing.spectral import SpectralFeatureConfig

    feature_config = SpectralFeatureConfig(
        feature_type="lfcc",
        normalization="per_coefficient_zscore",
        n_filters=40,
        n_coefficients=40,
    )
    probabilities = [
        _legacy_cnn_spoof_probability(signal, feature_config=feature_config)
        for signal in probe_signals().values()
    ]

    assert max(probabilities) - min(probabilities) < 0.01


@requires_cnn_legacy_checkpoint
@requires_aasist_checkpoint
def test_branch_disagreement_is_recorded_as_a_known_open_question() -> None:
    """Documents the measured relationship between the SUPERSEDED CNN
    checkpoint and AASIST.

    Across the probe set the branches agree far more often when configured
    with OPPOSITE class orders than with the same one, which indicates the
    two checkpoints were trained with different label conventions. This test
    pins that observation for the legacy CNN checkpoint so it cannot quietly
    change; it does NOT assert which of the two is correct, because the
    probes are unlabelled.

    CNN-V2 (the live default) does not carry this ambiguity: its own
    checkpoint attests ``label_map=["bonafide","spoof"]``/
    ``positive_class="spoof"`` directly (see
    ``test_cnn_v2_checkpoint_identity_is_recorded_through_the_provenance_mechanism``,
    ``tests/test_cnn_verification_guard.py``), matching AASIST's own
    attestation -- there is no comparable open question left to pin for it.
    """

    from app.models.preprocessing.spectral import SpectralFeatureConfig

    default_feature_config = SpectralFeatureConfig(
        feature_type="log_mel", normalization="global_zscore", n_filters=20, n_coefficients=20
    )

    def agreement(cnn_order: str, aasist_order: str) -> int:
        aasist = ModelFactory(real_model_settings(aasist_class_order=aasist_order)).create(
            "aasist"
        )
        spoof_index = 1 if cnn_order == "bonafide_spoof" else 0
        matches = 0
        for signal in probe_signals().values():
            audio = processed_audio(signal)
            raw_spoof = _legacy_cnn_spoof_probability(signal, feature_config=default_feature_config)
            cnn_spoof = raw_spoof if spoof_index == 1 else 1.0 - raw_spoof
            aasist_spoof = aasist.predict_safe(audio).probabilities.spoof
            matches += int((cnn_spoof >= 0.5) == (aasist_spoof >= 0.5))
        return matches

    same_order = agreement("bonafide_spoof", "bonafide_spoof")
    opposite_order = agreement("bonafide_spoof", "spoof_bonafide")

    assert opposite_order > same_order, (
        "The two checkpoints no longer look inverted relative to each other. "
        "Re-check CNN_CLASS_ORDER / AASIST_CLASS_ORDER against the training "
        f"label map (same={same_order}, opposite={opposite_order})."
    )


@requires_cnn_checkpoint
def test_settings_defaults_match_the_documented_evidence_based_choice() -> None:
    """The default must not silently drift back to a degenerate configuration."""

    defaults = Settings(
        cnn_model_mode="dummy",
        aasist_model_mode="dummy",
        ssl_model_mode="disabled",
        glottal_model_mode="disabled",
    )

    assert defaults.cnn_feature_type == "log_mel"
    assert defaults.cnn_feature_normalization == "global_zscore"
    assert defaults.cnn_feature_filters == 20
    # Unverified until someone confirms against training. Fail closed.
    assert defaults.cnn_preprocessing_verified is False
    assert defaults.cnn_class_mapping_verified is False
    assert defaults.aasist_preprocessing_verified is False
    assert defaults.aasist_class_mapping_verified is False
