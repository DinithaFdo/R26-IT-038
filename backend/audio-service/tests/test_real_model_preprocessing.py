"""Branch front ends: determinism, shape contracts, and normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from tests.real_model_helpers import requires_torch, synthetic_speech

pytestmark = requires_torch


def _spectral(**overrides):
    from app.models.preprocessing.spectral import (
        SpectralFeatureConfig,
        SpectralFeatureExtractor,
    )

    return SpectralFeatureExtractor(SpectralFeatureConfig(**overrides))


def _waveform_front_end(**overrides):
    from app.models.preprocessing.waveform import WaveformConfig, WaveformFrontEnd

    return WaveformFrontEnd(WaveformConfig(**overrides))


@pytest.mark.parametrize("feature_type", ["log_mel", "mfcc", "lfcc"])
def test_every_feature_type_produces_the_declared_shape(feature_type: str) -> None:
    import torch

    extractor = _spectral(feature_type=feature_type, n_filters=40, n_coefficients=40, max_frames=400)
    features = extractor.extract(torch.from_numpy(synthetic_speech()))

    assert tuple(features.shape) == (1, 1, 40, 400)
    assert bool(torch.isfinite(features).all())


def test_deltas_double_the_coefficient_axis() -> None:
    import torch

    extractor = _spectral(n_coefficients=20, n_filters=40, include_deltas=True, max_frames=300)
    features = extractor.extract(torch.from_numpy(synthetic_speech()))

    assert tuple(features.shape) == (1, 1, 40, 300)
    assert extractor.config.output_coefficients == 40


def test_feature_extraction_is_deterministic() -> None:
    import torch

    extractor = _spectral()
    audio = torch.from_numpy(synthetic_speech())

    assert torch.equal(extractor.extract(audio), extractor.extract(audio))


def test_long_audio_is_centre_cropped_not_randomly_cropped() -> None:
    """Evaluation must give the same answer for the same clip, every time."""

    import torch

    extractor = _spectral(max_frames=100)
    audio = torch.from_numpy(synthetic_speech(seconds=10.0))

    first = extractor.extract(audio)
    second = extractor.extract(audio)

    assert tuple(first.shape) == (1, 1, 40, 100)
    assert torch.equal(first, second)


def test_audio_shorter_than_one_analysis_window_is_padded_not_rejected() -> None:
    import torch

    extractor = _spectral(max_frames=50)
    features = extractor.extract(torch.zeros(100))

    assert tuple(features.shape) == (1, 1, 40, 50)
    assert bool(torch.isfinite(features).all())


def test_per_coefficient_normalisation_centres_each_coefficient() -> None:
    """The default matches the zero-mean/unit-variance distribution recovered
    from the checkpoint's BatchNorm running statistics."""

    import torch

    extractor = _spectral(normalization="per_coefficient_zscore", max_frames=200)
    features = extractor.extract(torch.from_numpy(synthetic_speech()))[0, 0]

    assert torch.allclose(features.mean(dim=1), torch.zeros(40), atol=1e-4)
    assert torch.allclose(features.std(dim=1), torch.ones(40), atol=1e-2)


def test_disabling_normalisation_leaves_coefficients_on_different_scales() -> None:
    """Without normalisation each coefficient keeps its own offset.

    That is the distribution the checkpoint's BatchNorm forensics rule out, so
    it must be visibly different from the normalised default -- but it stays
    available so the pipeline can be corrected without a code change.
    """

    import torch

    extractor = _spectral(normalization="none", feature_type="log_mel", max_frames=200)
    features = extractor.extract(torch.from_numpy(synthetic_speech()))[0, 0]

    per_coefficient_means = features.mean(dim=1)
    assert float(per_coefficient_means.std()) > 1.0

    normalised = _spectral(
        normalization="per_coefficient_zscore", feature_type="log_mel", max_frames=200
    ).extract(torch.from_numpy(synthetic_speech()))[0, 0]
    assert float(normalised.mean(dim=1).std()) < 1e-3


def test_invalid_feature_configuration_is_rejected() -> None:
    from app.models.preprocessing.spectral import SpectralFeatureConfig

    with pytest.raises(ValueError):
        SpectralFeatureConfig(n_coefficients=64, n_filters=40)
    with pytest.raises(ValueError):
        SpectralFeatureConfig(win_length=1024, n_fft=512)
    with pytest.raises(ValueError):
        SpectralFeatureConfig(max_frames=0)


def test_waveform_front_end_produces_the_declared_length() -> None:
    import torch

    front_end = _waveform_front_end(target_samples=64000)
    prepared = front_end.prepare(torch.from_numpy(synthetic_speech(seconds=1.0)))

    assert tuple(prepared.shape) == (1, 1, 64000)
    assert bool(torch.isfinite(prepared).all())


def test_long_waveform_is_centre_cropped_deterministically() -> None:
    import torch

    front_end = _waveform_front_end(target_samples=16000)
    audio = torch.from_numpy(synthetic_speech(seconds=10.0))

    assert torch.equal(front_end.prepare(audio), front_end.prepare(audio))
    assert tuple(front_end.prepare(audio).shape) == (1, 1, 16000)


def test_peak_normalisation_scales_to_unit_amplitude() -> None:
    import torch

    front_end = _waveform_front_end(normalization="peak", target_samples=16000)
    prepared = front_end.prepare(torch.from_numpy(synthetic_speech() * 0.01))

    assert float(prepared.abs().max()) == pytest.approx(1.0, abs=1e-5)


def test_rms_normalisation_hits_the_configured_target() -> None:
    import torch

    front_end = _waveform_front_end(
        normalization="rms", target_rms=0.05, target_samples=16000
    )
    prepared = front_end.prepare(torch.from_numpy(synthetic_speech()))

    assert float(torch.sqrt(prepared.pow(2).mean())) == pytest.approx(0.05, abs=1e-4)


def test_silence_is_not_amplified_into_noise() -> None:
    """Dividing digital silence by its own peak would produce garbage."""

    import torch

    front_end = _waveform_front_end(normalization="peak", target_samples=1000)
    prepared = front_end.prepare(torch.zeros(1000))

    assert bool(torch.isfinite(prepared).all())
    assert float(prepared.abs().max()) == 0.0


def test_repeat_padding_tiles_short_audio_instead_of_zero_filling() -> None:
    import torch

    front_end = _waveform_front_end(
        target_samples=1000, length_policy="repeat_pad", normalization="none"
    )
    prepared = front_end.prepare(torch.arange(100, dtype=torch.float32))[0, 0]

    assert int((prepared != 0).sum()) > 900


def test_waveform_config_rejects_a_length_the_frontend_cannot_survive() -> None:
    """Three MaxPool1d(4) stages divide time by 64; below that the GRU is empty."""

    from app.models.preprocessing.waveform import WaveformConfig

    with pytest.raises(ValueError):
        WaveformConfig(target_samples=32)


def test_aasist_light_v2_long_waveform_uses_first_samples_not_centre_crop() -> None:
    import torch

    from app.models.preprocessing.aasist_light_v2 import (
        AasistLightV2WaveformConfig,
        preprocess_aasist_light_v2_waveform,
    )

    config = AasistLightV2WaveformConfig()
    waveform = torch.arange(70_000, dtype=torch.float32)

    prepared = preprocess_aasist_light_v2_waveform(waveform, config=config)
    expected_source = waveform[: config.target_samples]
    expected = (expected_source - expected_source.mean()) / (
        expected_source.std(unbiased=False) + config.zscore_epsilon
    )

    assert prepared.numel() == 64_600
    assert torch.allclose(prepared, expected)


def test_aasist_light_v2_short_waveform_is_right_zero_padded() -> None:
    import torch

    from app.models.preprocessing.aasist_light_v2 import (
        AasistLightV2WaveformConfig,
        preprocess_aasist_light_v2_waveform,
    )

    config = AasistLightV2WaveformConfig()
    waveform = torch.linspace(-1.0, 1.0, 50_000)

    prepared = preprocess_aasist_light_v2_waveform(waveform, config=config)
    expected_source = torch.nn.functional.pad(
        waveform,
        (0, config.target_samples - waveform.numel()),
    )
    expected = (expected_source - expected_source.mean()) / (
        expected_source.std(unbiased=False) + config.zscore_epsilon
    )

    assert prepared.numel() == 64_600
    assert torch.allclose(prepared, expected)


def test_aasist_light_v2_zscore_normalises_typical_waveform() -> None:
    import torch

    from app.models.preprocessing.aasist_light_v2 import (
        preprocess_aasist_light_v2_waveform,
    )

    waveform = torch.from_numpy(synthetic_speech(seconds=5.0))

    prepared = preprocess_aasist_light_v2_waveform(waveform)

    assert float(prepared.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(prepared.std(unbiased=False)) == pytest.approx(1.0, abs=1e-4)


def test_aasist_light_v2_rejects_near_silent_waveform() -> None:
    import torch

    from app.core.exceptions import ModelInferenceError
    from app.models.preprocessing.aasist_light_v2 import (
        preprocess_aasist_light_v2_waveform,
    )

    waveform = torch.full((64_600,), 1e-8, dtype=torch.float32)

    with pytest.raises(ModelInferenceError):
        preprocess_aasist_light_v2_waveform(waveform)


def test_aasist_light_v2_preprocessing_does_not_peak_normalise() -> None:
    import torch

    from app.models.preprocessing.aasist_light_v2 import (
        AasistLightV2WaveformConfig,
        preprocess_aasist_light_v2_waveform,
    )

    config = AasistLightV2WaveformConfig()
    waveform = torch.linspace(-4e-7, 4e-7, config.target_samples)

    prepared = preprocess_aasist_light_v2_waveform(waveform, config=config)
    expected = (waveform - waveform.mean()) / (
        waveform.std(unbiased=False) + config.zscore_epsilon
    )

    assert torch.allclose(prepared, expected, atol=1e-7)
    assert float(prepared.std(unbiased=False)) < 0.3


def test_aasist_light_v2_frontend_declares_unnormalised_waveform_requirement() -> None:
    from app.models.preprocessing.aasist_light_v2 import AasistLightV2WaveformFrontEnd

    assert AasistLightV2WaveformFrontEnd.requires_unnormalized_waveform is True


def test_real_inference_uses_unnormalised_waveform_for_opt_in_frontends() -> None:
    from types import SimpleNamespace

    import torch

    from app.ingestion.audio import ProcessedAudio
    from app.models.real.inference import _prepare_input

    class RawWaveformFrontEnd:
        requires_unnormalized_waveform = True

        def prepare(self, waveform):
            return waveform.unsqueeze(0).unsqueeze(0)

    processed = ProcessedAudio(
        waveform=torch.ones(10, dtype=torch.float32).numpy(),
        unnormalised_waveform=torch.arange(10, dtype=torch.float32).numpy(),
        sample_rate=16_000,
        original_sample_rate=16_000,
        original_channels=1,
        duration_seconds=10 / 16_000,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=1.0,
        rms_energy=1.0,
    )
    runtime_model = SimpleNamespace(
        front_end=RawWaveformFrontEnd(),
        preprocessing={"sample_rate": 16_000},
        device="cpu",
    )
    config = SimpleNamespace(branch_name="aasist")

    prepared = _prepare_input(torch, processed, runtime_model, config)

    assert torch.equal(prepared[0, 0], torch.arange(10, dtype=torch.float32))


def test_existing_cnn_and_aasist_v1_frontends_do_not_opt_into_raw_waveform() -> None:
    from app.models.preprocessing.spectral import SpectralFeatureExtractor
    from app.models.preprocessing.ssl_waveform import (
        SSLWaveformConfig,
        SSLWaveformFrontEnd,
    )
    from app.models.preprocessing.waveform import WaveformFrontEnd

    assert getattr(_spectral(), "requires_unnormalized_waveform", False) is False
    assert (
        getattr(_waveform_front_end(), "requires_unnormalized_waveform", False)
        is False
    )
    ssl_front_end = SSLWaveformFrontEnd(SSLWaveformConfig())
    assert getattr(ssl_front_end, "requires_unnormalized_waveform", False) is False
    assert not hasattr(SpectralFeatureExtractor, "requires_unnormalized_waveform")
    assert not hasattr(WaveformFrontEnd, "requires_unnormalized_waveform")


def test_aasist_public_branch_identifier_remains_canonical() -> None:
    from app.models.runtime import (
        BRANCH_ALIASES,
        CANONICAL_BRANCH_ORDER,
        DISPLAY_NAMES,
        PUBLIC_MODEL_NAMES,
    )

    assert "aasist" in CANONICAL_BRANCH_ORDER
    assert BRANCH_ALIASES["aasist"] == "aasist"
    assert PUBLIC_MODEL_NAMES["aasist"] == "aasist"
    assert DISPLAY_NAMES["aasist"] == "AASIST"


def test_extractor_matches_numpy_reference_for_dct_orthonormality() -> None:
    """The DCT matrix must be orthonormal, matching scipy's norm='ortho'."""

    import torch

    extractor = _spectral(feature_type="mfcc", n_filters=40, n_coefficients=40)
    matrix = extractor._build_dct_matrix()
    product = (matrix @ matrix.T).numpy()

    assert np.allclose(product, np.eye(40), atol=1e-5)
