"""Shared helpers for the real CNN / AASIST-Light integration tests.

The trained checkpoints are gitignored (they are large binaries), so these
tests must degrade to a skip rather than a failure on a machine or CI runner
that does not have them. PyTorch is an optional extra for the same reason.

Checkpoint path resolution below deliberately mirrors production exactly
(same Settings fields, same `checkpoint_identity_for_path` function
`ModelFactory` itself uses) rather than guessing a fixed test-only layout.
Guessing previously caused this module to silently point at
`model_artifacts/best_cnn_full_weighted.pth` and
`model_artifacts/best_aasist_light_full_weighted.pth` -- neither of which is
the deployed checkpoint -- so every test gated on "checkpoint present" skipped
without anyone noticing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config.settings import Settings
from app.ingestion.audio import ProcessedAudio
from app.models.runtime import CheckpointIdentity, checkpoint_identity_for_path

#: `tests/real_model_helpers.py` -> parents[1] is `backend/`. Computed from
#: this file's own location (not the process cwd) so path resolution below is
#: identical whether pytest is invoked from `backend/` or the repo root.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parent
_ENV_FILE = _BACKEND_ROOT / ".env"

#: Kept for backward compatibility with tests that anchor AASIST V2's own
#: checkpoint directly (`CHECKPOINT_ROOT / "aasist" / "aasist_light_v2_best.pt"`,
#: matching `AASIST_MODEL_PATH`'s default). Those call sites were already
#: correct; only the CNN/legacy-AASIST resolution below was stale.
CHECKPOINT_ROOT = _REPO_ROOT / "model_artifacts"


def torch_installed() -> bool:
    from app.models.torch_support import torch_available

    return torch_available()


def transformers_installed() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def deployment_settings() -> Settings:
    """The same Settings a locally started server would build.

    `tests/conftest.py` sets `MULTISCOPE_DISABLE_DOTENV=1` before the general
    test suite imports the app, so ordinary tests never touch local secrets
    from `backend/.env`. This module's entire purpose is to verify what is
    ACTUALLY deployed on this machine -- model checkpoint paths are not
    secrets -- so it deliberately loads the real `.env` when present, and
    falls back to pure environment/defaults (matching a CI runner with no
    `.env` file) otherwise. Checkpoint identity resolution below is what
    matters here, not the unrelated Mongo/Clerk/Cloudinary fields this also
    happens to read.
    """

    return Settings(_env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else None)


_DEPLOYMENT_SETTINGS = deployment_settings()


def _resolve_checkpoint(configured_path: str) -> CheckpointIdentity:
    """Resolve a configured checkpoint path exactly the way `ModelFactory`
    does in production: same root, same supported-extension allowlist, same
    path-escape guard, same SHA-256/identity computation. This is not a
    reimplementation -- it is the literal function `ModelFactory.branch_config`
    calls -- so a test can never silently diverge from what the server
    actually loads.
    """

    return checkpoint_identity_for_path(
        configured_path,
        model_root_dir=_DEPLOYMENT_SETTINGS.resolved_model_root_dir,
        supported_extensions=set(
            _DEPLOYMENT_SETTINGS.supported_checkpoint_extension_list
        ),
    )


CNN_CHECKPOINT_IDENTITY = _resolve_checkpoint(_DEPLOYMENT_SETTINGS.cnn_model_path)
AASIST_CHECKPOINT_IDENTITY = _resolve_checkpoint(_DEPLOYMENT_SETTINGS.aasist_model_path)
SSL_CHECKPOINT_IDENTITY = _resolve_checkpoint(_DEPLOYMENT_SETTINGS.ssl_model_path)

#: `Path | None` -- `None` when the configured path does not resolve to a
#: real, in-bounds, correctly-extensioned file (unconfigured, missing,
#: escapes `MODEL_ROOT_DIR`, wrong extension, unreadable, ...). Prefer
#: `CNN_CHECKPOINT_IDENTITY` / `AASIST_CHECKPOINT_IDENTITY` when the failure
#: reason matters.
CNN_CHECKPOINT = CNN_CHECKPOINT_IDENTITY.safe_path
AASIST_CHECKPOINT = AASIST_CHECKPOINT_IDENTITY.safe_path
SSL_CHECKPOINT = SSL_CHECKPOINT_IDENTITY.safe_path

#: The superseded AASIST-Light V1 architecture's own checkpoint. Kept
#: deliberately separate from `AASIST_CHECKPOINT` above, which now correctly
#: identifies whatever is actually deployed as the real `aasist` branch
#: (AASIST-Light V2). A handful of legacy tests exercise the V1 module
#: (`app/models/architectures/aasist_light.py`) directly and need V1's own
#: checkpoint format (a flat state dict), not V2's (a dict with training
#: metadata) -- pointing them at `AASIST_CHECKPOINT` would silently test the
#: wrong architecture/checkpoint pairing, the exact failure mode this module
#: exists to prevent.
AASIST_LIGHT_V1_CHECKPOINT = _REPO_ROOT / "models" / "best_aasist_light_full_weighted.pth"
requires_aasist_light_v1_checkpoint = pytest.mark.skipif(
    not AASIST_LIGHT_V1_CHECKPOINT.is_file(),
    reason=(
        "Superseded AASIST-Light V1 checkpoint not present at "
        f"{AASIST_LIGHT_V1_CHECKPOINT}. V1 is not the deployed AASIST "
        "architecture -- see AASIST_CHECKPOINT for the real deployed one."
    ),
)

#: The superseded CNN checkpoint (log-mel, 3-block architecture --
#: `app/models/architectures/cnn.py` / `app/models/preprocessing/spectral.py`).
#: `CNN_CHECKPOINT` now correctly identifies CNN-V2 (LFCC-40+delta+delta-delta,
#: 4-block VGG architecture -- `app/models/architectures/cnn_v2.py`); legacy
#: tests that specifically exercise the old architecture/checkpoint pairing
#: use this constant instead, mirroring `AASIST_LIGHT_V1_CHECKPOINT` above.
CNN_LEGACY_CHECKPOINT = _REPO_ROOT / "models" / "best_cnn_full_weighted.pth"
requires_cnn_legacy_checkpoint = pytest.mark.skipif(
    not CNN_LEGACY_CHECKPOINT.is_file(),
    reason=(
        "Superseded CNN checkpoint not present at "
        f"{CNN_LEGACY_CHECKPOINT}. It is not the deployed CNN architecture -- "
        "see CNN_CHECKPOINT for the real deployed one (CNN-V2)."
    ),
)

#: The superseded SSL Generalization V2 checkpoint (trainable-state-only,
#: no embedded self-description). `SSL_CHECKPOINT` now correctly identifies
#: the ASVspoof5 artifact (self-describing: xlsr_model_name/label_mapping/
#: sample_rate/max_duration_seconds); tests specific to the Generalization V2
#: artifact's own (different) format use this constant instead.
SSL_GENERALIZATION_V2_CHECKPOINT = (
    _REPO_ROOT / "model_artifacts" / "ssl" / "ssl_xlsr_mamba_generalization_v2_best.pt"
)
requires_ssl_generalization_v2_checkpoint = pytest.mark.skipif(
    not SSL_GENERALIZATION_V2_CHECKPOINT.is_file(),
    reason=(
        "Superseded SSL Generalization V2 checkpoint not present at "
        f"{SSL_GENERALIZATION_V2_CHECKPOINT}. It is not the deployed SSL "
        "artifact -- see SSL_CHECKPOINT for the real deployed one (ASVspoof5)."
    ),
)


def _skip_reason(label: str, configured_path: str, identity: CheckpointIdentity) -> str:
    return (
        f"{label} checkpoint not resolvable via the production Settings path "
        f"(MODEL_ROOT_DIR={_DEPLOYMENT_SETTINGS.model_root_dir!r} -> "
        f"{_DEPLOYMENT_SETTINGS.resolved_model_root_dir}, configured "
        f"path={configured_path!r}"
        + (
            f", resolved={identity.safe_path}"
            if identity.safe_path is not None
            else ""
        )
        + f"): {identity.error_code or 'unknown error'}. If a checkpoint is "
        "expected on this machine, check backend/.env's CNN_MODEL_PATH / "
        "AASIST_MODEL_PATH / MODEL_ROOT_DIR rather than this test file."
    )


requires_torch = pytest.mark.skipif(
    not torch_installed(),
    reason="PyTorch is not installed (optional 'models' extra).",
)
requires_cnn_checkpoint = pytest.mark.skipif(
    not CNN_CHECKPOINT_IDENTITY.valid,
    reason=_skip_reason(
        "CNN", _DEPLOYMENT_SETTINGS.cnn_model_path, CNN_CHECKPOINT_IDENTITY
    ),
)
requires_aasist_checkpoint = pytest.mark.skipif(
    not AASIST_CHECKPOINT_IDENTITY.valid,
    reason=_skip_reason(
        "AASIST", _DEPLOYMENT_SETTINGS.aasist_model_path, AASIST_CHECKPOINT_IDENTITY
    ),
)
requires_ssl_checkpoint = pytest.mark.skipif(
    not SSL_CHECKPOINT_IDENTITY.valid,
    reason=_skip_reason(
        "SSL", _DEPLOYMENT_SETTINGS.ssl_model_path, SSL_CHECKPOINT_IDENTITY
    ),
)
requires_transformers = pytest.mark.skipif(
    not transformers_installed(),
    reason="transformers is not installed (optional 'models' extra).",
)


def real_model_settings(**overrides: object) -> Settings:
    """Settings with CNN/AASIST real, SSL/glottal disabled -- the current
    deployment stage -- resolved from the SAME `MODEL_ROOT_DIR` /
    `CNN_MODEL_PATH` / `AASIST_MODEL_PATH` the running server reads from
    `backend/.env`, not a hardcoded test-only path. ``overrides`` still wins
    over these (e.g. a test that deliberately passes a bad path to exercise
    the missing-checkpoint branch).
    """

    base: dict[str, object] = {
        "cnn_model_mode": "real",
        "aasist_model_mode": "real",
        "ssl_model_mode": "disabled",
        "glottal_model_mode": "disabled",
        "model_root_dir": _DEPLOYMENT_SETTINGS.model_root_dir,
        "cnn_model_path": _DEPLOYMENT_SETTINGS.cnn_model_path,
        "aasist_model_path": _DEPLOYMENT_SETTINGS.aasist_model_path,
        "ssl_model_path": _DEPLOYMENT_SETTINGS.ssl_model_path,
        "fusion_contract_path": _DEPLOYMENT_SETTINGS.fusion_contract_path,
        "required_model_branches": "lfcc_cnn_tcn,aasist",
        "model_device_policy": "cpu",
        "model_default_device": "cpu",
        "model_load_strategy": "lazy",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def synthetic_speech(
    *,
    seconds: float = 3.0,
    sample_rate: int = 16000,
    seed: int = 0,
) -> np.ndarray:
    """A deterministic harmonic-plus-noise signal.

    Not real speech and not a spoofing artefact -- these tests assert on the
    mechanics of the inference path (shapes, ranges, determinism, error
    handling), never on whether a particular verdict is correct. Judging
    accuracy would require labelled evaluation audio, which this repository
    does not contain.
    """

    time = np.arange(int(seconds * sample_rate)) / sample_rate
    generator = np.random.RandomState(seed)
    signal = (
        0.30 * np.sin(2 * np.pi * 140 * time)
        + 0.15 * np.sin(2 * np.pi * 280 * time)
        + 0.02 * generator.randn(time.size)
    )
    return signal.astype(np.float32)


def processed_audio(
    waveform: np.ndarray | None = None,
    *,
    sample_rate: int = 16000,
) -> ProcessedAudio:
    """Build a ``ProcessedAudio`` for a real-branch test.

    Deliberately does NOT route through the shared production
    `preprocess_waveform` pipeline: some callers (e.g.
    `test_non_finite_audio_is_rejected_rather_than_fed_to_the_model`)
    intentionally inject a corrupted/non-finite waveform to prove the model
    adapter's own input validation is a real second line of defence, not just
    a restatement of ingestion's NaN sanitisation -- routing through
    `preprocess_waveform` would silently sanitise that NaN away before the
    assertion ever runs.

    `unnormalised_waveform` is populated (previously it was omitted, so any
    real AASIST V2 branch -- which requires it -- raised
    `model_input_invalid` before this helper could be used at all; see
    `test_real_model_helpers_processed_audio_can_drive_real_aasist_v2` for the
    regression test). This helper does not itself apply the ~0.95-peak
    normalisation the real ingestion pipeline performs, so `waveform` and
    `unnormalised_waveform` are the same array here -- adequate for the shape,
    determinism, and label-mapping assertions these tests make, but not a
    stand-in for testing the real amplitude-normalisation behaviour, which
    belongs to `tests/test_audio_preprocessing.py`.
    """

    samples = synthetic_speech(sample_rate=sample_rate) if waveform is None else waveform
    samples = np.asarray(samples, dtype=np.float32)
    return ProcessedAudio(
        waveform=samples,
        sample_rate=sample_rate,
        original_sample_rate=sample_rate,
        original_channels=1,
        duration_seconds=float(samples.size / sample_rate),
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=float(np.abs(samples).max()),
        rms_energy=float(np.sqrt((samples**2).mean())),
        unnormalised_waveform=np.ascontiguousarray(samples.copy(), dtype=np.float32),
    )
