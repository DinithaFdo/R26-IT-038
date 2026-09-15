import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:  # Import only for typing: these pull in torch at runtime.
    from app.models.preprocessing.spectral import SpectralFeatureConfig
    from app.models.preprocessing.waveform import WaveformConfig

ModelMode = Literal["dummy", "real", "disabled"]
StoragePolicy = Literal["required", "optional", "disabled"]
PaddingPolicy = Literal["right_zero"]
TrimPolicy = Literal["none", "disabled"]
DevicePolicy = Literal["cpu", "cuda", "mps", "auto"]
CnnFeatureType = Literal["log_mel", "mfcc", "lfcc"]
CnnFeatureNormalization = Literal["none", "global_zscore", "per_coefficient_zscore"]
AasistWaveformNormalization = Literal["none", "peak", "rms"]
AasistLengthPolicy = Literal["center_crop_pad", "repeat_pad"]
ClassOrder = Literal["bonafide_spoof", "spoof_bonafide"]
PrecisionPolicy = Literal["float32", "float16", "bfloat16"]
LoadStrategy = Literal["startup", "lazy"]
FusionMethod = Literal["weighted_average", "simple_average", "majority_vote"]
FusionModeSetting = Literal["convex_4branch_v3", "legacy_3branch"]
XaiMode = Literal["disabled", "mock", "real"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "MULTI-SCOPE Voice Classification API"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    docs_enabled: bool = True
    openapi_enabled: bool = True
    model_health_enabled: bool = True
    trusted_hosts: str = "localhost,127.0.0.1,testserver"

    upload_dir: str = "uploads"
    max_request_body_mb: int = 27
    max_upload_size_mb: int = 25
    allowed_audio_extensions: str = "wav,flac,mp3,m4a,aac,opus,ogg,webm"
    target_sample_rate: int = 16000
    min_audio_duration_seconds: float = Field(default=1.0, gt=0)
    max_audio_duration_seconds: float = 180.0
    max_audio_channels: int = 2
    max_input_sample_rate: int = Field(
        default=96000,
        validation_alias=AliasChoices("MAX_INPUT_SAMPLE_RATE", "MAX_AUDIO_SAMPLE_RATE"),
    )
    max_decoded_audio_size_mb: int = 32
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    ffmpeg_timeout_seconds: int = 30
    ffprobe_timeout_seconds: int = 15

    mongodb_uri: str = ""
    mongodb_database: str = "multiscope"
    mongodb_required: bool = True
    mongodb_connect_timeout_ms: int = 5000
    mongodb_server_selection_timeout_ms: int = 5000
    mongodb_socket_timeout_ms: int = 20000
    mongodb_retry_reads: bool = True
    mongodb_retry_writes: bool = True

    clerk_issuer: str = ""
    clerk_jwks_url: str = ""
    clerk_audience: str = ""
    clerk_authorized_parties: str = ""
    clerk_jwks_cache_seconds: int = 3600
    clerk_jwks_min_refresh_interval_seconds: int = Field(default=60, ge=0)
    clerk_jwks_negative_kid_cache_seconds: int = Field(default=60, ge=0)
    dev_auth_bypass: bool = False

    api_key_hash_secret: str = "development-only-change-me"

    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""
    cloudinary_audio_folder: str = "multiscope/audio"
    storage_policy: StoragePolicy = "required"
    cloudinary_storage_enabled: bool = True
    cloudinary_sdk_timeout_seconds: float = Field(default=60, gt=0)
    cloudinary_upload_timeout_seconds: float = Field(default=60, gt=0)

    prediction_job_timeout_seconds: float = Field(default=180, gt=0)
    prediction_shutdown_timeout_seconds: float = Field(default=30, gt=0)
    max_concurrent_predictions: int = Field(
        default=2,
        ge=1,
        validation_alias=AliasChoices(
            "MAX_CONCURRENT_PREDICTIONS",
            "PREDICTION_JOB_CONCURRENCY",
            "prediction_job_concurrency",
        ),
    )
    max_queued_predictions: int = Field(default=4, ge=0)
    max_active_predictions_per_principal: int = Field(default=1, ge=1)
    rate_limit_window_seconds: float = Field(default=60, gt=0)
    rate_limit_requests_per_window: int = Field(default=120, ge=1)
    prediction_rate_limit_per_window: int = Field(default=20, ge=1)
    api_key_creation_rate_limit_per_window: int = Field(default=10, ge=1)
    enable_legacy_anonymous_prediction: bool = False
    mcp_small_payload_max_bytes: int = Field(default=1_048_576, ge=1)

    xai_enabled: bool = False
    xai_mode: XaiMode = "disabled"
    xai_pipeline_version: str = "voice-xai-pipeline-v1"
    # Backwards-compatible name: this currently controls active XAI admission
    # capacity together with XAI_MAX_QUEUED_JOBS. The queue runs on one
    # persistent OS thread to avoid unsafe GPU/Mongo event-loop concurrency.
    xai_max_workers: int = Field(default=1, ge=1)
    xai_max_queued_jobs: int = Field(default=8, ge=0)
    # Temporal attention extraction runs XLS-R over several sliding windows;
    # the established pipeline can legitimately take longer than a standard
    # request on a CPU-only development machine.
    xai_job_timeout_seconds: float = Field(default=600.0, gt=0.0)
    xai_artifact_retention_seconds: int = Field(default=604800, ge=60)
    xai_artifact_root: str = "private_xai_artifacts"
    xai_temporal_threshold_config_path: str = (
        "app/voice_xai/temporal/calibrations/partialspoof_v1_2_dev_attention_threshold.json"
    )
    # Must match the locked temporal-calibration artifact. This is separate
    # from the shared-model segmentation settings used by other branches.
    xai_temporal_window_stride_seconds: float = Field(default=3.0, gt=0.0)
    xai_temporal_global_hop_seconds: float = Field(default=0.02, gt=0.0)
    xai_temporal_smoothing_ms: float = Field(default=80.0, ge=0.0)
    xai_temporal_minimum_region_duration_seconds: float = Field(default=0.04, ge=0.0)
    xai_temporal_merge_gap_seconds: float = Field(default=0.1, ge=0.0)
    # Display-only selection. It deliberately does not replace the fixed
    # PartialSpoof DEV-calibrated evaluation threshold.
    xai_temporal_visualization_quantile: float = Field(
        default=0.90, ge=0.50, le=0.99
    )
    xai_temporal_visualization_minimum_density: float = Field(
        default=1.0, ge=0.0
    )
    # Bounded display-only spectrogram generated from the same preprocessed
    # waveform as the classifier. These values do not alter model inference.
    xai_spectrogram_n_mels: int = Field(default=64, ge=16, le=128)
    xai_spectrogram_n_fft: int = Field(default=512, ge=128, le=4096)
    xai_spectrogram_hop_length: int = Field(default=160, ge=1, le=4096)
    xai_spectrogram_max_frames: int = Field(default=900, ge=32, le=2000)
    xai_semantic_model_path: str = (
        "app/voice_xai/semantic/artifacts/xgboost_surrogate_v4.json"
    )
    xai_semantic_manifest_path: str = (
        "app/voice_xai/semantic/manifests/xgboost-surrogate-v4.json"
    )
    xai_semantic_feature_columns_path: str = (
        "app/voice_xai/semantic/artifacts/feature_columns_v4.json"
    )
    xai_semantic_threshold_path: str = (
        "app/voice_xai/semantic/artifacts/threshold_v4.json"
    )
    xai_semantic_training_metadata_path: str = (
        "app/voice_xai/semantic/artifacts/training_metadata_v4.json"
    )
    xai_semantic_imputer_path: str = (
        "app/voice_xai/semantic/artifacts/imputer_v4.joblib"
    )
    xai_semantic_shap_background_path: str = ""
    xai_semantic_window_duration_seconds: float = Field(default=1.0, gt=0.0)
    xai_semantic_window_overlap_seconds: float = Field(default=0.5, ge=0.0)
    # Intermediate-tensor capture is opt-in even when XAI itself is enabled.
    # Hooks necessarily run in the classifier worker, so keeping this false by
    # default preserves the normal prediction latency profile.
    xai_capture_enabled: bool = False
    xai_capture_max_total_elements: int = Field(default=1_000_000, ge=1)

    # Optional post-processing only: this client receives compact validated
    # XAI evidence, never raw audio or classifier inputs. It is disabled by
    # default so normal prediction and deterministic-report behaviour stays
    # unchanged until Model Studio is explicitly configured.
    xai_narrative_enabled: bool = False
    xai_narrative_api_key: str = ""
    xai_narrative_base_url: str = ""
    xai_narrative_model: str = "qwen3.6-flash"
    xai_narrative_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    xai_narrative_enable_thinking: bool = True
    xai_narrative_thinking_budget: int = Field(default=1024, ge=128, le=32768)
    xai_narrative_summary_max_words: int = Field(default=200, ge=50, le=500)
    xai_narrative_detailed_max_words: int = Field(default=500, ge=100, le=1000)
    xai_narrative_max_semantic_features: int = Field(default=6, ge=1, le=20)
    # Bound the independent narrative-only retry path. A provider hiccup must
    # not force users to repeat the deterministic semantic/temporal analysis.
    xai_narrative_retry_max_attempts: int = Field(default=3, ge=1, le=10)
    xai_narrative_retry_rate_limit_requests: int = Field(default=3, ge=1)
    xai_narrative_retry_rate_limit_window_seconds: float = Field(
        default=60.0, gt=0.0
    )

    # Per-authenticated-user limits on explanation trigger/retry, independent
    # of the generic per-path RateLimitMiddleware (which keys on the literal
    # request path, including prediction_id, and so does not actually bound
    # how many distinct explanation jobs one user can create -- see SEC-3 in
    # the 2026-08-08 reliability fixes).
    xai_trigger_rate_limit_requests: int = Field(default=5, ge=1)
    xai_trigger_rate_limit_window_seconds: float = Field(default=60.0, gt=0.0)
    xai_retry_rate_limit_requests: int = Field(default=5, ge=1)
    xai_retry_rate_limit_window_seconds: float = Field(default=60.0, gt=0.0)

    # Operational maintenance for the XAI subsystem (see OPS-1/OPS-2 in the
    # 2026-08-08 reliability fixes).
    xai_artifact_cleanup_interval_seconds: float = Field(default=3600.0, gt=0.0)
    xai_stale_run_recovery_threshold_seconds: float = Field(default=300.0, gt=0.0)

    model_window_duration_seconds: float = Field(default=6.0, gt=0)
    model_window_overlap_seconds: float = Field(default=1.0, ge=0)
    audio_padding_policy: PaddingPolicy = "right_zero"
    audio_trim_policy: TrimPolicy = "none"
    preprocessing_version: str = "audio-preprocessing-v2"

    cnn_model_mode: ModelMode = "dummy"
    aasist_model_mode: ModelMode = "dummy"
    ssl_model_mode: ModelMode = "dummy"
    glottal_model_mode: ModelMode = "dummy"

    # Default points at the finalized CNN-V2 artifact (self-describing:
    # sample rate, LFCC/delta feature contract, and label map are all
    # embedded and validated at load time -- see
    # ``load_cnn_v2_checkpoint_strict`` in ``app/models/real/inference.py``).
    # The superseded checkpoint (``models/best_cnn_full_weighted.pth``) is
    # left on disk for research history and is never selected by this
    # default.
    cnn_model_path: str = Field(
        default="cnn/cnn_v2_lfcc_delta_aug_asvspoof2019_inference_best.pt",
        validation_alias=AliasChoices(
            "CNN_MODEL_PATH",
            "LFCC_CNN_TCN_MODEL_PATH",
            "cnn_model_path",
            "lfcc_cnn_tcn_model_path",
        ),
    )
    aasist_model_path: str = "aasist/aasist_light_v2_best.pt"
    # Default points at the finalized ASVspoof5 research artifact
    # (self-describing: xlsr_model_name/label_mapping/sample_rate/
    # max_duration_seconds are embedded and cross-checked at load time --
    # see ``_load_artifact_package`` in ``app/models/real/ssl_sequence_inference.py``).
    # The frozen XLS-R backbone is fetched separately (see
    # ``ssl_xlsr_model_name`` below). Two earlier checkpoints
    # (``models/xlsr_mamba_asvspoof2019_best.pt`` and
    # ``ssl/ssl_xlsr_mamba_generalization_v2_best.pt``, a later fine-tune of
    # this one) are left on disk for research history and are never selected
    # by this default.
    ssl_model_path: str = Field(
        default="ssl/ssl_asvspoof5_best.pt",
        validation_alias=AliasChoices(
            "SSL_MODEL_PATH",
            "SSL_SEQUENCE_MODEL_PATH",
            "ssl_model_path",
            "ssl_sequence_model_path",
        ),
    )
    # The 4.49 MB Generalization V2 artifact stores only the trainable head
    # (projection + Mamba blocks + classifier); it does not embed which
    # frozen Hugging Face backbone it was fine-tuned against. That name is
    # therefore trusted local configuration, matching the AASIST/CNN pattern
    # of declaring what the checkpoint itself cannot self-describe.
    ssl_xlsr_model_name: str = Field(
        default="facebook/wav2vec2-large-xlsr-53",
        validation_alias=AliasChoices("SSL_XLSR_MODEL_NAME", "ssl_xlsr_model_name"),
    )
    ssl_xlsr_local_files_only: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "SSL_XLSR_LOCAL_FILES_ONLY", "ssl_xlsr_local_files_only"
        ),
    )
    # Glottal's artifact is a frozen scikit-learn Pipeline (joblib), not a
    # PyTorch checkpoint -- ``.joblib`` is included in
    # ``model_supported_checkpoint_extensions`` below for exactly this branch.
    glottal_model_path: str = ""
    # Companion manifest declaring the exact 20 selected feature names (and
    # their order) the pipeline was fit on. Resolved relative to
    # MODEL_ROOT_DIR, same convention/escape checks as checkpoint paths --
    # see ``build_glottal_loader`` in app/models/real/glottal_inference.py.
    glottal_selected_features_path: str = "glottal/glottal_selected_features_v1.json"
    # Checkpoints live beside ``backend/`` in the repository's model-artifact
    # root.  ``app/models`` is Python source code and must never hold weights.
    model_root_dir: str = "../model_artifacts"
    model_supported_checkpoint_extensions: str = ".pt,.pth,.ckpt,.onnx,.safetensors,.joblib"
    model_device_policy: DevicePolicy = "cpu"
    model_default_device: DevicePolicy = "cpu"
    model_allow_cpu_fallback: bool = True
    model_precision: PrecisionPolicy = "float32"
    model_load_strategy: LoadStrategy = "lazy"
    model_load_timeout_seconds: float = Field(default=30, gt=0)
    model_load_retry_count: int = Field(default=0, ge=0)
    model_branch_timeout_seconds: float = Field(default=30, gt=0)
    lfcc_cnn_tcn_timeout_seconds: float | None = Field(default=None, gt=0)
    aasist_timeout_seconds: float | None = Field(default=None, gt=0)
    ssl_sequence_timeout_seconds: float | None = Field(default=None, gt=0)
    glottal_timeout_seconds: float | None = Field(default=None, gt=0)
    lfcc_cnn_tcn_device: DevicePolicy | None = None
    aasist_device: DevicePolicy | None = None
    ssl_sequence_device: DevicePolicy | None = None
    glottal_device: DevicePolicy | None = None
    required_model_branches: str = "lfcc_cnn_tcn,aasist,ssl_sequence,glottal"

    # ---------------------------------------------------------------- real models
    # LEGACY: the fields below (`cnn_feature_*`) describe the front end for
    # the SUPERSEDED CNN checkpoint (`models/best_cnn_full_weighted.pth`,
    # `app/models/architectures/cnn.py` / `app/models/preprocessing/spectral.py`),
    # which shipped without a training notebook, a feature configuration, or a
    # recorded label map. The live default (`cnn_model_path` above) now points
    # at CNN-V2, whose loader (`build_cnn_loader`, `app/models/real/inference.py`)
    # uses its own self-describing, hardcoded LFCC-40+delta+delta-delta front
    # end (`app/models/preprocessing/cnn_v2.py`) and ignores every field below
    # entirely. These fields are kept, unused by the live path, only so the
    # superseded checkpoint's own legacy tests/tooling keep working.
    # Chosen empirically for the superseded checkpoint: an `lfcc` /
    # `per_coefficient_zscore` / 40-filter configuration drove it into
    # saturation, returning spoof=1.000000 for every input tested -- a
    # detector that cannot detect. Sweeping 72 configurations, log-mel with
    # global standardisation and 20 filters both discriminated across all
    # probe signals and matched the input distribution recovered from that
    # checkpoint's BatchNorm statistics. That makes it the best-evidenced
    # default for the legacy checkpoint; it was never a verified one.
    cnn_feature_type: CnnFeatureType = "log_mel"
    cnn_feature_filters: int = Field(default=20, gt=0, le=512)
    cnn_feature_coefficients: int = Field(default=20, gt=0, le=512)
    cnn_feature_n_fft: int = Field(default=512, gt=0)
    cnn_feature_hop_length: int = Field(default=160, gt=0)
    cnn_feature_win_length: int = Field(default=400, gt=0)
    cnn_feature_fmin: float = Field(default=0.0, ge=0.0)
    cnn_feature_fmax: float | None = Field(default=None, gt=0.0)
    cnn_feature_include_deltas: bool = False
    cnn_feature_normalization: CnnFeatureNormalization = "global_zscore"
    cnn_feature_max_frames: int = Field(default=400, gt=0)

    aasist_target_samples: int = Field(default=64600, ge=64)
    aasist_waveform_normalization: AasistWaveformNormalization = "peak"
    aasist_length_policy: AasistLengthPolicy = "center_crop_pad"
    aasist_target_rms: float = Field(default=0.05, gt=0.0)

    # Which logit index means "spoof". Both conventions are widespread in
    # ASVspoof code and choosing wrong inverts every prediction, so this is
    # configuration, never an inference.
    cnn_class_order: ClassOrder = "bonafide_spoof"
    aasist_class_order: ClassOrder = "bonafide_spoof"
    # The superseded xlsr_mamba_asvspoof2019_best.pt artifact's own
    # label_mapping was {"bonafide": 0, "spoof": 1} -- i.e. bonafide_spoof.
    # The current Generalization V2 artifact does not embed a label_mapping
    # (trainable-head-only checkpoint); this convention is a declared
    # continuation of the same training pipeline, not re-derived from V2's
    # own file. See ``DECLARED_LABEL_MAPPING`` in ssl_sequence_inference.py.
    ssl_class_order: ClassOrder = "bonafide_spoof"

    ssl_target_samples: int = Field(default=96000, ge=64)

    # Operator attestations. Default false: until someone confirms these
    # against the training pipeline, real branches still run, but nothing they
    # produce is allowed to claim research validity.
    cnn_preprocessing_verified: bool = False
    cnn_class_mapping_verified: bool = False
    aasist_preprocessing_verified: bool = False
    aasist_class_mapping_verified: bool = False
    # The declared preprocessing (16kHz mono, 6s/96000-sample window,
    # zero-mean/peak normalisation) and label mapping (bonafide=0, spoof=1)
    # are supplied by the integration brief for the Generalization V2
    # artifact, not read from the checkpoint file itself -- it carries no
    # such metadata. The Mamba scan is also a from-scratch pure-PyTorch
    # reimplementation (mamba-ssm cannot install on a non-CUDA target; see
    # architectures/xlsr_mamba.py) that has not been cross-checked against
    # the documented Generalization V2 evaluation metrics on this runtime.
    # Flip to true only after that comparison run.
    ssl_preprocessing_verified: bool = False
    ssl_class_mapping_verified: bool = False
    # Glottal's preprocessing (exact DisVoice/parselmouth/spectral extraction
    # pipeline and selected-20 feature order) and label mapping
    # (bonafide=0/spoof=1) are both fully declared by
    # model_artifacts/glottal/glottal_selected_features_v1.json and the
    # feature extraction code itself -- unlike CNN/AASIST/SSL there is no
    # reverse-engineered guesswork here. Still defaults to false, matching the
    # same operator-attestation convention as every other real branch: it
    # never gates the primary fusion result either way (Glottal is auxiliary
    # evidence, excluded from FULL_SYSTEM_BRANCHES), but an unverified branch
    # must not claim `research_result: true` in its own metadata.
    glottal_preprocessing_verified: bool = False
    glottal_class_mapping_verified: bool = False

    fusion_method: FusionMethod = "weighted_average"
    fusion_decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    fusion_min_successful_branches: int = Field(default=2, ge=1, le=4)
    fusion_weight_lfcc_cnn_tcn: float = Field(default=0.25, ge=0.0)
    fusion_weight_aasist: float = Field(default=0.25, ge=0.0)
    fusion_weight_ssl_sequence: float = Field(default=0.25, ge=0.0)
    fusion_weight_glottal: float = Field(default=0.25, ge=0.0)
    fusion_config_version: str = "fusion-config-v1"
    fusion_required_branches: str = ""
    fusion_allow_equal_weight_fallback: bool = True
    # When a file exists at this path (resolved relative to
    # `resolved_model_root_dir`, same convention as checkpoint paths), it is
    # the authoritative source for the final research detector's fusion
    # method/threshold/branch set -- see `frozen_fusion_contract` below and
    # `app.utils.fusion.load_fusion_contract`. Blank disables contract
    # loading entirely (plain Settings-driven fusion, e.g. for tests that
    # construct a FusionEngine directly). A present-but-invalid contract file
    # fails startup rather than being silently ignored.
    fusion_contract_path: str = "fusion/final_detector_v1.json"

    # Fusion V3: constrained 4-branch convex weighted fusion
    # (CNN + AASIST + SSL + Glottal). "convex_4branch_v3" (default) makes
    # VoiceService try the frozen V3 contract below first, falling back to
    # the legacy 3-branch contract above only when V3 is unavailable for a
    # given request (see app.services.voice_service.VoiceService._fuse) and
    # LEGACY_FUSION_FALLBACK_ENABLED is true. "legacy_3branch" disables V3
    # outright -- the legacy detector becomes the deliberately configured
    # primary again, not a fallback.
    fusion_mode: FusionModeSetting = "convex_4branch_v3"
    # Resolved the same way as `fusion_contract_path` (relative to
    # `resolved_model_root_dir`). See `frozen_convex_fusion_v3_contract` and
    # `app.utils.fusion.load_convex_fusion_contract`. Blank disables V3
    # contract loading (V3 is then always unavailable, same effect as the
    # file not existing).
    fusion_v3_config_path: str = "fusion/fusion_convex_4branch_v3.json"
    # Cross-validated against the JSON contract when both are present and
    # `joblib` is importable (best-effort -- `joblib`/`scikit-learn` are only
    # guaranteed installed alongside the `glottal` extra). Blank skips the
    # joblib cross-check entirely; the JSON contract remains authoritative.
    fusion_v3_model_path: str = "fusion/fusion_convex_4branch_v3.joblib"
    # Cross-checked against the loaded V3 contract's own threshold the same
    # way FUSION_DECISION_THRESHOLD is cross-checked against the legacy
    # contract -- an explicit override that disagrees fails startup rather
    # than silently overriding the frozen research artifact.
    fusion_v3_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    legacy_fusion_fallback_enabled: bool = True

    allowed_origins: str = (
        "http://localhost:3000,"
        "http://127.0.0.1:3000"
    )

    @field_validator(
        "allowed_audio_extensions",
        "allowed_origins",
        "model_supported_checkpoint_extensions",
    )
    @classmethod
    def validate_comma_separated_values(cls, value: str) -> str:
        if not cls._split_csv(value):
            raise ValueError("At least one value is required.")
        return value

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {
            "development",
            "local",
            "test",
            "testing",
            "research",
            "production",
        }:
            raise ValueError("APP_ENV must be development, local, test, testing, research, or production.")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL is invalid.")
        return normalized

    @field_validator(
        "lfcc_cnn_tcn_timeout_seconds",
        "aasist_timeout_seconds",
        "ssl_sequence_timeout_seconds",
        "glottal_timeout_seconds",
        "lfcc_cnn_tcn_device",
        "aasist_device",
        "ssl_sequence_device",
        "glottal_device",
        "cnn_feature_fmax",
        mode="before",
    )
    @classmethod
    def blank_optional_settings_are_none(cls, value: object) -> object:
        """Treat blank optional dotenv values as unset rather than invalid strings."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("allowed_origins")
    @classmethod
    def validate_production_origins(cls, value: str, info) -> str:
        app_env = str(info.data.get("app_env", "development")).lower()
        origins = cls._split_csv(value)
        if app_env == "production" and "*" in origins:
            raise ValueError("Wildcard CORS origins are not allowed in production.")
        for origin in origins:
            if origin == "*":
                continue
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("ALLOWED_ORIGINS entries must be HTTP(S) origins.")
        return value

    @field_validator("api_key_hash_secret")
    @classmethod
    def validate_api_key_hash_secret(cls, value: str, info) -> str:
        app_env = str(info.data.get("app_env", "development")).lower()
        if app_env == "production" and value == "development-only-change-me":
            raise ValueError("API_KEY_HASH_SECRET must be configured in production.")
        if len(value.strip()) < 16:
            raise ValueError("API_KEY_HASH_SECRET must be at least 16 characters.")
        return value

    @model_validator(mode="after")
    def validate_phase2_settings(self) -> "Settings":
        production_like = self.app_env in {"production", "research"}
        if production_like and self.debug:
            raise ValueError("DEBUG must be false in production/research.")
        if self.dev_auth_bypass:
            if production_like:
                raise ValueError(
                    "DEV_AUTH_BYPASS cannot be enabled in production/research."
                )
            if self.app_env not in {"development", "local", "test", "testing"}:
                raise ValueError("DEV_AUTH_BYPASS requires a local development environment.")
        if self.xai_enabled and self.xai_mode == "disabled":
            raise ValueError("XAI_MODE cannot be disabled when XAI_ENABLED is true.")
        if self.xai_narrative_enabled:
            if not self.xai_enabled:
                raise ValueError("XAI_NARRATIVE_ENABLED requires XAI_ENABLED.")
            if not self.xai_narrative_api_key.strip():
                raise ValueError(
                    "XAI_NARRATIVE_API_KEY is required when XAI_NARRATIVE_ENABLED is true."
                )
            if not self.xai_narrative_model.strip():
                raise ValueError(
                    "XAI_NARRATIVE_MODEL is required when XAI_NARRATIVE_ENABLED is true."
                )
            parsed_narrative_url = urlparse(self.xai_narrative_base_url)
            if (
                parsed_narrative_url.scheme != "https"
                or not parsed_narrative_url.netloc
                or not parsed_narrative_url.path.rstrip("/").endswith(
                    "/compatible-mode/v1"
                )
            ):
                raise ValueError(
                    "XAI_NARRATIVE_BASE_URL must be an HTTPS Model Studio "
                    "OpenAI-compatible base URL ending in /compatible-mode/v1."
                )
        if self.xai_enabled and self.xai_mode == "real":
            if not self.xai_semantic_manifest_path.strip():
                raise ValueError("XAI_SEMANTIC_MANIFEST_PATH is required for XAI_MODE=real.")
            for field_name, value in {
                "XAI_SEMANTIC_MODEL_PATH": self.xai_semantic_model_path,
                "XAI_SEMANTIC_FEATURE_COLUMNS_PATH": self.xai_semantic_feature_columns_path,
                "XAI_SEMANTIC_THRESHOLD_PATH": self.xai_semantic_threshold_path,
                "XAI_SEMANTIC_TRAINING_METADATA_PATH": self.xai_semantic_training_metadata_path,
                "XAI_SEMANTIC_IMPUTER_PATH": self.xai_semantic_imputer_path,
            }.items():
                if not value.strip():
                    raise ValueError(f"{field_name} is required for XAI_MODE=real.")
        if not self.xai_pipeline_version.strip():
            raise ValueError("XAI_PIPELINE_VERSION must be configured.")
        if not self.xai_artifact_root.strip():
            raise ValueError("XAI_ARTIFACT_ROOT must be configured.")
        if production_like and not self.mongodb_required:
            raise ValueError("MONGODB_REQUIRED cannot be false in production/research.")
        if production_like and not self.mongodb_uri.strip():
            raise ValueError("MONGODB_URI is required when MongoDB is required.")
        if production_like and not self.mongodb_database.strip():
            raise ValueError("MONGODB_DATABASE is required when MongoDB is required.")
        if production_like and self.storage_policy != "required":
            raise ValueError("STORAGE_POLICY must be required in production/research.")
        if self.storage_policy == "required":
            missing_cloudinary = [
                name
                for name, value in {
                    "CLOUDINARY_CLOUD_NAME": self.cloudinary_cloud_name,
                    "CLOUDINARY_API_KEY": self.cloudinary_api_key,
                    "CLOUDINARY_API_SECRET": self.cloudinary_api_secret,
                }.items()
                if not value.strip()
            ]
            if production_like and missing_cloudinary:
                raise ValueError(f"{missing_cloudinary[0]} is required when storage is required.")
        if production_like:
            if not self.clerk_issuer.strip():
                raise ValueError("CLERK_ISSUER is required in production/research.")
            if not self.clerk_jwks_url.strip():
                raise ValueError("CLERK_JWKS_URL is required in production/research.")
            if urlparse(self.clerk_jwks_url).scheme != "https":
                raise ValueError("CLERK_JWKS_URL must use HTTPS in production/research.")
            if not self.clerk_audience_list:
                raise ValueError("CLERK_AUDIENCE is required in production/research.")
            if not self.clerk_authorized_party_list:
                raise ValueError("CLERK_AUTHORIZED_PARTIES is required in production/research.")
        if self.max_request_body_mb <= self.max_upload_size_mb:
            raise ValueError("MAX_REQUEST_BODY_MB must be greater than MAX_UPLOAD_SIZE_MB.")
        if self.max_audio_duration_seconds <= self.min_audio_duration_seconds:
            raise ValueError("MAX_AUDIO_DURATION_SECONDS must be greater than MIN_AUDIO_DURATION_SECONDS.")
        if self.max_input_sample_rate < self.target_sample_rate:
            raise ValueError("MAX_INPUT_SAMPLE_RATE must be greater than or equal to TARGET_SAMPLE_RATE.")
        if self.model_window_overlap_seconds >= self.model_window_duration_seconds:
            raise ValueError("MODEL_WINDOW_OVERLAP_SECONDS must be smaller than MODEL_WINDOW_DURATION_SECONDS.")
        if (
            self.xai_temporal_window_stride_seconds
            > self.ssl_waveform_config.target_duration_seconds
        ):
            raise ValueError(
                "XAI_TEMPORAL_WINDOW_STRIDE_SECONDS must not exceed the SSL "
                "model window duration."
            )
        if self.cnn_feature_win_length > self.cnn_feature_n_fft:
            raise ValueError("CNN_FEATURE_WIN_LENGTH must not exceed CNN_FEATURE_N_FFT.")
        if (
            self.cnn_feature_type in {"mfcc", "lfcc"}
            and self.cnn_feature_coefficients > self.cnn_feature_filters
        ):
            raise ValueError(
                "CNN_FEATURE_COEFFICIENTS must not exceed CNN_FEATURE_FILTERS."
            )
        if (
            self.cnn_feature_fmax is not None
            and self.cnn_feature_fmax <= self.cnn_feature_fmin
        ):
            raise ValueError("CNN_FEATURE_FMAX must be greater than CNN_FEATURE_FMIN.")
        # Unverified real branches are a development state. Letting them run in a
        # production/research deployment would put unvalidated model output in
        # front of users under a research banner.
        for branch, mode in (
            ("lfcc_cnn_tcn", self.cnn_model_mode),
            ("aasist", self.aasist_model_mode),
            ("ssl_sequence", self.ssl_model_mode),
            ("glottal", self.glottal_model_mode),
        ):
            if (
                production_like
                and mode == "real"
                and not self.branch_verification(branch)["verified"]
            ):
                raise ValueError(
                    f"Branch '{branch}' cannot run in real mode in "
                    "production/research until its preprocessing and class "
                    "mapping are verified."
                )
        if (
            self.xai_semantic_window_overlap_seconds
            >= self.xai_semantic_window_duration_seconds
        ):
            raise ValueError(
                "XAI_SEMANTIC_WINDOW_OVERLAP_SECONDS must be smaller than "
                "XAI_SEMANTIC_WINDOW_DURATION_SECONDS."
            )
        step_seconds = self.model_window_duration_seconds - self.model_window_overlap_seconds
        max_segments = int((self.max_audio_duration_seconds / step_seconds) + 2)
        if max_segments > 500:
            raise ValueError("Windowing configuration creates too many segments.")
        semantic_step_seconds = (
            self.xai_semantic_window_duration_seconds
            - self.xai_semantic_window_overlap_seconds
        )
        max_semantic_windows = int(
            (self.max_audio_duration_seconds / semantic_step_seconds) + 2
        )
        if max_semantic_windows > 500:
            raise ValueError("Semantic windowing configuration creates too many windows.")
        if "/" in self.cloudinary_audio_folder.strip("\\"):
            folder_parts = [part for part in self.cloudinary_audio_folder.split("/") if part]
        else:
            folder_parts = [self.cloudinary_audio_folder]
        if any(part in {".", ".."} for part in folder_parts):
            raise ValueError("CLOUDINARY_AUDIO_FOLDER is invalid.")
        if self.trusted_host_list:
            for host in self.trusted_host_list:
                if "/" in host or "\\" in host:
                    raise ValueError("TRUSTED_HOSTS entries must be hostnames, not URLs.")
        for name, value in {
            "MODEL_LOAD_TIMEOUT_SECONDS": self.model_load_timeout_seconds,
            "MODEL_BRANCH_TIMEOUT_SECONDS": self.model_branch_timeout_seconds,
            "FUSION_DECISION_THRESHOLD": self.fusion_decision_threshold,
            "FUSION_WEIGHT_LFCC_CNN_TCN": self.fusion_weight_lfcc_cnn_tcn,
            "FUSION_WEIGHT_AASIST": self.fusion_weight_aasist,
            "FUSION_WEIGHT_SSL_SEQUENCE": self.fusion_weight_ssl_sequence,
            "FUSION_WEIGHT_GLOTTAL": self.fusion_weight_glottal,
            "FUSION_V3_THRESHOLD": self.fusion_v3_threshold,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not self.fusion_config_version.strip():
            raise ValueError("FUSION_CONFIG_VERSION must be configured.")
        if production_like and self.fusion_config_version.startswith("development"):
            raise ValueError("Development fusion config versions are not allowed in production/research.")
        if self.fusion_method == "weighted_average" and not any(
            weight > 0
            for weight in self.fusion_weight_map.values()
        ):
            raise ValueError("At least one fusion weight must be positive.")
        # A present frozen fusion contract (model_artifacts/fusion/final_detector_v1.json
        # by default) is the authoritative source for method/threshold once
        # loaded (see FusionEngine.from_settings). An *explicit* env/kwarg
        # override that disagrees with it is exactly the "bad configuration
        # silently overriding the research contract" failure mode this guard
        # exists to catch -- fail startup instead of picking one silently.
        # `frozen_fusion_contract` itself raises FusionContractError (a
        # ValueError subclass) for a present-but-invalid file, which
        # pydantic-settings surfaces the same way as any other validator error.
        contract = self.frozen_fusion_contract
        if contract is not None:
            if (
                "fusion_decision_threshold" in self.model_fields_set
                and not math.isclose(
                    self.fusion_decision_threshold,
                    contract.decision_threshold,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "FUSION_DECISION_THRESHOLD "
                    f"({self.fusion_decision_threshold}) conflicts with the "
                    f"frozen research contract at {contract.source_path} "
                    f"({contract.decision_threshold}). Remove the override or "
                    "update the contract file -- never let them silently disagree."
                )
            if (
                "fusion_method" in self.model_fields_set
                and self.fusion_method != contract.fusion_method
            ):
                raise ValueError(
                    f"FUSION_METHOD ({self.fusion_method!r}) conflicts with the "
                    f"frozen research contract at {contract.source_path} "
                    f"({contract.fusion_method!r})."
                )
        # Same "explicit override disagreeing with the frozen artifact fails
        # startup" guard as above, for the Fusion V3 contract.
        v3_contract = self.frozen_convex_fusion_v3_contract
        if (
            v3_contract is not None
            and "fusion_v3_threshold" in self.model_fields_set
            and not math.isclose(
                self.fusion_v3_threshold,
                v3_contract.threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"FUSION_V3_THRESHOLD ({self.fusion_v3_threshold}) conflicts with "
                f"the frozen Fusion V3 contract at {v3_contract.source_json_path} "
                f"({v3_contract.threshold}). Remove the override or update the "
                "contract file -- never let them silently disagree."
            )
        if self.fusion_min_successful_branches > len(self.required_model_branch_list):
            raise ValueError("FUSION_MIN_SUCCESSFUL_BRANCHES cannot exceed required branch count.")
        # Validate branch CSV fields early so startup/readiness cannot drift.
        self.required_model_branch_list
        self.fusion_required_branch_list
        return self

    @property
    def allowed_audio_extension_list(self) -> list[str]:
        return [
            extension.lower().lstrip(".")
            for extension in self._split_csv(self.allowed_audio_extensions)
        ]

    @property
    def supported_checkpoint_extension_list(self) -> list[str]:
        extensions = []
        for extension in self._split_csv(self.model_supported_checkpoint_extensions):
            normalized = extension.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            extensions.append(normalized)
        return extensions

    @property
    def allowed_origin_list(self) -> list[str]:
        return self._split_csv(self.allowed_origins)

    @property
    def trusted_host_list(self) -> list[str]:
        return self._split_csv(self.trusted_hosts)

    @property
    def clerk_audience_list(self) -> list[str]:
        return self._split_csv(self.clerk_audience)

    @property
    def clerk_authorized_party_list(self) -> list[str]:
        return self._split_csv(self.clerk_authorized_parties)

    @property
    def backend_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def resolved_upload_dir(self) -> Path:
        upload_path = Path(self.upload_dir)
        if upload_path.is_absolute():
            return upload_path
        return self.backend_root / upload_path

    @property
    def resolved_model_root_dir(self) -> Path:
        model_root = Path(self.model_root_dir)
        if model_root.is_absolute():
            return model_root.expanduser().resolve()
        return (self.backend_root / model_root).resolve()

    def resolve_xai_artifact_path(self, configured_path: str) -> Path:
        artifact_path = Path(configured_path)
        if artifact_path.is_absolute():
            return artifact_path
        return self.backend_root / artifact_path

    @property
    def cnn_feature_config(self) -> "SpectralFeatureConfig":
        from app.models.preprocessing.spectral import SpectralFeatureConfig

        return SpectralFeatureConfig(
            feature_type=self.cnn_feature_type,
            sample_rate=self.target_sample_rate,
            n_filters=self.cnn_feature_filters,
            n_coefficients=self.cnn_feature_coefficients,
            n_fft=self.cnn_feature_n_fft,
            hop_length=self.cnn_feature_hop_length,
            win_length=self.cnn_feature_win_length,
            fmin=self.cnn_feature_fmin,
            fmax=self.cnn_feature_fmax,
            include_deltas=self.cnn_feature_include_deltas,
            normalization=self.cnn_feature_normalization,
            max_frames=self.cnn_feature_max_frames,
        )

    @property
    def aasist_waveform_config(self) -> "WaveformConfig":
        from app.models.preprocessing.waveform import WaveformConfig

        return WaveformConfig(
            sample_rate=self.target_sample_rate,
            target_samples=self.aasist_target_samples,
            normalization=self.aasist_waveform_normalization,
            length_policy=self.aasist_length_policy,
            target_rms=self.aasist_target_rms,
        )

    @property
    def ssl_waveform_config(self) -> "SSLWaveformConfig":
        from app.models.preprocessing.ssl_waveform import SSLWaveformConfig

        return SSLWaveformConfig(
            sample_rate=self.target_sample_rate,
            target_samples=self.ssl_target_samples,
        )

    def branch_class_order(self, branch_name: str) -> str:
        return {
            "lfcc_cnn_tcn": self.cnn_class_order,
            "aasist": self.aasist_class_order,
            "ssl_sequence": self.ssl_class_order,
        }.get(branch_name, "bonafide_spoof")

    def branch_verification(self, branch_name: str) -> dict[str, bool]:
        """Operator attestations for a real branch.

        ``verified`` requires BOTH the feature pipeline and the label map to be
        confirmed. Either one being wrong invalidates the output, so this is an
        AND, and it defaults to false.
        """

        flags = {
            "lfcc_cnn_tcn": (
                self.cnn_preprocessing_verified,
                self.cnn_class_mapping_verified,
            ),
            "aasist": (
                self.aasist_preprocessing_verified,
                self.aasist_class_mapping_verified,
            ),
            "ssl_sequence": (
                self.ssl_preprocessing_verified,
                self.ssl_class_mapping_verified,
            ),
            "glottal": (
                self.glottal_preprocessing_verified,
                self.glottal_class_mapping_verified,
            ),
        }.get(branch_name, (False, False))
        return {
            "preprocessing_verified": flags[0],
            "class_mapping_verified": flags[1],
            "verified": flags[0] and flags[1],
        }

    @property
    def frozen_fusion_contract(self):
        """The final research detector's frozen fusion contract, if present.

        Lazily imported to avoid a circular import (``app.utils.fusion``
        already imports ``Settings`` for ``FusionEngine.from_settings``).
        Returns ``None`` when ``fusion_contract_path`` is blank or the file
        does not exist; raises ``FusionContractError`` for a present-but-invalid
        file rather than silently falling back.
        """

        if not self.fusion_contract_path.strip():
            return None
        from app.utils.fusion import load_fusion_contract

        path = self.resolved_model_root_dir / self.fusion_contract_path
        return load_fusion_contract(path)

    @property
    def frozen_convex_fusion_v3_contract(self):
        """The Fusion V3 (constrained 4-branch convex weighted fusion)
        contract, if present.

        Same conventions as `frozen_fusion_contract`: lazily imported,
        ``None`` when `fusion_v3_config_path` is blank or the file does not
        exist, raises ``ConvexFusionContractError`` for a present-but-invalid
        file.
        """

        if not self.fusion_v3_config_path.strip():
            return None
        from app.utils.fusion import load_convex_fusion_contract

        json_path = self.resolved_model_root_dir / self.fusion_v3_config_path
        joblib_path = (
            self.resolved_model_root_dir / self.fusion_v3_model_path
            if self.fusion_v3_model_path.strip()
            else None
        )
        return load_convex_fusion_contract(json_path, joblib_path)

    @property
    def fusion_weight_map(self) -> dict[str, float]:
        return {
            "lfcc_cnn_tcn": self.fusion_weight_lfcc_cnn_tcn,
            "aasist": self.fusion_weight_aasist,
            "ssl_sequence": self.fusion_weight_ssl_sequence,
            "glottal": self.fusion_weight_glottal,
        }

    @property
    def required_model_branch_list(self) -> list[str]:
        return self._canonical_branch_list(self.required_model_branches)

    @property
    def fusion_required_branch_list(self) -> list[str]:
        return self._canonical_branch_list(self.fusion_required_branches)

    @property
    def max_audio_sample_rate(self) -> int:
        """Backward-compatible name for the input sample-rate limit."""

        return self.max_input_sample_rate

    @property
    def prediction_job_concurrency(self) -> int:
        """Backward-compatible name for the prediction concurrency limit."""

        return self.max_concurrent_predictions

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @classmethod
    def _canonical_branch_list(cls, value: str) -> list[str]:
        aliases = {
            "lfcc_cnn_tcn": "lfcc_cnn_tcn",
            "lfcc": "lfcc_cnn_tcn",
            "cnn": "lfcc_cnn_tcn",
            "cnn_acoustic": "lfcc_cnn_tcn",
            "aasist": "aasist",
            "ssl_sequence": "ssl_sequence",
            "ssl": "ssl_sequence",
            "ssl_wavlm_xlsr": "ssl_sequence",
            "glottal": "glottal",
            "glottal_features": "glottal",
        }
        branches: list[str] = []
        seen = set()
        for raw_branch in cls._split_csv(value):
            normalized = raw_branch.lower()
            if normalized not in aliases:
                raise ValueError(f"Unknown model branch: {raw_branch}")
            branch = aliases[normalized]
            if branch in seen:
                raise ValueError(f"Duplicate model branch: {branch}")
            branches.append(branch)
            seen.add(branch)
        return branches


settings = Settings(
    _env_file=None
    if os.getenv("MULTISCOPE_DISABLE_DOTENV") == "1"
    else Settings.model_config.get("env_file")
)
