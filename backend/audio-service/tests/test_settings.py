from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_settings_defaults_are_safe_for_development() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "MULTI-SCOPE Voice Classification API"
    assert settings.app_version == "0.1.0"
    assert settings.app_env == "development"
    assert settings.debug is True
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.log_level == "INFO"
    assert settings.max_request_body_mb == 27
    assert settings.max_upload_size_mb == 25
    assert settings.allowed_audio_extension_list == [
        "wav",
        "flac",
        "mp3",
        "m4a",
        "aac",
        "opus",
        "ogg",
        "webm",
    ]
    assert settings.max_audio_duration_seconds == 180
    assert settings.min_audio_duration_seconds == 1.0
    assert settings.max_audio_channels == 2
    assert settings.max_input_sample_rate == 96000
    assert settings.max_audio_sample_rate == 96000
    assert settings.max_decoded_audio_size_mb == 32
    assert settings.mongodb_uri == ""
    assert settings.mongodb_database == "multiscope"
    assert settings.mongodb_required is True
    assert settings.mongodb_connect_timeout_ms == 5000
    assert settings.mongodb_server_selection_timeout_ms == 5000
    assert settings.mongodb_socket_timeout_ms == 20000
    assert settings.mongodb_retry_reads is True
    assert settings.mongodb_retry_writes is True
    assert settings.clerk_issuer == ""
    assert settings.clerk_jwks_url == ""
    assert settings.clerk_audience == ""
    assert settings.clerk_authorized_parties == ""
    assert settings.clerk_jwks_cache_seconds == 3600
    assert settings.clerk_jwks_min_refresh_interval_seconds == 60
    assert settings.clerk_jwks_negative_kid_cache_seconds == 60
    assert settings.dev_auth_bypass is False
    assert settings.api_key_hash_secret == "development-only-change-me"
    assert settings.cloudinary_cloud_name == ""
    assert settings.cloudinary_api_key == ""
    assert settings.cloudinary_api_secret == ""
    assert settings.cloudinary_audio_folder == "multiscope/audio"
    assert settings.cloudinary_storage_enabled is True
    assert settings.storage_policy == "required"
    assert settings.cloudinary_sdk_timeout_seconds == 60
    assert settings.cloudinary_upload_timeout_seconds == 60
    assert settings.prediction_job_timeout_seconds == 180
    assert settings.max_concurrent_predictions == 2
    assert settings.prediction_job_concurrency == 2
    assert settings.enable_legacy_anonymous_prediction is False
    assert settings.docs_enabled is True
    assert settings.openapi_enabled is True
    assert settings.model_health_enabled is True
    assert settings.trusted_host_list == ["localhost", "127.0.0.1", "testserver"]
    assert settings.max_queued_predictions == 4
    assert settings.max_active_predictions_per_principal == 1
    assert settings.rate_limit_window_seconds == 60
    assert settings.rate_limit_requests_per_window == 120
    assert settings.prediction_rate_limit_per_window == 20
    assert settings.api_key_creation_rate_limit_per_window == 10
    assert settings.preprocessing_version == "audio-preprocessing-v2"
    assert settings.model_window_duration_seconds == 6.0
    assert settings.model_window_overlap_seconds == 1.0
    assert settings.xai_semantic_window_duration_seconds == 1.0
    assert settings.xai_semantic_window_overlap_seconds == 0.5
    assert settings.xai_semantic_model_path.endswith("xgboost_surrogate_v4.json")
    assert settings.xai_semantic_feature_columns_path.endswith("feature_columns_v4.json")
    assert settings.xai_semantic_imputer_path.endswith("imputer_v4.joblib")
    assert settings.audio_padding_policy == "right_zero"
    assert settings.audio_trim_policy == "none"
    assert settings.allowed_origin_list == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    assert settings.resolved_upload_dir == settings.backend_root / "uploads"


def test_settings_parse_comma_separated_values() -> None:
    settings = Settings(
        _env_file=None,
        allowed_audio_extensions=" wav, .flac,mp3 ",
        allowed_origins=" http://localhost:3000, http://127.0.0.1:3000 ",
    )

    assert settings.allowed_audio_extension_list == ["wav", "flac", "mp3"]
    assert settings.allowed_origin_list == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


def test_settings_treat_blank_optional_dotenv_values_as_unset(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LFCC_CNN_TCN_TIMEOUT_SECONDS=",
                "AASIST_TIMEOUT_SECONDS=",
                "SSL_SEQUENCE_TIMEOUT_SECONDS=",
                "GLOTTAL_TIMEOUT_SECONDS=",
                "LFCC_CNN_TCN_DEVICE=",
                "AASIST_DEVICE=",
                "SSL_SEQUENCE_DEVICE=",
                "GLOTTAL_DEVICE=",
                "CNN_FEATURE_FMAX=",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.lfcc_cnn_tcn_timeout_seconds is None
    assert settings.aasist_timeout_seconds is None
    assert settings.ssl_sequence_timeout_seconds is None
    assert settings.glottal_timeout_seconds is None
    assert settings.lfcc_cnn_tcn_device is None
    assert settings.aasist_device is None
    assert settings.ssl_sequence_device is None
    assert settings.glottal_device is None
    assert settings.cnn_feature_fmax is None


def test_settings_reject_dev_auth_bypass_in_research_or_production() -> None:
    with pytest.raises(ValidationError, match="DEV_AUTH_BYPASS cannot be enabled"):
        Settings(
            _env_file=None,
            app_env="research",
            debug=False,
            dev_auth_bypass=True,
        )


def test_settings_load_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_ENV=testing",
                "DEBUG=false",
                "PORT=9000",
                "LOG_LEVEL=DEBUG",
                "MAX_REQUEST_BODY_MB=12",
                "MAX_UPLOAD_SIZE_MB=10",
                "MAX_INPUT_SAMPLE_RATE=48000",
                "MONGODB_URI=mongodb+srv://example.invalid",
                "MONGODB_DATABASE=multiscope_test",
                "MONGODB_REQUIRED=false",
                "MONGODB_CONNECT_TIMEOUT_MS=1000",
                "MONGODB_SERVER_SELECTION_TIMEOUT_MS=2000",
                "MONGODB_SOCKET_TIMEOUT_MS=3000",
                "MONGODB_RETRY_READS=false",
                "MONGODB_RETRY_WRITES=false",
                "CLERK_ISSUER=https://example.clerk.accounts.dev",
                "CLERK_JWKS_URL=https://example.clerk.accounts.dev/.well-known/jwks.json",
                "CLERK_AUDIENCE=multi-scope",
                "CLERK_AUTHORIZED_PARTIES=http://localhost:3000",
                "CLERK_JWKS_CACHE_SECONDS=600",
                "CLERK_JWKS_MIN_REFRESH_INTERVAL_SECONDS=30",
                "CLERK_JWKS_NEGATIVE_KID_CACHE_SECONDS=45",
                "DEV_AUTH_BYPASS=true",
                "API_KEY_HASH_SECRET=test-secret-value",
                "CLOUDINARY_CLOUD_NAME=demo-cloud",
                "CLOUDINARY_API_KEY=demo-key",
                "CLOUDINARY_API_SECRET=demo-secret",
                "CLOUDINARY_AUDIO_FOLDER=multiscope/test-audio",
                "CLOUDINARY_STORAGE_ENABLED=false",
                "CLOUDINARY_SDK_TIMEOUT_SECONDS=8",
                "CLOUDINARY_UPLOAD_TIMEOUT_SECONDS=10",
                "PREDICTION_JOB_TIMEOUT_SECONDS=60",
                "MAX_CONCURRENT_PREDICTIONS=4",
                "ENABLE_LEGACY_ANONYMOUS_PREDICTION=true",
                "CNN_MODEL_MODE=real",
                "ALLOWED_AUDIO_EXTENSIONS=wav,aac",
                "ALLOWED_ORIGINS=http://localhost:3000",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.app_env == "testing"
    assert settings.debug is False
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.max_request_body_mb == 12
    assert settings.max_upload_size_mb == 10
    assert settings.max_input_sample_rate == 48000
    assert settings.mongodb_uri == "mongodb+srv://example.invalid"
    assert settings.mongodb_database == "multiscope_test"
    assert settings.mongodb_required is False
    assert settings.mongodb_connect_timeout_ms == 1000
    assert settings.mongodb_server_selection_timeout_ms == 2000
    assert settings.mongodb_socket_timeout_ms == 3000
    assert settings.mongodb_retry_reads is False
    assert settings.mongodb_retry_writes is False
    assert settings.clerk_issuer == "https://example.clerk.accounts.dev"
    assert settings.clerk_jwks_url.endswith("/.well-known/jwks.json")
    assert settings.clerk_audience_list == ["multi-scope"]
    assert settings.clerk_authorized_party_list == ["http://localhost:3000"]
    assert settings.clerk_jwks_cache_seconds == 600
    assert settings.clerk_jwks_min_refresh_interval_seconds == 30
    assert settings.clerk_jwks_negative_kid_cache_seconds == 45
    assert settings.dev_auth_bypass is True
    assert settings.api_key_hash_secret == "test-secret-value"
    assert settings.cloudinary_cloud_name == "demo-cloud"
    assert settings.cloudinary_api_key == "demo-key"
    assert settings.cloudinary_api_secret == "demo-secret"
    assert settings.cloudinary_audio_folder == "multiscope/test-audio"
    assert settings.cloudinary_storage_enabled is False
    assert settings.cloudinary_sdk_timeout_seconds == 8
    assert settings.cloudinary_upload_timeout_seconds == 10
    assert settings.prediction_job_timeout_seconds == 60
    assert settings.max_concurrent_predictions == 4
    assert settings.prediction_job_concurrency == 4
    assert settings.enable_legacy_anonymous_prediction is True
    assert settings.cnn_model_mode == "real"
    assert settings.allowed_audio_extension_list == ["wav", "aac"]
    assert settings.allowed_origin_list == ["http://localhost:3000"]


def test_settings_reject_invalid_model_mode() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cnn_model_mode="experimental")


def test_settings_require_semantic_artifacts_for_real_xai_mode() -> None:
    with pytest.raises(ValidationError, match="XAI_SEMANTIC_MODEL_PATH"):
        Settings(
            _env_file=None,
            xai_enabled=True,
            xai_mode="real",
            xai_semantic_model_path="",
        )

    with pytest.raises(ValidationError, match="XAI_SEMANTIC_FEATURE_COLUMNS_PATH"):
        Settings(
            _env_file=None,
            xai_enabled=True,
            xai_mode="real",
            xai_semantic_feature_columns_path="",
        )

    with pytest.raises(ValidationError, match="XAI_SEMANTIC_IMPUTER_PATH"):
        Settings(
            _env_file=None,
            xai_enabled=True,
            xai_mode="real",
            xai_semantic_imputer_path="",
        )


def test_settings_reject_wildcard_origin_in_production() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production", allowed_origins="*")


def test_settings_require_api_key_hash_secret_in_production() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production")


def test_settings_allow_production_with_required_external_configuration() -> None:
    settings = production_settings()

    assert settings.app_env == "production"
    assert settings.debug is False
    assert settings.storage_policy == "required"
    assert settings.allowed_origin_list == ["https://dashboard.example.edu"]


@pytest.mark.parametrize("app_env", ["production", "research"])
def test_settings_reject_non_required_storage_policy_outside_development(
    app_env,
) -> None:
    with pytest.raises(ValidationError):
        production_settings(app_env=app_env, storage_policy="optional")


def test_settings_reject_debug_in_production() -> None:
    with pytest.raises(ValidationError):
        production_settings(debug=True)


def test_settings_reject_insecure_clerk_jwks_url_in_production() -> None:
    with pytest.raises(ValidationError):
        production_settings(clerk_jwks_url="http://issuer.example.edu/jwks.json")


def test_settings_reject_invalid_trusted_host() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, trusted_hosts="localhost,https://bad-host.example")


def test_settings_reject_invalid_audio_window_overlap() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            model_window_duration_seconds=6.0,
            model_window_overlap_seconds=6.0,
        )


def test_settings_reject_invalid_semantic_window_overlap() -> None:
    with pytest.raises(ValidationError, match="XAI_SEMANTIC_WINDOW_OVERLAP_SECONDS"):
        Settings(
            _env_file=None,
            xai_semantic_window_duration_seconds=1.0,
            xai_semantic_window_overlap_seconds=1.0,
        )


def test_settings_resolve_absolute_upload_dir(tmp_path: Path) -> None:
    upload_dir = tmp_path / "multi-scope-uploads"
    settings = Settings(_env_file=None, upload_dir=str(upload_dir))

    assert settings.resolved_upload_dir == upload_dir


def test_default_model_root_is_the_repository_artifact_directory() -> None:
    settings = Settings(_env_file=None)

    assert settings.resolved_model_root_dir == (
        Path(__file__).resolve().parents[2] / "model_artifacts"
    )


def production_settings(**overrides) -> Settings:
    values = {
        "app_env": "production",
        "debug": False,
        "allowed_origins": "https://dashboard.example.edu",
        "mongodb_uri": "mongodb://mongo.example.edu:27017",
        "mongodb_database": "multiscope",
        "api_key_hash_secret": "test-production-secret",
        "clerk_issuer": "https://issuer.example.edu",
        "clerk_jwks_url": "https://issuer.example.edu/.well-known/jwks.json",
        "clerk_audience": "multi-scope-api",
        "clerk_authorized_parties": "https://dashboard.example.edu",
        "cloudinary_cloud_name": "demo-cloud",
        "cloudinary_api_key": "demo-key",
        "cloudinary_api_secret": "demo-secret",
        "storage_policy": "required",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)
