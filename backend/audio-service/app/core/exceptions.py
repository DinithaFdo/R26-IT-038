class MultiScopeError(Exception):
    """Base exception for application-level errors."""


class AuthenticationError(MultiScopeError):
    """Raised when authentication fails.

    ``category`` is an internal-only, safe-to-log failure reason (e.g.
    ``missing_jwks_configuration``, ``invalid_signature``, ``expired_token``).
    It is never sent to the client -- every authentication failure still
    returns the same generic 401 body -- but letting the server log distinguish
    "Clerk is misconfigured" from "this specific token is bad" is the
    difference between a five-minute fix and a long debugging session. Defaults
    to ``"unspecified"`` so any call site that does not pass one still works.
    """

    def __init__(self, message: str = "Authentication failed.", *, category: str = "unspecified") -> None:
        super().__init__(message)
        self.category = category


class AdmissionControlError(MultiScopeError):
    """Raised when the service rejects work before execution starts."""

    status_code = 429
    error_code = "admission_control_rejected"
    public_message = "Request cannot be accepted right now."
    retry_after_seconds: int | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message or self.public_message)
        self.retry_after_seconds = (
            retry_after_seconds
            if retry_after_seconds is not None
            else self.retry_after_seconds
        )


class PredictionRunnerShuttingDownError(AdmissionControlError):
    """Raised when predictions are rejected during service shutdown."""

    status_code = 503
    error_code = "prediction_runner_shutting_down"
    public_message = "Prediction service is shutting down."
    retry_after_seconds = 5


class PredictionQueueFullError(AdmissionControlError):
    """Raised when the in-process prediction backlog is full."""

    error_code = "prediction_queue_full"
    public_message = "Prediction queue is full."
    retry_after_seconds = 5


class PrincipalPredictionLimitExceededError(AdmissionControlError):
    """Raised when one principal has too many active/queued predictions."""

    error_code = "principal_prediction_limit_exceeded"
    public_message = "Principal prediction concurrency limit reached."
    retry_after_seconds = 5


class IdempotencyConflictError(MultiScopeError):
    """Raised when one idempotency key is reused for a different logical request.

    Reusing a key for a *different* request is a client bug, not a retry, so it
    is rejected with a dedicated 409 error code instead of a generic
    ``http_error``. ``details`` stays ``None`` outside local development: it is
    only ever populated with the caller's own request-shaping fields so a
    developer can see which part of the logical request changed.
    """

    status_code = 409
    error_code = "idempotency_conflict"
    public_message = (
        "Idempotency key was already used for a different request. Use a new "
        "unique idempotency_key for a new submission, or resend the exact "
        "original request to replay its result."
    )

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict | None = None,
    ) -> None:
        super().__init__(message or self.public_message)
        self.details = details


class ModelNotIntegratedError(MultiScopeError):
    """Raised when a requested model branch has not been integrated yet."""


class ModelLoadError(MultiScopeError):
    """Raised when a model branch cannot be loaded."""

    def __init__(
        self,
        message: str = "Model load failed.",
        *,
        public_message: str = "Model load failed.",
        error_code: str = "model_load_failed",
    ) -> None:
        super().__init__(message)
        self.public_message = public_message
        self.error_code = error_code


class ModelInferenceError(MultiScopeError):
    """Raised when a loaded model branch cannot complete inference."""

    def __init__(
        self,
        message: str = "Model inference failed.",
        *,
        public_message: str = "Model inference failed.",
        error_code: str = "model_inference_failed",
    ) -> None:
        super().__init__(message)
        self.public_message = public_message
        self.error_code = error_code


class NoUsableModelBranchesError(MultiScopeError):
    """Raised when no model branch can produce a usable prediction."""


class AudioUploadError(MultiScopeError):
    """Base exception for audio upload validation failures."""


class MissingAudioFilenameError(AudioUploadError):
    """Raised when an uploaded audio file has no filename."""


class UnsafeAudioFilenameError(AudioUploadError):
    """Raised when an uploaded audio filename is unsafe."""


class UnsupportedAudioFormatError(AudioUploadError):
    """Raised when an audio file extension or detected format is unsupported."""


class EmptyAudioFileError(AudioUploadError):
    """Raised when an uploaded audio file has no content."""


class AudioFileTooLargeError(AudioUploadError):
    """Raised when an uploaded audio file exceeds the configured size limit."""


class CorruptedAudioError(AudioUploadError):
    """Raised when uploaded content cannot be parsed as valid audio."""


class AudioDurationExceededError(AudioUploadError):
    """Raised when uploaded audio exceeds the configured duration limit."""


class AudioTooShortError(AudioUploadError):
    """Raised when uploaded audio is shorter than the configured minimum."""


class AudioChannelCountExceededError(AudioUploadError):
    """Raised when uploaded audio has too many channels."""


class AudioSampleRateExceededError(AudioUploadError):
    """Raised when uploaded audio has an excessive sample rate."""


class AudioStreamValidationError(AudioUploadError):
    """Raised when an upload does not contain exactly one usable audio stream."""


class VideoStreamNotAllowedError(AudioUploadError):
    """Raised when an uploaded media file contains a video stream."""


class DecodedAudioTooLargeError(AudioUploadError):
    """Raised when decoded PCM would exceed the configured memory budget."""


class AudioProcessingTimeoutError(AudioUploadError):
    """Raised when ffprobe or FFmpeg exceeds its configured timeout."""


class AudioProcessingUnavailableError(MultiScopeError):
    """Raised when required FFmpeg tooling is unavailable."""


class UnusableAudioError(AudioUploadError):
    """Raised when decoded audio cannot be used for preprocessing."""


class SilentAudioError(UnusableAudioError):
    """Raised when decoded audio is silent or too close to silence to use safely."""
