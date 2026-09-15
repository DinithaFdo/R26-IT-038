import logging
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import (
    AdmissionControlError,
    AuthenticationError,
    AudioChannelCountExceededError,
    AudioDurationExceededError,
    AudioFileTooLargeError,
    AudioProcessingTimeoutError,
    AudioProcessingUnavailableError,
    AudioSampleRateExceededError,
    AudioStreamValidationError,
    AudioTooShortError,
    CorruptedAudioError,
    DecodedAudioTooLargeError,
    EmptyAudioFileError,
    IdempotencyConflictError,
    MissingAudioFilenameError,
    ModelLoadError,
    ModelNotIntegratedError,
    NoUsableModelBranchesError,
    SilentAudioError,
    UnsafeAudioFilenameError,
    UnsupportedAudioFormatError,
    UnusableAudioError,
    VideoStreamNotAllowedError,
)
from app.core.request_context import CLIENT_CORRELATION_ID_HEADER, get_request_id

logger = logging.getLogger(__name__)

AUDIO_ERROR_STATUS_CODES = {
    AudioFileTooLargeError: 413,
    DecodedAudioTooLargeError: 413,
    UnsupportedAudioFormatError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    VideoStreamNotAllowedError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}

AUDIO_ERROR_CODES = {
    AudioChannelCountExceededError: "audio_channel_count_exceeded",
    AudioDurationExceededError: "audio_duration_exceeded",
    AudioFileTooLargeError: "audio_file_too_large",
    AudioProcessingTimeoutError: "audio_processing_timeout",
    AudioSampleRateExceededError: "audio_sample_rate_exceeded",
    AudioStreamValidationError: "audio_stream_validation_failed",
    AudioTooShortError: "audio_too_short",
    CorruptedAudioError: "corrupted_audio",
    DecodedAudioTooLargeError: "decoded_audio_too_large",
    EmptyAudioFileError: "empty_audio_file",
    MissingAudioFilenameError: "missing_audio_filename",
    SilentAudioError: "silent_audio",
    UnsafeAudioFilenameError: "unsafe_audio_filename",
    UnsupportedAudioFormatError: "unsupported_audio_format",
    UnusableAudioError: "unusable_audio",
    VideoStreamNotAllowedError: "video_stream_not_allowed",
}


def register_exception_handlers(app) -> None:
    app.add_exception_handler(AdmissionControlError, admission_control_error_handler)
    app.add_exception_handler(AuthenticationError, authentication_error_handler)
    app.add_exception_handler(
        IdempotencyConflictError,
        idempotency_conflict_handler,
    )
    app.add_exception_handler(AudioChannelCountExceededError, audio_error_handler)
    app.add_exception_handler(AudioDurationExceededError, audio_error_handler)
    app.add_exception_handler(AudioFileTooLargeError, audio_error_handler)
    app.add_exception_handler(AudioProcessingTimeoutError, audio_error_handler)
    app.add_exception_handler(AudioSampleRateExceededError, audio_error_handler)
    app.add_exception_handler(AudioStreamValidationError, audio_error_handler)
    app.add_exception_handler(AudioTooShortError, audio_error_handler)
    app.add_exception_handler(CorruptedAudioError, audio_error_handler)
    app.add_exception_handler(DecodedAudioTooLargeError, audio_error_handler)
    app.add_exception_handler(EmptyAudioFileError, audio_error_handler)
    app.add_exception_handler(MissingAudioFilenameError, audio_error_handler)
    app.add_exception_handler(SilentAudioError, audio_error_handler)
    app.add_exception_handler(UnsafeAudioFilenameError, audio_error_handler)
    app.add_exception_handler(UnsupportedAudioFormatError, audio_error_handler)
    app.add_exception_handler(UnusableAudioError, audio_error_handler)
    app.add_exception_handler(VideoStreamNotAllowedError, audio_error_handler)
    app.add_exception_handler(
        AudioProcessingUnavailableError,
        audio_processing_unavailable_handler,
    )
    app.add_exception_handler(ModelLoadError, model_unavailable_handler)
    app.add_exception_handler(ModelNotIntegratedError, model_unavailable_handler)
    app.add_exception_handler(NoUsableModelBranchesError, model_unavailable_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)


async def admission_control_error_handler(
    request: Request,
    exc: AdmissionControlError,
) -> JSONResponse:
    logger.info(
        "Admission control rejected request.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
            "error_code": exc.error_code,
        },
    )
    headers = (
        {"Retry-After": str(exc.retry_after_seconds)}
        if exc.retry_after_seconds is not None
        else None
    )
    return _error_response(
        status_code=exc.status_code,
        code=exc.error_code,
        message=exc.public_message,
        request=request,
        headers=headers,
    )


async def authentication_error_handler(
    request: Request,
    exc: AuthenticationError,
) -> JSONResponse:
    # `category` is a safe, internal-only reason (e.g. "missing_jwks_configuration",
    # "expired_token", "invalid_signature") -- never the raw Authorization header,
    # the token, or any secret. The client always sees the same generic body
    # below regardless of category; this only makes the server log distinguish
    # "auth is broken for everyone" from "this one token is bad".
    logger.info(
        "Authentication failed.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": status.HTTP_401_UNAUTHORIZED,
            "failure_type": getattr(exc, "category", "unspecified"),
        },
    )
    return _error_response(
        status_code=status.HTTP_401_UNAUTHORIZED,
        code="authentication_failed",
        message="Authentication failed.",
        request=request,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def idempotency_conflict_handler(
    request: Request,
    exc: IdempotencyConflictError,
) -> JSONResponse:
    logger.info(
        "Idempotency key reused for a different logical request.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
            "error_code": exc.error_code,
        },
    )
    return _error_response(
        status_code=exc.status_code,
        code=exc.error_code,
        message=_safe_message(exc) or exc.public_message,
        request=request,
        details=getattr(exc, "details", None),
    )


async def audio_processing_unavailable_handler(
    request: Request,
    exc: AudioProcessingUnavailableError,
) -> JSONResponse:
    logger.error(
        "Audio processing tooling unavailable.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
        },
    )
    return _error_response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        code="audio_processing_unavailable",
        message="Audio processing service is unavailable.",
        request=request,
    )


async def audio_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.warning(
        "Domain audio error.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": _audio_status_code(exc),
        },
    )
    return _error_response(
        status_code=_audio_status_code(exc),
        code=AUDIO_ERROR_CODES.get(type(exc), "audio_error"),
        message=_safe_message(exc),
        request=request,
    )


async def model_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.warning(
        "Model unavailable.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
        },
    )
    return _error_response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        code="model_unavailable",
        message=_safe_message(exc) or "No usable model branches are available.",
        request=request,
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.info(
        "Request validation error.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": 422,
        },
    )
    return _error_response(
        status_code=422,
        code="request_validation_error",
        message="Request validation failed.",
        request=request,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    logger.info(
        "HTTP exception.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
        },
    )
    return _error_response(
        status_code=exc.status_code,
        code=_http_error_code(exc.status_code),
        message=_safe_message(exc) or "Request failed.",
        request=request,
        # Forward standard headers set on the raised HTTPException itself
        # (e.g. Retry-After from a 429) instead of silently dropping them --
        # this handler still owns the response body/shape, so nothing about
        # the safe-error contract changes.
        headers=dict(exc.headers) if exc.headers else None,
    )


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unexpected internal error.",
        extra={
            "request_id": _request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )
    return _error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_server_error",
        message="Unexpected internal server error.",
        request=request,
    )


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request: Request,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    response_headers = {"X-Request-ID": request_id}
    client_correlation_id = getattr(request.state, "client_correlation_id", None)
    if client_correlation_id is not None:
        response_headers[CLIENT_CORRELATION_ID_HEADER] = client_correlation_id
    if headers:
        response_headers.update(headers)
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "error": {
                "code": code,
                "message": message,
                "details": details,
            },
        },
        headers=response_headers,
    )


def _audio_status_code(exc: Exception) -> int:
    return AUDIO_ERROR_STATUS_CODES.get(type(exc), status.HTTP_400_BAD_REQUEST)


def _safe_message(exc: Exception) -> str:
    if isinstance(exc, ModelLoadError):
        return exc.public_message
    if isinstance(exc, StarletteHTTPException):
        detail = exc.detail
        return detail if isinstance(detail, str) else "Request failed."
    return str(exc) or "Request failed."


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", get_request_id())


def _http_error_code(status_code: int) -> str:
    if status_code == status.HTTP_404_NOT_FOUND:
        return "not_found"
    if status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        return "method_not_allowed"
    return "http_error"
