import concurrent.futures
import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from app.config.settings import Settings, settings
from app.core.timing import StageTimings, stage_timer
from app.ingestion.audio import (
    AudioUploadMetadata,
    ProcessedAudio,
    audio_tool_versions,
    preprocess_audio_file,
)
from app.models.base import BaseVoiceModel
from app.models.registry import ModelRegistry
from app.schemas.common import BranchStatus, ModelMode
from app.schemas.prediction import (
    AudioMetadata,
    AudioStorageMetadata,
    BranchPrediction,
    FusionResult,
    VoicePredictionResponse,
)
from app.schemas.provenance import (
    BranchModelProvenance,
    FusionProvenance,
    PredictionProvenance,
    PreprocessingProvenance,
)
from app.utils.fusion import ConstrainedFourBranchFusion, FusionEngine
from app.voice_xai.capture.contracts import ExtractionBundle
from app.voice_xai.capture.session import PyTorchExtractionInterface
from app.voice_xai.temporal.contracts import TemporalAttentionWindowInput

logger = logging.getLogger(__name__)

PreprocessFunction = Callable[[AudioUploadMetadata], ProcessedAudio]
PREPROCESSING_VERSION = "audio-preprocessing-v2"
TARGET_CHANNELS = 1


def _build_primary_v3_fusion(app_settings: Settings) -> ConstrainedFourBranchFusion | None:
    """The Fusion V3 engine to try first, or ``None`` when V3 is disabled or
    its artifact is simply absent.

    A present-but-invalid artifact raises ``ConvexFusionContractError`` (a
    ``ValueError``) here -- fails loudly at construction time rather than
    being silently treated as "unavailable", matching how a tampered legacy
    contract already fails startup via ``Settings.frozen_fusion_contract``.
    """

    if app_settings.fusion_mode != "convex_4branch_v3":
        return None
    contract = app_settings.frozen_convex_fusion_v3_contract
    if contract is None:
        logger.warning(
            "fusion_v3_contract_missing",
            extra={"fusion_v3_config_path": app_settings.fusion_v3_config_path},
        )
        return None
    return ConstrainedFourBranchFusion(contract=contract)


@dataclass(frozen=True, slots=True)
class PredictionExecutionResult:
    """Internal-only pairing of a completed prediction with the exact
    ``ProcessedAudio`` instance used to produce it.

    Never returned from a public route. Exists so a caller that needs the
    same audio interpretation the classifier used -- currently the
    post-persistence Voice XAI handoff -- can reuse it instead of
    re-decoding/re-normalising the upload a second time.
    """

    prediction: VoicePredictionResponse
    processed_audio: ProcessedAudio
    extraction: ExtractionBundle | None = None
    temporal_evidence: tuple[TemporalAttentionWindowInput, ...] | None = None


class VoiceService:
    def __init__(
        self,
        *,
        model_registry: ModelRegistry | None = None,
        fusion_engine: FusionEngine | None = None,
        preprocess_fn: PreprocessFunction | None = None,
        app_settings: Settings = settings,
    ) -> None:
        self._settings = app_settings
        self._model_registry = model_registry or ModelRegistry(app_settings=app_settings)
        self._fusion_engine = fusion_engine or FusionEngine.from_settings(app_settings)
        # An explicitly passed `fusion_engine` is a caller asking to control
        # fusion directly (existing convention, e.g. tests exercising a
        # specific dev-mode engine) -- Fusion V3 is only auto-attempted on
        # the default, settings-driven path.
        self._primary_fusion_v3 = (
            None if fusion_engine is not None else _build_primary_v3_fusion(app_settings)
        )
        self._legacy_fallback_enabled = app_settings.legacy_fusion_fallback_enabled
        self._preprocess_fn = preprocess_fn or _preprocess_from_upload_metadata

    def _fuse(
        self, branch_predictions: list[BranchPrediction]
    ) -> tuple[FusionResult, dict[str, float]]:
        """The final fusion decision for one request, plus the configured
        weights of whichever engine actually produced it.

        Tries Fusion V3 first when configured/available. Falls back to the
        legacy 3-branch detector -- tagged `fallback_used=True` -- only when
        V3 was tried and came back unavailable for this specific request's
        branches; never fabricates a partial V3 score.
        """

        if self._primary_fusion_v3 is not None:
            v3_result = self._primary_fusion_v3.fuse(branch_predictions)
            if v3_result.status == BranchStatus.success:
                return v3_result, dict(self._primary_fusion_v3.branch_weights)
            if self._legacy_fallback_enabled:
                logger.warning(
                    "fusion_v3_unavailable_falling_back_to_legacy",
                    extra={
                        "v3_warning": v3_result.warning,
                        "v3_excluded_branches": v3_result.excluded_branches,
                    },
                )
                legacy_result = self._fusion_engine.fuse(branch_predictions)
                legacy_result = legacy_result.model_copy(update={"fallback_used": True})
                return legacy_result, dict(self._fusion_engine.branch_weights)
            return v3_result, dict(self._primary_fusion_v3.branch_weights)
        legacy_result = self._fusion_engine.fuse(branch_predictions)
        return legacy_result, dict(self._fusion_engine.branch_weights)

    def predict_from_validated_upload(
        self,
        upload_metadata: AudioUploadMetadata,
        *,
        request_id: str | None = None,
        storage_metadata: AudioStorageMetadata | None = None,
        cleanup_upload: bool = True,
    ) -> VoicePredictionResponse:
        return self._predict_from_validated_upload(
            upload_metadata,
            request_id=request_id,
            storage_metadata=storage_metadata,
            cleanup_upload=cleanup_upload,
        ).prediction

    def predict_from_validated_upload_with_processed_audio(
        self,
        upload_metadata: AudioUploadMetadata,
        *,
        request_id: str | None = None,
        storage_metadata: AudioStorageMetadata | None = None,
        cleanup_upload: bool = True,
        capture_extraction: bool = False,
    ) -> PredictionExecutionResult:
        """Same behaviour as :meth:`predict_from_validated_upload`, plus the
        exact ``ProcessedAudio`` instance used for classifier inference.

        Internal callers only (e.g. the post-persistence Voice XAI handoff).
        The public prediction response/contract is unaffected -- this method
        returns it unchanged, just alongside the processed audio.
        """

        return self._predict_from_validated_upload(
            upload_metadata,
            request_id=request_id,
            storage_metadata=storage_metadata,
            cleanup_upload=cleanup_upload,
            capture_extraction=capture_extraction,
        )

    def _predict_from_validated_upload(
        self,
        upload_metadata: AudioUploadMetadata,
        *,
        request_id: str | None = None,
        storage_metadata: AudioStorageMetadata | None = None,
        cleanup_upload: bool = True,
        capture_extraction: bool = False,
    ) -> PredictionExecutionResult:
        request_id = request_id or str(uuid4())
        start_time = perf_counter()
        captured_branches: dict = {}
        temporal_evidence: tuple[TemporalAttentionWindowInput, ...] | None = None
        capture_metadata: dict[str, object] = {
            "capture_requested": capture_extraction,
            "capture_enabled": False,
        }

        logger.info(
            "Voice prediction request started.",
            extra={
                "request_id": request_id,
                "upload_filename": upload_metadata.sanitized_filename,
                "file_size_bytes": upload_metadata.file_size_bytes,
            },
        )

        try:
            timings = StageTimings()
            with stage_timer(
                "audio_preprocessing",
                timings=timings,
                logger_name=__name__,
                request_id=request_id,
            ):
                processed_audio = self._preprocess_fn(upload_metadata)
            logger.info(
                "Audio preprocessing complete.",
                extra={
                    "request_id": request_id,
                    "sample_rate": processed_audio.sample_rate,
                    "duration_seconds": processed_audio.duration_seconds,
                    "was_resampled": processed_audio.was_resampled,
                    "was_converted_to_mono": processed_audio.was_converted_to_mono,
                },
            )

            branch_predictions = []
            for model in self._model_registry.models:
                branch_stage = f"branch_{_branch_stage_name(model.model_name)}"
                capture_temporal_evidence = (
                    self._settings.xai_enabled
                    and model.mode == ModelMode.real
                    and model.branch_name == "ssl_sequence"
                )
                capture_handler = (
                    (
                        lambda captured_model, audio, captured_request_id: self._capture_branch(
                            captured_model,
                            audio,
                            captured_request_id,
                            capture_temporal_evidence=capture_temporal_evidence,
                        )
                    )
                    if capture_extraction
                    else None
                )
                with stage_timer(
                    branch_stage,
                    timings=timings,
                    logger_name=__name__,
                    request_id=request_id,
                    branch_name=model.model_name,
                    model_mode=model.mode.value,
                ):
                    (
                        branch_prediction,
                        branch_captures,
                        branch_capture_metadata,
                        branch_temporal_evidence,
                    ) = _predict_branch_with_timeout(
                        model,
                        processed_audio,
                        request_id=request_id,
                        capture=capture_handler,
                        capture_temporal_evidence=capture_temporal_evidence,
                    )
                if branch_captures:
                    captured_branches.update(branch_captures)
                if branch_capture_metadata is not None:
                    capture_metadata["capture_enabled"] = True
                    capture_metadata.setdefault("branches", {})[model.branch_name] = (
                        branch_capture_metadata
                    )
                if branch_temporal_evidence:
                    temporal_evidence = branch_temporal_evidence
                branch_prediction = _branch_with_model_provenance(
                    branch_prediction,
                    model.provenance(),
                )
                branch_predictions.append(branch_prediction)
                branch_log_message = (
                    "Voice model branch complete."
                    if branch_prediction.status.value == "success"
                    else "Voice model branch failed."
                )
                logger.info(
                    branch_log_message,
                    extra={
                        "request_id": request_id,
                        "model_name": branch_prediction.model_name,
                        "status": branch_prediction.status.value,
                        "mode": branch_prediction.mode.value,
                    },
                )

            with stage_timer(
                "fusion",
                timings=timings,
                logger_name=__name__,
                request_id=request_id,
            ):
                fusion, configured_weights = self._fuse(branch_predictions)
            logger.info(
                "Voice prediction fusion complete.",
                extra={
                    "request_id": request_id,
                    "status": fusion.status.value,
                    "method": fusion.method,
                    "fusion_version": fusion.fusion_version,
                    "fallback_used": fusion.fallback_used,
                    "contains_dummy_branches": fusion.contains_dummy_branches,
                    "fusion_config_version": fusion.config_version,
                },
            )

            prediction = VoicePredictionResponse(
                request_id=request_id,
                audio=_audio_metadata_from_upload(
                    upload_metadata,
                    storage_metadata=storage_metadata,
                ),
                branches=branch_predictions,
                fusion=fusion,
                provenance=_prediction_provenance(
                    processed_audio=processed_audio,
                    models=[
                        model.provenance() for model in self._model_registry.models
                    ],
                    fusion=fusion,
                    configured_weights=configured_weights,
                ),
                total_processing_time_ms=_elapsed_ms(start_time),
            )
            return PredictionExecutionResult(
                prediction=prediction,
                processed_audio=processed_audio,
                extraction=(
                    ExtractionBundle(
                        request_id=request_id,
                        branches=captured_branches,
                        metadata=capture_metadata,
                    )
                    if captured_branches
                    else None
                ),
                temporal_evidence=temporal_evidence,
            )
        finally:
            if cleanup_upload:
                upload_metadata.saved_path.unlink(missing_ok=True)
            logger.info(
                "Voice prediction request finished.",
                extra={
                    "request_id": request_id,
                    "total_processing_time_ms": _elapsed_ms(start_time),
                },
            )

    def model_health(self) -> list[dict]:
        return self._model_registry.health()

    def model_readiness(self) -> dict:
        return self._model_registry.readiness()

    def load_startup_models(self) -> None:
        self._model_registry.load_startup_models()

    def unload_models(self) -> None:
        self._model_registry.unload_all()

    def _capture_branch(
        self,
        model: BaseVoiceModel,
        processed_audio: ProcessedAudio,
        request_id: str,
        *,
        capture_temporal_evidence: bool,
    ) -> tuple[
        BranchPrediction,
        dict,
        dict[str, object] | None,
        tuple[TemporalAttentionWindowInput, ...] | None,
    ]:
        """Capture one real branch inside its existing inference worker.

        ``ContextVar`` state does not cross a ``ThreadPoolExecutor`` boundary,
        so this method is intentionally invoked *inside* the worker submitted
        by ``_predict_branch_with_timeout``. Any capture setup failure is
        isolated and falls back to the unmodified classifier call.
        """

        try:
            interface = self._capture_interface_for(model)
        except Exception:
            logger.exception(
                "voice_xai_capture_setup_failed",
                extra={"request_id": request_id, "branch_name": model.branch_name},
            )
            prediction, evidence = _predict_with_optional_temporal_evidence(
                model, processed_audio, capture_temporal_evidence
            )
            return prediction, {}, None, evidence
        if interface is None:
            prediction, evidence = _predict_with_optional_temporal_evidence(
                model, processed_audio, capture_temporal_evidence
            )
            return prediction, {}, None, evidence

        try:
            with interface.capture(request_id) as session:
                prediction, evidence = _predict_with_optional_temporal_evidence(
                    model, processed_audio, capture_temporal_evidence
                )
            bundle = session.build_bundle()
            return prediction, bundle.branches, bundle.metadata, evidence
        finally:
            # Hooks exist only for this opt-in inference. Subsequent ordinary
            # predictions have no registered callbacks and therefore retain
            # their original hot path.
            interface.close()

    def _capture_interface_for(
        self, model: BaseVoiceModel
    ) -> PyTorchExtractionInterface | None:
        """Create and register a short-lived interface for one real branch."""

        if model.mode != ModelMode.real:
            return None
        root_module = model.xai_capture_root()
        if root_module is None:
            return None

        interface = PyTorchExtractionInterface(
            max_total_elements=self._settings.xai_capture_max_total_elements
        )
        # Capture plans live beside the reconstructed model architectures,
        # preserving the classifier/XAI ownership boundary.
        from app.models.capture_targets import register_real_branch_capture

        if not register_real_branch_capture(interface, model.branch_name, root_module):
            interface.close()
            return None
        return interface


def _preprocess_from_upload_metadata(
    upload_metadata: AudioUploadMetadata,
) -> ProcessedAudio:
    return preprocess_audio_file(
        upload_metadata.saved_path,
        extension=upload_metadata.saved_path.suffix.lower().lstrip("."),
        inspection=upload_metadata.inspection,
    )


def _audio_metadata_from_upload(
    upload_metadata: AudioUploadMetadata,
    *,
    storage_metadata: AudioStorageMetadata | None = None,
) -> AudioMetadata:
    return AudioMetadata(
        original_filename=upload_metadata.original_filename,
        content_type=upload_metadata.content_type or "",
        original_extension=upload_metadata.original_extension,
        detected_container=upload_metadata.detected_container,
        detected_format=upload_metadata.detected_format,
        detected_codec=upload_metadata.detected_codec,
        size_bytes=upload_metadata.size_bytes,
        file_size_bytes=upload_metadata.file_size_bytes,
        duration_seconds=upload_metadata.duration_seconds,
        sample_rate=upload_metadata.sample_rate,
        channels=upload_metadata.channels,
        storage=storage_metadata,
    )


def _branch_with_model_provenance(
    branch: BranchPrediction,
    model_provenance: BranchModelProvenance,
) -> BranchPrediction:
    metadata = {
        **branch.metadata,
        "model_provenance": model_provenance.model_dump(mode="python"),
    }
    if branch.mode == ModelMode.dummy:
        metadata["research_result"] = False
    return branch.model_copy(update={"metadata": metadata})


def _predict_branch_with_timeout(
    model: BaseVoiceModel,
    processed_audio: ProcessedAudio,
    *,
    request_id: str,
    capture: Callable[
        [BaseVoiceModel, ProcessedAudio, str],
        tuple[
            BranchPrediction,
            dict,
            dict[str, object] | None,
            tuple[TemporalAttentionWindowInput, ...] | None,
        ],
    ]
    | None = None,
    capture_temporal_evidence: bool = False,
) -> tuple[
    BranchPrediction,
    dict,
    dict[str, object] | None,
    tuple[TemporalAttentionWindowInput, ...] | None,
]:
    timeout = (
        model._config.branch_timeout_seconds
        if getattr(model, "_config", None) is not None
        else 30
    )
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    if capture is None and not capture_temporal_evidence:
        future = executor.submit(
            lambda: (model.predict_safe(processed_audio), {}, None, None)
        )
    elif capture is None:
        future = executor.submit(
            _prediction_without_capture,
            model,
            processed_audio,
            capture_temporal_evidence,
        )
    else:
        # Capture opens its ContextVar session inside this exact worker. A
        # context set by the request thread would not propagate here.
        future = executor.submit(capture, model, processed_audio, request_id)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "model_inference_timed_out",
            extra={
                "branch_name": model.branch_name,
                "mode": model.mode.value,
                "error_code": "model_timeout",
            },
        )
        return (
            BranchPrediction(
                model_name=model.model_name,
                display_name=model.display_name,
                status=BranchStatus.failed,
                mode=model.mode,
                prediction=None,
                confidence=None,
                probabilities=None,
                processing_time_ms=max(timeout * 1000, 0),
                error="Model inference timed out.",
                metadata={
                    "error_code": "model_timeout",
                    "model_provenance": model.provenance().model_dump(mode="python"),
                    "research_result": False,
                },
            ),
            {},
            None,
            None,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _predict_with_optional_temporal_evidence(
    model: BaseVoiceModel,
    processed_audio: ProcessedAudio,
    capture_temporal_evidence: bool,
) -> tuple[BranchPrediction, tuple[TemporalAttentionWindowInput, ...] | None]:
    if not capture_temporal_evidence:
        return model.predict_safe(processed_audio), None
    prediction, evidence = model.predict_safe_with_temporal_evidence(processed_audio)
    return prediction, tuple(evidence) if evidence else None


def _prediction_without_capture(
    model: BaseVoiceModel,
    processed_audio: ProcessedAudio,
    capture_temporal_evidence: bool,
) -> tuple[
    BranchPrediction,
    dict,
    dict[str, object] | None,
    tuple[TemporalAttentionWindowInput, ...] | None,
]:
    prediction, evidence = _predict_with_optional_temporal_evidence(
        model, processed_audio, capture_temporal_evidence
    )
    return prediction, {}, None, evidence


def _prediction_provenance(
    *,
    processed_audio: ProcessedAudio,
    models: list[BranchModelProvenance],
    fusion: FusionResult,
    configured_weights: dict[str, float],
) -> PredictionProvenance:
    tool_versions = audio_tool_versions()
    return PredictionProvenance(
        preprocessing=PreprocessingProvenance(
            input_sample_rate=processed_audio.original_sample_rate,
            input_channels=processed_audio.original_channels,
            input_duration_seconds=processed_audio.duration_seconds,
            target_sample_rate=processed_audio.sample_rate,
            target_channels=TARGET_CHANNELS,
            resampled=processed_audio.was_resampled,
            mono_conversion_applied=processed_audio.was_converted_to_mono,
            normalisation_applied=processed_audio.normalisation_applied,
            preprocessing_version=processed_audio.preprocessing_version,
            minimum_duration_seconds=processed_audio.minimum_duration_seconds,
            model_window_duration_seconds=processed_audio.model_window_duration_seconds,
            model_window_overlap_seconds=processed_audio.model_window_overlap_seconds,
            segment_count=len(processed_audio.segments),
            padding_policy="right_zero",
            trim_policy="none",
            ffmpeg_version=tool_versions["ffmpeg_version"],
            ffprobe_version=tool_versions["ffprobe_version"],
        ),
        models=models,
        fusion=FusionProvenance(
            fusion_method=fusion.method,
            configured_weights=configured_weights,
            effective_weights=dict(fusion.branch_weights),
            threshold=fusion.decision_threshold,
            fusion_version=fusion.fusion_version,
            fusion_mode=fusion.fusion_mode,
            fallback_used=fusion.fallback_used,
            contains_dummy_branches=fusion.contains_dummy_branches,
            eligible_for_research_evaluation=(
                fusion.eligible_for_research_evaluation
            ),
            config_version=fusion.config_version,
            minimum_successful_branches=fusion.minimum_successful_branches,
            contributing_branches=fusion.contributing_branches,
            excluded_branches=fusion.excluded_branches,
        ),
    )


def _elapsed_ms(start_time: float) -> float:
    return max((perf_counter() - start_time) * 1000, 0.0)


def _branch_stage_name(model_name: str) -> str:
    if "cnn" in model_name:
        return "cnn"
    if "aasist" in model_name:
        return "aasist"
    if "ssl" in model_name:
        return "ssl"
    if "glottal" in model_name:
        return "glottal"
    return model_name.replace("-", "_")
