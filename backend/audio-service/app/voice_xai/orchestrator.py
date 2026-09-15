"""Independent post-prediction Voice XAI orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import logging
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Protocol

from app.config.settings import Settings
from app.models.runtime import canonical_branch_name
from app.schemas.common import PredictionStatus, SourceType
from app.schemas.prediction import AudioMetadata, FusionResult, VoicePredictionResponse
from app.schemas.xai import (
    CanonicalXaiBranch,
    ClassifierAuxiliaryEvidence,
    ClassifierBranchSnapshot,
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationArtifactReference,
    ExplanationComponent,
    ExplanationError,
    ExplanationProvenance,
    ExplanationQuality,
    NarrativeExplanation,
    SemanticExplanation,
    TemporalExplanation,
    XaiExplanationResponse,
)
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.artifacts.protocols import ExplanationArtifactStore, StoredArtifact
from app.voice_xai.artifacts.service import (
    ArtifactStorageError,
    LocalExplanationArtifactStore,
)
from app.voice_xai.capture.serialization import (
    EXTRACTION_ARTIFACT_CONTENT_TYPE,
    serialize_extraction_bundle,
)
from app.voice_xai.jobs.queue import AsynchronousExplanationQueue
from app.voice_xai.persistence.protocols import XaiExplanationRepository
from app.voice_xai.evaluation.consistency import quality_for_analysis
from app.voice_xai.report.service import DeterministicReportComposer
from app.voice_xai.narrative.service import (
    AlibabaQwenNarrativeService,
    NarrativeGenerationError,
)
from app.voice_xai.replay import StoredAudioReplayService, XaiSourceAudioUnavailableError
from app.voice_xai.evaluation.eligibility import evaluate_research_eligibility
from app.voice_xai.semantic.service import MockSemanticExplanationService
from app.voice_xai.status import XaiExplanationNotFoundError, XaiStatusService
from app.voice_xai.temporal.contracts import (
    TemporalAttentionError,
    TemporalAttentionWindowInput,
)
from app.voice_xai.temporal.serialization import (
    TEMPORAL_EVIDENCE_ARTIFACT_CONTENT_TYPE,
    deserialize_temporal_evidence,
    serialize_temporal_evidence,
)
from app.voice_xai.temporal.spectrogram import mel_spectrogram_payload

logger = logging.getLogger(__name__)


class PredictionLookup(Protocol):
    async def get_prediction_for_owner(
        self, *, prediction_id: str, owner_user_id: str, include_deleted: bool = False
    ) -> dict[str, Any] | None: ...


class XaiOrchestrationError(RuntimeError):
    """Safe application error for explanation trigger requests."""


class XaiRetryRequiredError(XaiOrchestrationError):
    """Raised when the latest explanation is terminally failed."""


class XaiReplayUnavailableError(XaiOrchestrationError):
    """Raised when a manual XAI run cannot reconstruct its source audio."""


class XaiArtifactUnavailableError(XaiOrchestrationError):
    """Raised when an owned artifact cannot safely be served."""


class XaiNarrativeRetryError(XaiOrchestrationError):
    """Raised when an optional narrative-only retry is not eligible."""


class XaiJobTimedOutError(RuntimeError):
    """Raised cooperatively between XAI component stages after a deadline."""


class SemanticExplanationService(Protocol):
    def analyze(self, inference: ClassifierInferenceBundle): ...


class NarrativeExplanationService(Protocol):
    async def generate(
        self,
        *,
        classifier: ClassifierSnapshot,
        temporal: TemporalExplanation | None,
        semantic: SemanticExplanation | None,
        quality: ExplanationQuality,
        report: CombinedExplanationReport,
        timeout_seconds: float,
    ) -> NarrativeExplanation: ...


@dataclass(frozen=True)
class NarrativeRetryJob:
    """Minimal queue payload for work that never needs source audio."""

    explanation_id: str
    prediction_id: str
    request_id: str
    owner_user_id: str


class VoiceXaiOrchestrator:
    """Creates and executes XAI runs without participating in prediction state."""

    def __init__(
        self,
        *,
        repository: XaiExplanationRepository,
        prediction_repository: PredictionLookup,
        queue: AsynchronousExplanationQueue,
        app_settings: Settings,
        artifact_store: ExplanationArtifactStore | None = None,
        temporal_service: Any,
        semantic_service: SemanticExplanationService | None = None,
        report_composer: DeterministicReportComposer | None = None,
        narrative_service: NarrativeExplanationService | None = None,
        replay_service: StoredAudioReplayService | None = None,
        worker_repository_factory: Callable[[], XaiExplanationRepository] | None = None,
    ) -> None:
        self._repository = repository
        self._prediction_repository = prediction_repository
        self._queue = queue
        self._settings = app_settings
        self._artifact_store = artifact_store or LocalExplanationArtifactStore(
            app_settings.resolve_xai_artifact_path(app_settings.xai_artifact_root),
            retention_seconds=app_settings.xai_artifact_retention_seconds,
        )
        self._temporal_service = temporal_service
        self._semantic_service = semantic_service or MockSemanticExplanationService()
        self._report_composer = report_composer or DeterministicReportComposer()
        self._narrative_service = narrative_service or AlibabaQwenNarrativeService(
            app_settings
        )
        self._replay_service = replay_service
        self._status = XaiStatusService(repository)
        # Resolved once per background run (see _worker_repository), never
        # reused across runs on a different event loop than the one it was
        # built for. Defaults to the main-loop repository, matching every
        # existing caller/test that does not run the queue's dedicated
        # persistent worker loop.
        self._worker_repository_factory = worker_repository_factory

    def set_semantic_service(self, service: SemanticExplanationService) -> None:
        """Install the startup-loaded production service before requests begin."""

        self._semantic_service = service

    def readiness(self) -> dict[str, object]:
        """Expose safe XAI readiness without making prediction readiness depend on XAI."""

        semantic_real = self._semantic_service.__class__.__name__ == "ProductionSemanticExplanationService"
        temporal_real = self._temporal_service.__class__.__name__ == "XLSRTemporalAttentionService"
        return {
            "enabled": self._settings.xai_enabled,
            "mode": self._settings.xai_mode,
            "ready": self._settings.xai_enabled and self._settings.xai_mode in {"mock", "real"},
            "research_ready": semantic_real and temporal_real,
            "semantic_service": "real" if semantic_real else "mock",
            "temporal_service": "xlsr",
            "narrative_enabled": self._settings.xai_narrative_enabled,
            "narrative_model": (
                self._settings.xai_narrative_model
                if self._settings.xai_narrative_enabled
                else None
            ),
            "narrative_retry_max_attempts": self._settings.xai_narrative_retry_max_attempts,
            "job_timeout_seconds": self._settings.xai_job_timeout_seconds,
            "artifact_retention_seconds": self._settings.xai_artifact_retention_seconds,
            "temporal_threshold_config_path": self._settings.xai_temporal_threshold_config_path,
            "temporal_visualization_quantile": (
                self._settings.xai_temporal_visualization_quantile
            ),
        }

    async def trigger_for_prediction(
        self, *, prediction_id: str, owner_user_id: str, retry: bool = False
    ) -> XaiExplanationResponse:
        self._require_enabled()
        document = await self._prediction_repository.get_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=owner_user_id,
        )
        if document is None:
            raise XaiExplanationNotFoundError("Prediction was not found.")
        if document.get("status") != PredictionStatus.completed.value:
            raise XaiOrchestrationError("An explanation can only be created for a completed prediction.")
        latest = await self._repository.get_latest_explanation_for_prediction(
            prediction_id=prediction_id, owner_user_id=owner_user_id
        )
        if not retry and latest is not None:
            latest_response = response_from_document(latest)
            if latest_response.status.value == "failed":
                logger.info(
                    "voice_xai_trigger_retry_required",
                    extra={
                        "explanation_id": latest_response.explanation_id,
                        "prediction_id": prediction_id,
                        "request_id": latest_response.request_id,
                    },
                )
                raise XaiRetryRequiredError(
                    "The latest explanation failed. Use the retry endpoint to create a new run."
                )
            logger.info(
                "voice_xai_trigger_reused_existing_run",
                extra={
                    "explanation_id": latest_response.explanation_id,
                    "prediction_id": prediction_id,
                    "request_id": latest_response.request_id,
                    "status": latest_response.status.value,
                },
            )
            return latest_response
        bundle = await self._replay_bundle(document, owner_user_id)
        return await self.enqueue(bundle)

    async def _replay_bundle(
        self, document: dict[str, Any], owner_user_id: str
    ) -> ClassifierInferenceBundle:
        bundle = inference_bundle_from_prediction_document(document)
        if self._replay_service is None:
            raise XaiReplayUnavailableError(
                "Source audio replay is not configured for manual explanations."
            )
        try:
            processed_audio = await self._replay_service.load_processed_audio(
                prediction_document=document,
                owner_user_id=owner_user_id,
            )
        except XaiSourceAudioUnavailableError as error:
            logger.warning(
                "voice_xai_source_audio_replay_unavailable",
                extra={
                    "prediction_id": bundle.prediction_id,
                    "request_id": bundle.request_id,
                },
            )
            raise XaiReplayUnavailableError(str(error)) from error
        temporal_evidence = await self._load_private_temporal_evidence(
            prediction_id=bundle.prediction_id,
            owner_user_id=owner_user_id,
        )
        return replace(
            bundle,
            processed_audio=processed_audio,
            temporal_evidence=temporal_evidence,
        )

    async def enqueue(self, bundle: ClassifierInferenceBundle) -> XaiExplanationResponse:
        """Persist a queued run, then request non-blocking background execution."""

        self._require_enabled()
        explanation_id = await self._repository.create_explanation(
            prediction_id=bundle.prediction_id,
            request_id=bundle.request_id,
            owner_user_id=bundle.owner_user_id,
            source_type=bundle.source_type,
            classifier_snapshot=classifier_snapshot_from_prediction(bundle.prediction),
            pipeline_version=self._settings.xai_pipeline_version,
            provenance=ExplanationProvenance(
                pipeline_version=self._settings.xai_pipeline_version,
                classifier_contract_version=bundle.contract_version,
                feature_extractor_version=(
                    bundle.extraction.acoustic_features.extractor_version
                    if bundle.extraction and bundle.extraction.acoustic_features
                    else None
                ),
                configuration_hash=_configuration_hash(self._settings),
                configuration_hash_inputs=_configuration_hash_inputs(),
            ),
        )
        logger.info(
            "voice_xai_run_queued",
            extra={
                "explanation_id": explanation_id,
                "prediction_id": bundle.prediction_id,
                "request_id": bundle.request_id,
            },
        )
        await self._store_private_temporal_evidence(
            explanation_id=explanation_id,
            bundle=bundle,
        )
        # `self._run` is a coroutine function: calling it (without awaiting)
        # returns a coroutine object for the queue to schedule on its own
        # persistent worker loop. No asyncio.run() is created per job here.
        future = self._queue.submit(
            bundle, lambda queued_bundle: self._run(explanation_id, queued_bundle)
        )
        if future is None:
            await self._status.block_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.temporal,
                error=_error(
                    ExplanationComponent.temporal,
                    "xai_queue_full",
                    "The explanation queue is at capacity. Retry this explanation later.",
                ),
            )
            logger.warning(
                "voice_xai_run_queue_rejected",
                extra={
                    "explanation_id": explanation_id,
                    "prediction_id": bundle.prediction_id,
                    "request_id": bundle.request_id,
                },
            )
        return await self.get_for_owner(
            explanation_id=explanation_id, owner_user_id=bundle.owner_user_id
        )

    async def retry_narrative_for_prediction(
        self, *, prediction_id: str, owner_user_id: str
    ) -> XaiExplanationResponse:
        """Queue only the optional language-model rendering for a completed run.

        The retry uses persisted, already validated evidence. It never replays
        audio, re-runs classifier/XAI components, or alters the deterministic
        report that the user already received.
        """

        self._require_enabled()
        if not self._settings.xai_narrative_enabled:
            raise XaiNarrativeRetryError("AI narrative generation is disabled.")
        document = await self._repository.get_latest_explanation_for_prediction(
            prediction_id=prediction_id,
            owner_user_id=owner_user_id,
        )
        if document is None:
            raise XaiExplanationNotFoundError("Explanation was not found.")
        response = response_from_document(document)
        if response.status.value != "completed" or response.combined_report is None:
            raise XaiNarrativeRetryError(
                "AI narrative can be retried only after the deterministic report completes."
            )
        narrative_status = response.component_statuses.narrative
        if narrative_status in {ComponentStatus.queued, ComponentStatus.running}:
            raise XaiNarrativeRetryError("AI narrative is already being generated.")
        if narrative_status == ComponentStatus.completed:
            raise XaiNarrativeRetryError("AI narrative has already been generated.")
        if narrative_status not in {
            ComponentStatus.failed,
            ComponentStatus.not_available,
        }:
            raise XaiNarrativeRetryError("AI narrative is not eligible for retry.")
        retry_count = document.get("narrative_retry_count", 0)
        if not isinstance(retry_count, int) or retry_count < 0:
            raise XaiNarrativeRetryError("AI narrative retry history is invalid.")
        if retry_count >= self._settings.xai_narrative_retry_max_attempts:
            raise XaiNarrativeRetryError(
                "AI narrative retry limit reached for this explanation."
            )
        reserved = await self._repository.begin_narrative_retry(
            response.explanation_id,
            max_attempts=self._settings.xai_narrative_retry_max_attempts,
        )
        if not reserved:
            raise XaiNarrativeRetryError(
                "AI narrative retry could not be started because its state changed."
            )
        job = NarrativeRetryJob(
            explanation_id=response.explanation_id,
            prediction_id=response.prediction_id,
            request_id=response.request_id,
            owner_user_id=owner_user_id,
        )
        future = self._queue.submit(
            job,
            lambda queued_job: self._run_narrative_retry(queued_job),
        )
        if future is None:
            await self._repository.finish_narrative_retry(
                response.explanation_id,
                error=_error(
                    ExplanationComponent.narrative,
                    "narrative_retry_queue_full",
                    "The AI narrative retry queue is at capacity. Try again later.",
                ),
            )
            logger.warning(
                "voice_xai_narrative_retry_queue_rejected",
                extra={
                    "explanation_id": response.explanation_id,
                    "prediction_id": prediction_id,
                    "request_id": response.request_id,
                },
            )
        else:
            logger.info(
                "voice_xai_narrative_retry_queued",
                extra={
                    "explanation_id": response.explanation_id,
                    "prediction_id": prediction_id,
                    "request_id": response.request_id,
                },
            )
        return await self.get_for_owner(
            explanation_id=response.explanation_id,
            owner_user_id=owner_user_id,
        )

    async def _run_narrative_retry(self, job: NarrativeRetryJob) -> None:
        """Execute the provider call for a previously atomically reserved retry."""

        repository = self._worker_repository()
        document = await repository.get_explanation_for_owner(
            explanation_id=job.explanation_id,
            owner_user_id=job.owner_user_id,
        )
        if document is None:
            logger.warning(
                "voice_xai_narrative_retry_missing",
                extra={"explanation_id": job.explanation_id},
            )
            return
        response = response_from_document(document)
        if response.component_statuses.narrative != ComponentStatus.running:
            return
        try:
            if response.combined_report is None:
                raise XaiNarrativeRetryError(
                    "The deterministic report is unavailable for AI narrative retry."
                )
            narrative = await self._narrative_service.generate(
                classifier=response.classifier_snapshot,
                temporal=response.temporal,
                semantic=response.semantic,
                quality=response.quality,
                report=response.combined_report,
                timeout_seconds=self._settings.xai_narrative_timeout_seconds,
            )
            await repository.finish_narrative_retry(
                job.explanation_id,
                result=narrative,
            )
            logger.info(
                "voice_xai_narrative_retry_completed",
                extra={
                    "explanation_id": job.explanation_id,
                    "prediction_id": job.prediction_id,
                    "request_id": job.request_id,
                },
            )
        except NarrativeGenerationError as error:
            await self._record_narrative_retry_failure(
                repository=repository,
                job=job,
                code=error.code,
                message=str(error),
            )
        except Exception:
            logger.exception(
                "Voice XAI narrative retry failed for %s", job.explanation_id
            )
            await self._record_narrative_retry_failure(
                repository=repository,
                job=job,
                code="narrative_retry_failed",
                message="The AI narrative could not be generated.",
            )

    async def _record_narrative_retry_failure(
        self,
        *,
        repository: XaiExplanationRepository,
        job: NarrativeRetryJob,
        code: str,
        message: str,
    ) -> None:
        await repository.finish_narrative_retry(
            job.explanation_id,
            error=_error(ExplanationComponent.narrative, code, message),
        )
        logger.warning(
            "voice_xai_narrative_retry_failed",
            extra={
                "explanation_id": job.explanation_id,
                "prediction_id": job.prediction_id,
                "request_id": job.request_id,
                "narrative_error_code": code,
            },
        )

    async def _store_private_temporal_evidence(
        self, *, explanation_id: str, bundle: ClassifierInferenceBundle
    ) -> None:
        """Persist retry-only compact evidence outside public artifact metadata."""

        if not bundle.temporal_evidence:
            return
        try:
            reference = self._artifact_store.store_bytes(
                explanation_id=explanation_id,
                content=serialize_temporal_evidence(bundle.temporal_evidence),
                kind="temporal_evidence",
                content_type=TEMPORAL_EVIDENCE_ARTIFACT_CONTENT_TYPE,
            )
            await self._repository.set_private_temporal_evidence_artifact(
                explanation_id,
                reference,
            )
        except Exception:  # noqa: BLE001 - prediction/XAI run remains usable
            logger.exception(
                "voice_xai_temporal_evidence_artifact_store_failed",
                extra={
                    "explanation_id": explanation_id,
                    "prediction_id": bundle.prediction_id,
                    "request_id": bundle.request_id,
                },
            )

    async def _load_private_temporal_evidence(
        self, *, prediction_id: str, owner_user_id: str
    ) -> tuple[TemporalAttentionWindowInput, ...] | None:
        """Find the newest valid private timeline from earlier XAI runs."""

        try:
            explanations = await self._repository.list_explanations_for_prediction_for_owner(
                prediction_id=prediction_id,
                owner_user_id=owner_user_id,
            )
        except Exception:  # noqa: BLE001 - semantic replay may still proceed
            logger.exception(
                "voice_xai_temporal_evidence_lookup_failed",
                extra={"prediction_id": prediction_id},
            )
            return None
        for explanation in explanations:
            reference = _private_temporal_evidence_reference(explanation)
            if reference is None:
                continue
            try:
                artifact = self._artifact_store.read(reference)
                if artifact is None:
                    continue
                return deserialize_temporal_evidence(artifact.content)
            except Exception:  # noqa: BLE001 - try an older valid artifact
                logger.warning(
                    "voice_xai_temporal_evidence_artifact_unavailable",
                    extra={
                        "prediction_id": prediction_id,
                        "explanation_id": explanation.get("id"),
                    },
                    exc_info=True,
                )
        return None

    async def get_for_prediction(
        self, *, prediction_id: str, owner_user_id: str
    ) -> XaiExplanationResponse:
        document = await self._repository.get_latest_explanation_for_prediction(
            prediction_id=prediction_id, owner_user_id=owner_user_id
        )
        if document is None:
            raise XaiExplanationNotFoundError("XAI explanation was not found.")
        return response_from_document(document)

    async def get_for_owner(
        self, *, explanation_id: str, owner_user_id: str
    ) -> XaiExplanationResponse:
        document = await self._repository.get_explanation_for_owner(
            explanation_id=explanation_id, owner_user_id=owner_user_id
        )
        if document is None:
            raise XaiExplanationNotFoundError("XAI explanation was not found.")
        return response_from_document(document)

    async def get_artifact_for_prediction(
        self,
        *,
        prediction_id: str,
        artifact_id: str,
        owner_user_id: str,
    ) -> StoredArtifact:
        explanation = await self.get_for_prediction(
            prediction_id=prediction_id,
            owner_user_id=owner_user_id,
        )
        reference = next(
            (artifact for artifact in explanation.artifacts if artifact.artifact_id == artifact_id),
            None,
        )
        if reference is None:
            raise XaiExplanationNotFoundError("XAI artifact was not found.")
        try:
            artifact = self._artifact_store.read(reference)
        except ArtifactStorageError as error:
            logger.warning(
                "voice_xai_artifact_read_failed",
                extra={
                    "prediction_id": prediction_id,
                    "artifact_id": artifact_id,
                    "owner_user_id": owner_user_id,
                },
                exc_info=True,
            )
            raise XaiArtifactUnavailableError(
                "The explanation artifact is unavailable."
            ) from error
        if artifact is None:
            raise XaiExplanationNotFoundError("XAI artifact was not found.")
        return artifact

    async def _run(self, explanation_id: str, bundle: ClassifierInferenceBundle) -> None:
        """Outer failure boundary for the whole background job.

        ``_run_internal`` already contains and reports temporal/semantic/
        report component failures individually. This catches everything
        that happens *outside* those three try/except blocks (status/
        repository calls made directly in ``_run_internal``, e.g. `start_run`
        or `update_run_evidence`) so such a failure can never escape this
        coroutine unlogged. An uninspected exception here would otherwise be
        stored, and silently discarded, on the queue's fire-and-forget
        `concurrent.futures.Future` -- leaving the explanation permanently
        stuck `queued`/`running` with no operator-visible error. It never
        touches classifier prediction persistence.
        """

        try:
            logger.info(
                "voice_xai_run_started",
                extra={
                    "explanation_id": explanation_id,
                    "prediction_id": bundle.prediction_id,
                    "request_id": bundle.request_id,
                },
            )
            await self._run_internal(explanation_id, bundle)
            logger.info(
                "voice_xai_run_finished",
                extra={
                    "explanation_id": explanation_id,
                    "prediction_id": bundle.prediction_id,
                    "request_id": bundle.request_id,
                },
            )
        except Exception:
            logger.exception(
                "Unhandled Voice XAI orchestration failure.",
                extra={
                    "explanation_id": explanation_id,
                    "prediction_id": bundle.prediction_id,
                    "request_id": bundle.request_id,
                },
            )
            try:
                status_service = self._status_service_for_run()
                await status_service.fail_run(
                    explanation_id=explanation_id,
                    owner_user_id=bundle.owner_user_id,
                    error=_error(
                        None,
                        "xai_unexpected_failure",
                        "An unexpected internal error stopped this explanation run.",
                    ),
                )
            except Exception:
                logger.exception(
                    "Could not record the unexpected Voice XAI failure for %s",
                    explanation_id,
                )

    async def _run_internal(
        self, explanation_id: str, bundle: ClassifierInferenceBundle
    ) -> None:
        """Contain component errors in XAI state; never touch classifier persistence."""

        repository = self._worker_repository()
        status_service = XaiStatusService(repository)
        temporal = None
        semantic = None
        temporal_result = None
        semantic_result = None
        deadline = monotonic() + self._settings.xai_job_timeout_seconds
        await status_service.start_run(
            explanation_id=explanation_id, owner_user_id=bundle.owner_user_id
        )
        if bundle.extraction is not None and bundle.extraction.branches:
            await self._store_extraction_artifact(
                repository=repository,
                explanation_id=explanation_id,
                bundle=bundle,
            )
        try:
            await status_service.start_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.temporal,
            )
            self._log_component_event(
                "started", explanation_id, bundle, ExplanationComponent.temporal
            )
            temporal_result = self._temporal_service.analyze(bundle)
            self._raise_if_timed_out(deadline)
            visualization_artifact = self._artifact_store.store_json(
                explanation_id=explanation_id,
                payload=temporal_result.visualization.as_json_dict(),
            )
            await repository.append_artifact(explanation_id, visualization_artifact)
            temporal_artifacts = [visualization_artifact]
            if bundle.processed_audio is not None:
                spectrogram_artifact = self._artifact_store.store_json(
                    explanation_id=explanation_id,
                    kind="attention_spectrogram",
                    payload=mel_spectrogram_payload(
                        bundle.processed_audio,
                        temporal_result.explanation,
                        temporal_result.visualization,
                        n_mels=self._settings.xai_spectrogram_n_mels,
                        n_fft=self._settings.xai_spectrogram_n_fft,
                        hop_length=self._settings.xai_spectrogram_hop_length,
                        max_frames=self._settings.xai_spectrogram_max_frames,
                    ),
                )
                await repository.append_artifact(explanation_id, spectrogram_artifact)
                temporal_artifacts.append(spectrogram_artifact)
            temporal = temporal_result.explanation.model_copy(
                update={"artifacts": temporal_artifacts}
            )
            await status_service.complete_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.temporal,
                result=temporal,
            )
            self._log_component_event(
                "completed", explanation_id, bundle, ExplanationComponent.temporal
            )
        except XaiJobTimedOutError:
            await self._mark_timeout(status_service, explanation_id, bundle.owner_user_id)
            return
        except TemporalAttentionError as error:
            logger.exception(
                "Voice XAI temporal validation failed for %s",
                explanation_id,
                extra={"temporal_error_code": error.code},
            )
            await self._fail_if_mutable(
                status_service,
                explanation_id,
                bundle.owner_user_id,
                ExplanationComponent.temporal,
                code=error.code,
                message="Temporal attention evidence could not be generated.",
            )
        except Exception:
            logger.exception("Voice XAI temporal analysis failed for %s", explanation_id)
            await self._fail_if_mutable(
                status_service, explanation_id, bundle.owner_user_id, ExplanationComponent.temporal
            )

        try:
            await status_service.start_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.semantic,
            )
            self._log_component_event(
                "started", explanation_id, bundle, ExplanationComponent.semantic
            )
            analyze_windows = getattr(self._semantic_service, "analyze_windows", None)
            if bundle.processed_audio is not None and callable(analyze_windows):
                semantic_result = analyze_windows(
                    bundle.processed_audio,
                    request_id=bundle.request_id,
                )
            else:
                semantic_result = self._semantic_service.analyze(bundle)
            self._raise_if_timed_out(deadline)
            semantic = semantic_result.explanation
            await status_service.complete_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.semantic,
                result=semantic,
            )
            self._log_component_event(
                "completed", explanation_id, bundle, ExplanationComponent.semantic
            )
        except XaiJobTimedOutError:
            await self._mark_timeout(status_service, explanation_id, bundle.owner_user_id)
            return
        except Exception:
            logger.exception("Voice XAI semantic analysis failed for %s", explanation_id)
            await self._fail_if_mutable(
                status_service, explanation_id, bundle.owner_user_id, ExplanationComponent.semantic
            )

        quality = quality_for_analysis(temporal, semantic)
        eligibility = evaluate_research_eligibility(
            classifier=classifier_snapshot_from_prediction(bundle.prediction),
            temporal=temporal,
            semantic=semantic,
        )
        if not eligibility.eligible:
            logger.debug(
                "Voice XAI run is not research eligible.",
                extra={
                    "explanation_id": explanation_id,
                    "unmet_requirements": eligibility.unmet_requirements,
                },
            )
        await repository.update_run_evidence(
            explanation_id,
            development_placeholder=(
                temporal is None
                or semantic is None
                or temporal.development_placeholder
                or semantic.development_placeholder
            ),
            research_eligible=eligibility.eligible,
            warnings=[
                *getattr(temporal_result, "warnings", ()),
                *getattr(semantic_result, "warnings", ()),
            ],
            quality=quality,
            provenance=_completed_provenance(
                bundle,
                temporal=temporal,
                semantic=semantic,
                settings=self._settings,
            ),
        )

        try:
            await status_service.start_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.report,
            )
            self._log_component_event(
                "started", explanation_id, bundle, ExplanationComponent.report
            )
            report = self._report_composer.compose(
                classifier=classifier_snapshot_from_prediction(bundle.prediction),
                temporal=temporal,
                semantic=semantic,
                quality=quality,
            )
            await status_service.complete_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.report,
                result=report,
            )
            self._log_component_event(
                "completed", explanation_id, bundle, ExplanationComponent.report
            )
        except Exception:
            logger.exception("Voice XAI report composition failed for %s", explanation_id)
            await self._fail_if_mutable(
                status_service, explanation_id, bundle.owner_user_id, ExplanationComponent.report
            )
            return

        await self._generate_narrative(
            status_service=status_service,
            explanation_id=explanation_id,
            bundle=bundle,
            classifier=classifier_snapshot_from_prediction(bundle.prediction),
            temporal=temporal,
            semantic=semantic,
            quality=quality,
            report=report,
            deadline=deadline,
        )

    async def _generate_narrative(
        self,
        *,
        status_service: XaiStatusService,
        explanation_id: str,
        bundle: ClassifierInferenceBundle,
        classifier: ClassifierSnapshot,
        temporal: TemporalExplanation | None,
        semantic: SemanticExplanation | None,
        quality: ExplanationQuality,
        report: CombinedExplanationReport,
        deadline: float,
    ) -> None:
        """Run optional Qwen narration after the authoritative report exists.

        A narrative failure is deliberately recorded as a non-critical component
        failure. It cannot alter the classifier, temporal/semantic evidence, or
        the completed deterministic report.
        """

        if not self._settings.xai_narrative_enabled:
            await status_service.mark_component_not_available(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.narrative,
                error=_error(
                    ExplanationComponent.narrative,
                    "narrative_disabled",
                    "AI narrative generation is disabled.",
                ),
            )
            return
        remaining_seconds = deadline - monotonic()
        if remaining_seconds <= 0:
            await self._fail_if_mutable(
                status_service,
                explanation_id,
                bundle.owner_user_id,
                ExplanationComponent.narrative,
                code="narrative_timeout",
                message="The AI narrative time budget was exhausted.",
            )
            return
        try:
            await status_service.start_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.narrative,
            )
            self._log_component_event(
                "started", explanation_id, bundle, ExplanationComponent.narrative
            )
            narrative = await self._narrative_service.generate(
                classifier=classifier,
                temporal=temporal,
                semantic=semantic,
                quality=quality,
                report=report,
                timeout_seconds=remaining_seconds,
            )
            await status_service.complete_component(
                explanation_id=explanation_id,
                owner_user_id=bundle.owner_user_id,
                component=ExplanationComponent.narrative,
                result=narrative,
            )
            self._log_component_event(
                "completed", explanation_id, bundle, ExplanationComponent.narrative
            )
        except NarrativeGenerationError as error:
            logger.warning(
                "Voice XAI narrative generation failed for %s: %s",
                explanation_id,
                error.code,
            )
            await self._fail_if_mutable(
                status_service,
                explanation_id,
                bundle.owner_user_id,
                ExplanationComponent.narrative,
                code=error.code,
                message=str(error),
            )
        except Exception:
            logger.exception("Voice XAI narrative generation failed for %s", explanation_id)
            await self._fail_if_mutable(
                status_service,
                explanation_id,
                bundle.owner_user_id,
                ExplanationComponent.narrative,
                code="narrative_generation_failed",
                message="The AI narrative could not be generated.",
            )

    def _worker_repository(self) -> XaiExplanationRepository:
        """Repository bound to whichever loop is actually running this job.

        Defaults to the main-loop repository supplied at construction time,
        matching every test and caller that never starts the queue's
        dedicated worker loop. Real app wiring supplies
        ``worker_repository_factory`` so this instead resolves a repository
        bound to the queue's own dedicated MongoDB client/loop (see
        ``app.database.mongodb.build_dedicated_async_mongo_client`` and
        ``AsynchronousExplanationQueue.worker_database``).
        """

        if self._worker_repository_factory is not None:
            return self._worker_repository_factory()
        return self._repository

    async def _store_extraction_artifact(
        self,
        *,
        repository: XaiExplanationRepository,
        explanation_id: str,
        bundle: ClassifierInferenceBundle,
    ) -> None:
        """Persist bounded capture bytes only from the background XAI worker.

        The classifier has already completed and its result has already been
        persisted when this runs. A storage or serialization error is recorded
        internally and must not alter either the classifier response or the
        rest of the explanation pipeline.
        """

        extraction = bundle.extraction
        if extraction is None or not extraction.branches:
            return
        try:
            reference = self._artifact_store.store_bytes(
                explanation_id=explanation_id,
                content=serialize_extraction_bundle(extraction),
                kind="intermediate_representations",
                content_type=EXTRACTION_ARTIFACT_CONTENT_TYPE,
            )
            await repository.append_artifact(explanation_id, reference)
        except Exception:  # noqa: BLE001 - artifact evidence is best-effort
            logger.exception(
                "voice_xai_extraction_artifact_store_failed",
                extra={
                    "explanation_id": explanation_id,
                    "request_id": bundle.request_id,
                },
            )

    def _status_service_for_run(self) -> XaiStatusService:
        return XaiStatusService(self._worker_repository())

    def _raise_if_timed_out(self, deadline: float) -> None:
        if monotonic() > deadline:
            raise XaiJobTimedOutError

    async def _mark_timeout(
        self, status_service: XaiStatusService, explanation_id: str, owner_user_id: str
    ) -> None:
        statuses = await status_service.component_statuses(
            explanation_id=explanation_id,
            owner_user_id=owner_user_id,
        )
        # A failed report makes the whole run terminal. Fail all other pending
        # components first, so their timeout state is persisted before the
        # report transition closes the run.
        ordered_components = (
            *(component for component in ExplanationComponent if component != ExplanationComponent.report),
            ExplanationComponent.report,
        )
        for component in ordered_components:
            current = getattr(statuses, component.value)
            if current in {
                ComponentStatus.completed,
                ComponentStatus.failed,
                ComponentStatus.not_available,
            }:
                continue
            await self._fail_if_mutable(
                status_service,
                explanation_id,
                owner_user_id,
                component,
                code="xai_job_timeout",
                message="The XAI job exceeded its configured time limit.",
            )

    async def _fail_if_mutable(
        self,
        status_service: XaiStatusService,
        explanation_id: str,
        owner_user_id: str,
        component: ExplanationComponent,
        *,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        try:
            await status_service.fail_component(
                explanation_id=explanation_id,
                owner_user_id=owner_user_id,
                component=component,
                error=_error(
                    component,
                    code or f"{component.value}_analysis_failed",
                    message or "The XAI component could not be completed.",
                ),
            )
            logger.warning(
                "voice_xai_component_failed",
                extra={
                    "explanation_id": explanation_id,
                    "component": component.value,
                    "code": code or f"{component.value}_analysis_failed",
                },
            )
        except Exception:
            logger.exception("Could not record Voice XAI failure for %s", explanation_id)

    @staticmethod
    def _log_component_event(
        event: str,
        explanation_id: str,
        bundle: ClassifierInferenceBundle,
        component: ExplanationComponent,
    ) -> None:
        logger.info(
            f"voice_xai_component_{event}",
            extra={
                "explanation_id": explanation_id,
                "prediction_id": bundle.prediction_id,
                "request_id": bundle.request_id,
                "component": component.value,
            },
        )

    def _require_enabled(self) -> None:
        if not self._settings.xai_enabled or self._settings.xai_mode == "disabled":
            raise XaiOrchestrationError("Voice XAI is disabled.")
        if self._settings.xai_mode not in {"mock", "real"}:
            raise XaiOrchestrationError("The configured Voice XAI mode is not available.")


def classifier_snapshot_from_prediction(prediction: VoicePredictionResponse) -> ClassifierSnapshot:
    branches = []
    glottal_spoof_probability = None
    for branch in prediction.branches:
        canonical = canonical_branch_name(branch.model_name)
        spoof_probability = (
            branch.probabilities.spoof if branch.probabilities is not None else None
        )
        if canonical == "glottal":
            # Auxiliary evidence only -- never folded into spoof_probability/
            # verdict above, and never counted toward contributing branches.
            # See ClassifierAuxiliaryEvidence's docstring.
            glottal_spoof_probability = spoof_probability
        branches.append(
            ClassifierBranchSnapshot(
                branch_name=CanonicalXaiBranch(canonical),
                model_name=branch.model_name,
                status=branch.status,
                mode=branch.mode,
                spoof_probability=spoof_probability,
            )
        )
    fusion = prediction.fusion
    return ClassifierSnapshot(
        verdict=fusion.prediction,
        spoof_probability=(fusion.probabilities.spoof if fusion.probabilities else None),
        bonafide_probability=(fusion.probabilities.bonafide if fusion.probabilities else None),
        confidence=fusion.confidence,
        # Was hardcoded to 0.5 -- now reads the threshold FusionEngine
        # actually applied (frozen research contract when present, see
        # app/utils/fusion.py), so XAI never reports a stale value.
        decision_threshold=fusion.decision_threshold,
        contains_dummy_branches=fusion.contains_dummy_branches,
        research_eligible=fusion.eligible_for_research_evaluation,
        branches=branches,
        auxiliary_evidence=ClassifierAuxiliaryEvidence(
            glottal_spoof_probability=glottal_spoof_probability,
        ),
    )


def inference_bundle_from_prediction_document(document: dict[str, Any]) -> ClassifierInferenceBundle:
    try:
        prediction = VoicePredictionResponse(
            request_id=document["request_id"],
            audio=AudioMetadata(
                original_filename=document.get("original_filename") or "unknown-audio",
                content_type="application/octet-stream",
                original_extension=document.get("original_extension"),
                detected_container=document.get("detected_container"),
                detected_codec=document.get("detected_codec"),
                file_size_bytes=document.get("size_bytes") or 0,
                duration_seconds=document.get("duration_seconds") or 0.01,
                sample_rate=document.get("sample_rate") or 1,
                channels=document.get("channels") or 1,
            ),
            branches=document.get("branches") or [],
            fusion=FusionResult.model_validate(document["fusion"]),
            total_processing_time_ms=document.get("total_processing_time_ms") or 0.0,
            created_at=document.get("created_at") or datetime.now(UTC),
        )
    except (KeyError, ValueError) as error:
        raise XaiOrchestrationError("The completed prediction cannot be used for XAI.") from error
    return ClassifierInferenceBundle(
        prediction_id=document["id"],
        request_id=prediction.request_id,
        owner_user_id=document["owner_user_id"],
        source_type=SourceType(document["source_type"]),
        prediction=prediction,
    )


def _private_temporal_evidence_reference(
    document: dict[str, Any],
) -> ExplanationArtifactReference | None:
    raw = document.get("private_temporal_evidence_artifact")
    if not isinstance(raw, dict):
        return None
    try:
        reference = ExplanationArtifactReference.model_validate(raw)
    except ValueError:
        return None
    if (
        reference.kind != "temporal_evidence"
        or reference.content_type != TEMPORAL_EVIDENCE_ARTIFACT_CONTENT_TYPE
    ):
        return None
    return reference


def _configuration_hash(settings: Settings) -> str:
    payload = _configuration_hash_payload(settings)
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _configuration_hash_inputs() -> list[str]:
    """Return the public names of settings represented by ``configuration_hash``."""
    return list(_CONFIGURATION_HASH_INPUTS)


_CONFIGURATION_HASH_INPUTS = (
    "pipeline_version",
    "mode",
    "semantic_window_duration_seconds",
    "semantic_window_overlap_seconds",
    "temporal_threshold_config_path",
    "temporal_visualization_quantile",
    "temporal_visualization_minimum_density",
    "capture_enabled",
    "capture_max_total_elements",
    "spectrogram_n_mels",
    "spectrogram_n_fft",
    "spectrogram_hop_length",
    "spectrogram_max_frames",
    "narrative_enabled",
    "narrative_model",
    "narrative_prompt_version",
    "narrative_thinking_budget",
)


def _configuration_hash_payload(settings: Settings) -> dict[str, object]:
    return {
        "pipeline_version": settings.xai_pipeline_version,
        "mode": settings.xai_mode,
        "semantic_window_duration_seconds": settings.xai_semantic_window_duration_seconds,
        "semantic_window_overlap_seconds": settings.xai_semantic_window_overlap_seconds,
        "temporal_threshold_config_path": settings.xai_temporal_threshold_config_path,
        "temporal_visualization_quantile": (
            settings.xai_temporal_visualization_quantile
        ),
        "temporal_visualization_minimum_density": (
            settings.xai_temporal_visualization_minimum_density
        ),
        "capture_enabled": settings.xai_capture_enabled,
        "capture_max_total_elements": settings.xai_capture_max_total_elements,
        "spectrogram_n_mels": settings.xai_spectrogram_n_mels,
        "spectrogram_n_fft": settings.xai_spectrogram_n_fft,
        "spectrogram_hop_length": settings.xai_spectrogram_hop_length,
        "spectrogram_max_frames": settings.xai_spectrogram_max_frames,
        "narrative_enabled": settings.xai_narrative_enabled,
        "narrative_model": settings.xai_narrative_model,
        "narrative_prompt_version": "voice-xai-narrative-v1",
        "narrative_thinking_budget": settings.xai_narrative_thinking_budget,
    }
def _completed_provenance(
    bundle: ClassifierInferenceBundle,
    *,
    temporal,
    semantic,
    settings: Settings,
) -> ExplanationProvenance:
    return ExplanationProvenance(
        pipeline_version=settings.xai_pipeline_version,
        classifier_contract_version=bundle.contract_version,
        feature_extractor_version=(
            semantic.extractor_version
            if semantic is not None
            else (
                bundle.extraction.acoustic_features.extractor_version
                if bundle.extraction and bundle.extraction.acoustic_features
                else None
            )
        ),
        semantic_model_version=semantic.model_version if semantic is not None else None,
        temporal_model_version=temporal.model_version if temporal is not None else None,
        configuration_hash=_configuration_hash(settings),
        configuration_hash_inputs=_configuration_hash_inputs(),
    )


def response_from_document(document: dict[str, Any]) -> XaiExplanationResponse:
    fields = XaiExplanationResponse.model_fields
    payload = {name: document[name] for name in fields if name in document}
    payload["explanation_id"] = document["id"]
    return XaiExplanationResponse.model_validate(payload)


def _error(
    component: ExplanationComponent | None, code: str, message: str
) -> ExplanationError:
    return ExplanationError(component=component, code=code, message=message)
