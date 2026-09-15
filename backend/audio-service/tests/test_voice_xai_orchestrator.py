import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.config.settings import Settings
from app.ingestion.audio import ProcessedAudio
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.xai import (
    ComponentStatus,
    ExplanationComponent,
    NarrativeExplanation,
    TemporalExplanation,
    TemporalRegion,
)
from app.schemas.prediction import (
    AudioMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.schemas.xai import ComponentStatus, TemporalExplanation, TemporalRegion
from app.voice_xai.artifacts.service import LocalExplanationArtifactStore
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.orchestrator import (
    VoiceXaiOrchestrator,
    XaiRetryRequiredError,
    _configuration_hash,
    classifier_snapshot_from_prediction,
)
from app.voice_xai.persistence.mongodb import MongoXaiExplanationRepository
from app.voice_xai.narrative.service import NarrativeGenerationError
from app.voice_xai.temporal.contracts import (
    TemporalAnalysisResult,
    TemporalAttentionError,
    TemporalAttentionWindowInput,
    TemporalVisualization,
)
from tests.test_mongodb import FakeDatabase


@pytest.mark.anyio
async def test_windowed_semantic_evidence_and_interval_quality_are_persisted(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_windowed_run"))
    prediction = _prediction()
    bundle = _bundle(prediction, processed_audio=_audio(2.0))
    orchestrator = _orchestrator(repository, settings, tmp_path)
    explanation_id = await _create(repository, bundle, settings)

    await orchestrator._run(explanation_id, bundle)

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert response.semantic is not None
    assert response.semantic.windows
    assert response.semantic.windows[0].feature_importance[0].start_seconds == 0.0
    assert response.quality.temporal_semantic_iou.status.value == "available"
    assert response.development_placeholder is True
    assert response.research_eligible is False
    assert response.provenance.semantic_model_version == response.semantic.model_version
    assert response.provenance.temporal_model_version == response.temporal.model_version
    assert response.temporal.candidate_region_count == 1
    assert {artifact.kind for artifact in response.temporal.artifacts} == {
        "attention_visualization",
        "attention_spectrogram",
    }


@pytest.mark.anyio
async def test_narrative_is_optional_and_cannot_change_deterministic_report(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        xai_enabled=True,
        xai_mode="mock",
        xai_narrative_enabled=True,
        xai_narrative_api_key="test-key",
        xai_narrative_base_url=(
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        ),
    )
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_narrative_complete"))
    prediction = _prediction()
    bundle = _bundle(prediction)
    narrative_service = RecordingNarrativeService()
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=StaticTemporalService(),
        narrative_service=narrative_service,
    )
    explanation_id = await _create(repository, bundle, settings)

    await orchestrator._run(explanation_id, bundle)

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert response.status.value == "completed"
    assert response.combined_report is not None
    assert response.combined_report.finding.startswith("The classifier returned")
    assert response.narrative is not None
    assert response.narrative.summary == "Validated evidence requires qualified human review."
    assert response.component_statuses.narrative == ComponentStatus.completed
    assert narrative_service.received_report == response.combined_report


@pytest.mark.anyio
async def test_narrative_failure_preserves_completed_deterministic_report(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        xai_enabled=True,
        xai_mode="mock",
        xai_narrative_enabled=True,
        xai_narrative_api_key="test-key",
        xai_narrative_base_url=(
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        ),
    )
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_narrative_failure"))
    prediction = _prediction()
    bundle = _bundle(prediction)
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=StaticTemporalService(),
        narrative_service=FailingNarrativeService(),
    )
    explanation_id = await _create(repository, bundle, settings)

    await orchestrator._run(explanation_id, bundle)

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert response.status.value == "completed"
    assert response.combined_report is not None
    assert response.narrative is None
    assert response.component_statuses.narrative == ComponentStatus.failed
    assert response.component_errors.narrative.code == "narrative_invalid_response"
    assert response.error is None


@pytest.mark.anyio
async def test_queue_full_is_recorded_and_retry_creates_a_new_run(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_queue_full"))
    prediction = _prediction()
    bundle = _bundle(prediction)
    queue = FullQueue()
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=queue,
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=StaticTemporalService(),
        replay_service=ReplayService(),
    )

    blocked = await orchestrator.enqueue(bundle)
    assert blocked.status.value == "blocked"
    assert blocked.component_errors.temporal.code == "xai_queue_full"

    first = await orchestrator.trigger_for_prediction(
        prediction_id="prediction-123", owner_user_id="user-123", retry=True
    )
    second = await orchestrator.trigger_for_prediction(
        prediction_id="prediction-123", owner_user_id="user-123", retry=True
    )
    assert first.explanation_id != second.explanation_id


@pytest.mark.anyio
async def test_failed_trigger_requires_retry_and_retry_replays_source_audio(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_retry_lifecycle"))
    prediction = _prediction()
    bundle = _bundle(prediction)
    replay = ReplayService()
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=StaticTemporalService(),
        replay_service=replay,
    )
    explanation_id = await _create(repository, bundle, settings)
    await orchestrator._status.fail_run(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        error=_run_error("xai_run_interrupted"),
    )

    with pytest.raises(XaiRetryRequiredError):
        await orchestrator.trigger_for_prediction(
            prediction_id="prediction-123", owner_user_id="user-123"
        )

    retried = await orchestrator.trigger_for_prediction(
        prediction_id="prediction-123", owner_user_id="user-123", retry=True
    )
    assert retried.explanation_id != explanation_id
    assert replay.calls == 1


@pytest.mark.anyio
async def test_compact_temporal_evidence_is_private_and_reused_by_manual_retry(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_temporal_retry"))
    prediction = _prediction()
    queue = CapturingQueue()
    replay = ReplayService()
    store = LocalExplanationArtifactStore(tmp_path / "private")
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=queue,
        app_settings=settings,
        artifact_store=store,
        temporal_service=StaticTemporalService(),
        replay_service=replay,
    )

    initial = await orchestrator.enqueue(
        _bundle(prediction, temporal_evidence=_temporal_evidence())
    )
    document = await repository.get_explanation_for_owner(
        explanation_id=initial.explanation_id,
        owner_user_id="user-123",
    )
    assert document is not None
    reference = document["private_temporal_evidence_artifact"]
    assert reference["kind"] == "temporal_evidence"
    assert initial.artifacts == []

    await orchestrator._status.fail_run(
        explanation_id=initial.explanation_id,
        owner_user_id="user-123",
        error=_run_error("xai_run_interrupted"),
    )
    retried = await orchestrator.trigger_for_prediction(
        prediction_id="prediction-123",
        owner_user_id="user-123",
        retry=True,
    )

    assert retried.explanation_id != initial.explanation_id
    assert replay.calls == 1
    retry_bundle = queue.bundles[-1]
    assert retry_bundle.temporal_evidence is not None
    recovered = retry_bundle.temporal_evidence[0]
    expected = _temporal_evidence()[0]
    np.testing.assert_allclose(recovered.token_times_seconds, expected.token_times_seconds)
    np.testing.assert_allclose(recovered.attention_density, expected.attention_density)
    assert recovered.spoof_probability == expected.spoof_probability


@pytest.mark.anyio
async def test_timeout_and_component_errors_are_sanitized(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        xai_enabled=True,
        xai_mode="mock",
        xai_job_timeout_seconds=0.001,
    )
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_timeout"))
    prediction = _prediction()
    bundle = _bundle(prediction)
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=SlowTemporalService(),
    )
    explanation_id = await _create(repository, bundle, settings)

    await orchestrator._run(explanation_id, bundle)

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert response.status.value == "failed"
    assert response.error.code == "xai_job_timeout"
    assert "C:\\" not in response.error.message


@pytest.mark.anyio
async def test_timeout_preserves_a_completed_component(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_timeout_completed"))
    prediction = _prediction()
    bundle = _bundle(prediction)
    orchestrator = _orchestrator(repository, settings, tmp_path)
    explanation_id = await _create(repository, bundle, settings)
    temporal_result = StaticTemporalService().analyze(bundle)

    await orchestrator._status.complete_component(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.temporal,
        result=temporal_result.explanation,
    )
    await orchestrator._mark_timeout(
        orchestrator._status,
        explanation_id,
        "user-123",
    )

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert response.status.value == "failed"
    assert response.component_statuses.temporal == ComponentStatus.completed
    assert response.component_statuses.semantic == ComponentStatus.failed
    assert response.component_statuses.report == ComponentStatus.failed


@pytest.mark.anyio
async def test_unexpected_failure_outside_component_handlers_reaches_terminal_status(
    tmp_path: Path,
) -> None:
    """SEC-2 regression test.

    ``update_run_evidence`` runs between the semantic and report try/except
    blocks in ``_run_internal`` -- outside all three per-component error
    handlers. Before the outer failure boundary was added, an exception
    there propagated out of ``_run`` uncaught and was silently discarded by
    the queue's fire-and-forget Future, leaving the run stuck ``running``
    forever with no recorded error. It must now reach a terminal ``failed``
    status with a safe, logged error instead.
    """

    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    repository = RaisingUpdateRunEvidenceRepository(
        MongoXaiExplanationRepository(FakeDatabase("xai_unexpected_failure"))
    )
    prediction = _prediction()
    bundle = _bundle(prediction)
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=StaticTemporalService(),
    )
    explanation_id = await _create(repository, bundle, settings)

    # Must not raise: the outer boundary in _run() must contain this.
    await orchestrator._run(explanation_id, bundle)

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert response.status.value == "failed"
    assert response.error.code == "xai_unexpected_failure"
    assert response.error.component is None
    assert "boom" not in response.model_dump_json()
    assert "Traceback" not in response.model_dump_json()


@pytest.mark.anyio
async def test_internal_semantic_paths_and_stack_details_do_not_reach_response(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_sanitized_error"))
    prediction = _prediction()
    bundle = _bundle(prediction)
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=StaticTemporalService(),
        semantic_service=FailingSemanticService(),
    )
    explanation_id = await _create(repository, bundle, settings)

    await orchestrator._run(explanation_id, bundle)

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert response.component_errors.semantic.code == "semantic_analysis_failed"
    assert "C:\\research" not in response.model_dump_json()
    assert "traceback" not in response.model_dump_json().lower()


@pytest.mark.anyio
async def test_temporal_validation_failure_has_a_safe_specific_error_code(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_temporal_error"))
    prediction = _prediction()
    bundle = _bundle(prediction)
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(prediction)),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=FailingTemporalService(),
    )
    explanation_id = await _create(repository, bundle, settings)

    await orchestrator._run(explanation_id, bundle)

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert response.component_errors.temporal.code == "xai_temporal_window_invalid"
    assert "C:\\research" not in response.model_dump_json()
    assert response.component_errors.temporal.message == (
        "Temporal attention evidence could not be generated."
    )


def test_classifier_snapshot_marks_glottal_primary_for_v3_contributor() -> None:
    prediction = _prediction_with_glottal(
        fusion_mode="learned_constrained",
        contributing_branches=[
            "cnn_acoustic",
            "aasist",
            "ssl_wavlm_xlsr",
            "glottal_features",
        ],
    )

    snapshot = classifier_snapshot_from_prediction(prediction)

    assert snapshot.auxiliary_evidence.glottal_spoof_probability == pytest.approx(0.7)
    assert snapshot.auxiliary_evidence.used_for_primary_decision is True


def test_classifier_snapshot_marks_glottal_auxiliary_for_legacy_fusion() -> None:
    prediction = _prediction_with_glottal(
        fusion_mode="legacy_average",
        contributing_branches=["cnn_acoustic", "aasist", "ssl_wavlm_xlsr"],
    )

    snapshot = classifier_snapshot_from_prediction(prediction)

    assert snapshot.auxiliary_evidence.glottal_spoof_probability == pytest.approx(0.7)
    assert snapshot.auxiliary_evidence.used_for_primary_decision is False


def test_classifier_snapshot_represents_failed_glottal_without_primary_use() -> None:
    prediction = _prediction_with_glottal(
        fusion_mode="learned_constrained",
        contributing_branches=["cnn_acoustic", "aasist", "ssl_wavlm_xlsr"],
        glottal_status=BranchStatus.failed,
        glottal_probability=None,
    )

    snapshot = classifier_snapshot_from_prediction(prediction)
    glottal = next(
        branch for branch in snapshot.branches if branch.branch_name.value == "glottal"
    )

    assert glottal.status == BranchStatus.failed
    assert glottal.spoof_probability is None
    assert snapshot.auxiliary_evidence.glottal_spoof_probability is None
    assert snapshot.auxiliary_evidence.used_for_primary_decision is False


def test_classifier_snapshot_represents_disabled_glottal_without_primary_use() -> None:
    prediction = _prediction_with_glottal(
        fusion_mode="learned_constrained",
        contributing_branches=["cnn_acoustic", "aasist", "ssl_wavlm_xlsr"],
        glottal_status=BranchStatus.failed,
        glottal_mode=ModelMode.disabled,
        glottal_probability=None,
    )

    snapshot = classifier_snapshot_from_prediction(prediction)
    glottal = next(
        branch for branch in snapshot.branches if branch.branch_name.value == "glottal"
    )

    assert glottal.mode == ModelMode.disabled
    assert glottal.spoof_probability is None
    assert snapshot.auxiliary_evidence.used_for_primary_decision is False


def test_xai_configuration_hash_is_stable_for_same_settings() -> None:
    settings = Settings(_env_file=None, xai_mode="mock")

    assert _configuration_hash(settings) == _configuration_hash(settings)


def test_xai_configuration_hash_changes_for_behavioral_setting() -> None:
    original = Settings(_env_file=None, xai_mode="mock", xai_temporal_smoothing_ms=80)
    changed = Settings(_env_file=None, xai_mode="mock", xai_temporal_smoothing_ms=120)

    assert _configuration_hash(original) != _configuration_hash(changed)


def test_xai_configuration_hash_ignores_filesystem_relocation(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "nested" / "second.json"
    second.parent.mkdir()
    first.write_text('{"same": true}', encoding="utf-8")
    second.write_text('{"same": true}', encoding="utf-8")
    original = Settings(
        _env_file=None,
        xai_mode="mock",
        xai_semantic_manifest_path=str(first),
    )
    relocated = Settings(
        _env_file=None,
        xai_mode="mock",
        xai_semantic_manifest_path=str(second),
    )

    assert _configuration_hash(original) == _configuration_hash(relocated)


class StaticTemporalService:
    def analyze(self, inference):
        duration = inference.prediction.audio.duration_seconds
        explanation = TemporalExplanation(
            status=ComponentStatus.completed,
            method_version="test-xlsr-attention-v1",
            model_version="test-xlsr",
            development_placeholder=False,
            research_eligible=False,
            attention_score_peak=1.2,
            attention_threshold=1.1,
            regions=[
                TemporalRegion(
                    region_id=1,
                    start_seconds=0.0,
                    end_seconds=duration,
                    attention_score=1.2,
                )
            ],
        )
        return TemporalAnalysisResult(
            explanation=explanation,
            visualization=TemporalVisualization(
                time_seconds=np.array([duration / 2], dtype=np.float32),
                attention_scores=np.array([1.2], dtype=np.float32),
                threshold=1.1,
                high_attention_mask=np.array([True]),
            ),
            development_placeholder=False,
            research_eligible=False,
            warnings=(),
        )


class SlowTemporalService(StaticTemporalService):
    def analyze(self, inference):
        time.sleep(0.01)
        return super().analyze(inference)


class FailingSemanticService:
    def analyze(self, inference):
        raise RuntimeError(r"C:\research\private\semantic-model.json traceback details")


class FailingTemporalService:
    def analyze(self, inference):
        raise TemporalAttentionError(
            r"C:\research\private\attention.npy is malformed.",
            code="xai_temporal_window_invalid",
        )


class RecordingNarrativeService:
    def __init__(self) -> None:
        self.received_report = None

    async def generate(self, *, report, **_kwargs):
        self.received_report = report
        return NarrativeExplanation(
            provider="alibaba_model_studio",
            model_id="qwen-test-snapshot",
            prompt_version="voice-xai-narrative-v1",
            input_sha256="0" * 64,
            summary="Validated evidence requires qualified human review.",
            detailed_explanation="The deterministic report remains authoritative.",
            evidence_references=["classifier:verdict"],
        )


class FailingNarrativeService:
    async def generate(self, **_kwargs):
        raise NarrativeGenerationError(
            "narrative_invalid_response",
            "The AI narrative service returned an invalid structured response.",
        )


class RaisingUpdateRunEvidenceRepository:
    """Delegates to a real repository except for a call outside the normal
    per-component try/except blocks, to exercise the outer failure boundary."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def update_run_evidence(self, *args, **kwargs):
        raise RuntimeError("boom")


class FullQueue:
    def submit(self, bundle, task):
        return None

    def close(self, *, wait=True):
        return None


class PausedQueue:
    def submit(self, bundle, task):
        return object()

    def close(self, *, wait=True):
        return None


class CapturingQueue(PausedQueue):
    def __init__(self) -> None:
        self.bundles = []

    def submit(self, bundle, task):
        self.bundles.append(bundle)
        return object()


class ReplayService:
    def __init__(self) -> None:
        self.calls = 0

    async def load_processed_audio(self, *, prediction_document, owner_user_id):
        self.calls += 1
        return _audio(2.0)


class PredictionLookup:
    def __init__(self, document):
        self.document = document

    async def get_prediction_for_owner(self, *, prediction_id, owner_user_id, include_deleted=False):
        return self.document if prediction_id == "prediction-123" and owner_user_id == "user-123" else None


def _orchestrator(repository, settings, tmp_path):
    return VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionLookup(_document(_prediction())),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        temporal_service=StaticTemporalService(),
        replay_service=ReplayService(),
    )


async def _create(repository, bundle, settings):
    return await repository.create_explanation(
        prediction_id=bundle.prediction_id,
        request_id=bundle.request_id,
        owner_user_id=bundle.owner_user_id,
        source_type=bundle.source_type,
        classifier_snapshot=classifier_snapshot_from_prediction(bundle.prediction),
        pipeline_version=settings.xai_pipeline_version,
    )


def _bundle(prediction, *, processed_audio=None, temporal_evidence=None):
    return ClassifierInferenceBundle(
        prediction_id="prediction-123",
        request_id=prediction.request_id,
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
        prediction=prediction,
        processed_audio=processed_audio,
        temporal_evidence=temporal_evidence,
    )


def _temporal_evidence():
    return (
        TemporalAttentionWindowInput(
            start_seconds=0.0,
            end_seconds=2.0,
            token_times_seconds=np.array([0.5, 1.0, 1.5], dtype=np.float32),
            attention_density=np.array([0.8, 1.4, 0.9], dtype=np.float32),
            spoof_probability=0.8,
        ),
    )


def _prediction():
    now = datetime.now(UTC)
    probabilities = ProbabilityScores(bonafide=0.2, spoof=0.8)
    return VoicePredictionResponse(
        request_id="request-123",
        audio=AudioMetadata(original_filename="sample.wav", content_type="audio/wav", file_size_bytes=100, duration_seconds=2.0, sample_rate=16000, channels=1),
        branches=[BranchPrediction(model_name="cnn_acoustic", display_name="CNN", status=BranchStatus.success, mode=ModelMode.dummy, prediction=PredictionLabel.spoof, confidence=0.8, probabilities=probabilities, processing_time_ms=1)],
        fusion=FusionResult(status=BranchStatus.success, prediction=PredictionLabel.spoof, confidence=0.8, probabilities=probabilities, method="weighted_average", contains_dummy_branches=True, eligible_for_research_evaluation=False),
        total_processing_time_ms=1,
        created_at=now,
    )


def _prediction_with_glottal(
    *,
    fusion_mode: str,
    contributing_branches: list[str],
    glottal_status: BranchStatus = BranchStatus.success,
    glottal_mode: ModelMode = ModelMode.real,
    glottal_probability: float | None = 0.7,
):
    prediction = _prediction()
    glottal_probabilities = (
        ProbabilityScores(bonafide=1.0 - glottal_probability, spoof=glottal_probability)
        if glottal_probability is not None
        else None
    )
    prediction.branches.append(
        BranchPrediction(
            model_name="glottal_features",
            display_name="Glottal",
            status=glottal_status,
            mode=glottal_mode,
            prediction=(
                PredictionLabel.spoof
                if glottal_probability is not None and glottal_probability >= 0.5
                else None
            ),
            confidence=glottal_probability,
            probabilities=glottal_probabilities,
            processing_time_ms=1,
        )
    )
    prediction.fusion = prediction.fusion.model_copy(
        update={
            "fusion_mode": fusion_mode,
            "contributing_branches": contributing_branches,
        }
    )
    return prediction


def _document(prediction):
    return {"id": "prediction-123", "request_id": prediction.request_id, "owner_user_id": "user-123", "source_type": SourceType.dashboard_upload.value, "status": "completed", "original_filename": "sample.wav", "size_bytes": 100, "duration_seconds": 2.0, "sample_rate": 16000, "channels": 1, "branches": [branch.model_dump(mode="python") for branch in prediction.branches], "fusion": prediction.fusion.model_dump(mode="python"), "total_processing_time_ms": 1, "created_at": prediction.created_at}


def _run_error(code: str):
    from app.schemas.xai import ExplanationError

    return ExplanationError(component=None, code=code, message="The run stopped.")


def _audio(seconds):
    sample_rate = 16000
    samples = int(seconds * sample_rate)
    waveform = np.sin(np.arange(samples, dtype=np.float32) * 0.07).astype(np.float32)
    return ProcessedAudio(waveform=waveform, sample_rate=sample_rate, original_sample_rate=sample_rate, original_channels=1, duration_seconds=seconds, was_resampled=False, was_converted_to_mono=False, normalisation_applied=True, peak_amplitude=1.0, rms_energy=float(np.sqrt(np.mean(waveform**2))))
