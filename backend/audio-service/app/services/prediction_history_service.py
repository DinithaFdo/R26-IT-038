import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status

from app.auth.schemas import AuthPrincipal
from app.repositories.protocols import PredictionRepository
from app.schemas.common import ModelMode, PredictionLabel, PredictionStatus, SourceType
from app.schemas.prediction import BranchPrediction, FusionResult
from app.schemas.prediction_history import (
    PredictionAudioPlaybackResponse,
    PredictionDeleteResponse,
    PredictionDetailResponse,
    PredictionHistoryAudioMetadata,
    PredictionHistoryItem,
    PredictionHistoryListResponse,
    PredictionModeSummary,
)
from app.storage.protocols import AudioStorage
from app.voice_xai.artifacts.protocols import ExplanationArtifactStore
from app.voice_xai.persistence.protocols import XaiExplanationRepository

logger = logging.getLogger(__name__)

PLAYBACK_URL_EXPIRES_IN_SECONDS = 300


class PredictionHistoryService:
    def __init__(
        self,
        *,
        repository: PredictionRepository,
        storage: AudioStorage,
        xai_repository: XaiExplanationRepository | None = None,
        xai_artifact_store: ExplanationArtifactStore | None = None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._xai_repository = xai_repository
        self._xai_artifact_store = xai_artifact_store

    async def list_predictions(
        self,
        *,
        principal: AuthPrincipal,
        page: int,
        limit: int,
        status_filter: PredictionStatus | None = None,
        source_type: SourceType | None = None,
        prediction_label: PredictionLabel | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> PredictionHistoryListResponse:
        _validate_date_range(created_from, created_to)
        if status_filter == PredictionStatus.deleted:
            return PredictionHistoryListResponse(
                items=[],
                page=page,
                limit=limit,
                has_next=False,
            )

        documents = await self._repository.list_predictions_for_owner(
            owner_user_id=_owner_user_id(principal),
            page=page,
            limit=limit,
            status=status_filter,
            source_type=source_type,
            prediction_label=prediction_label,
            created_from=created_from,
            created_to=created_to,
        )
        return PredictionHistoryListResponse(
            items=[_history_item(document) for document in documents[:limit]],
            page=page,
            limit=limit,
            has_next=len(documents) > limit,
        )

    async def get_prediction_detail(
        self,
        *,
        principal: AuthPrincipal,
        prediction_id: str,
    ) -> PredictionDetailResponse:
        document = await self._get_visible_prediction(principal, prediction_id)
        return _detail_response(document)

    async def get_audio_playback_url(
        self,
        *,
        principal: AuthPrincipal,
        prediction_id: str,
    ) -> PredictionAudioPlaybackResponse:
        document = await self._get_visible_prediction(principal, prediction_id)
        public_id = _cloudinary_public_id(document)
        if public_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction audio is unavailable.",
            )
        try:
            playback_url = await self._storage.generate_signed_playback_url(
                public_id,
                owner_user_id=_owner_user_id(principal),
                expires_in_seconds=PLAYBACK_URL_EXPIRES_IN_SECONDS,
            )
        except Exception:
            logger.exception(
                "Prediction audio playback URL generation failed.",
                extra={
                    "prediction_id": prediction_id,
                    "owner_user_id": _owner_user_id(principal),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction audio playback is unavailable.",
            )
        return PredictionAudioPlaybackResponse(
            playback_url=playback_url,
            expires_in_seconds=PLAYBACK_URL_EXPIRES_IN_SECONDS,
        )

    async def delete_prediction(
        self,
        *,
        principal: AuthPrincipal,
        prediction_id: str,
    ) -> PredictionDeleteResponse:
        document = await self._repository.get_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=_owner_user_id(principal),
            include_deleted=True,
        )
        if document is None:
            _raise_not_found()
        already_deleted = _prediction_status(document) == PredictionStatus.deleted
        if not already_deleted and _prediction_status(document) != PredictionStatus.deleting:
            await self._repository.update_prediction_status(
                str(document["request_id"]),
                PredictionStatus.deleting,
            )

        public_id = _cloudinary_public_id(document)
        if not already_deleted and public_id is not None:
            try:
                await self._storage.delete_audio(public_id)
            except Exception:
                logger.exception(
                    "Prediction audio deletion failed.",
                    extra={
                        "prediction_id": prediction_id,
                        "owner_user_id": _owner_user_id(principal),
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Prediction audio deletion is unavailable.",
                )

        await self._delete_prediction_explanations(
            prediction_id=prediction_id,
            owner_user_id=_owner_user_id(principal),
        )
        if already_deleted:
            return PredictionDeleteResponse(
                prediction_id=prediction_id,
                status=PredictionStatus.deleted,
            )

        deleted = await self._repository.soft_delete_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=_owner_user_id(principal),
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction deletion could not be persisted.",
            )
        return PredictionDeleteResponse(
            prediction_id=prediction_id,
            status=PredictionStatus.deleted,
        )

    async def _delete_prediction_explanations(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> None:
        """Remove owner-scoped XAI evidence before removing its prediction."""
        if self._xai_repository is None or self._xai_artifact_store is None:
            return
        try:
            explanations = await self._xai_repository.list_explanations_for_prediction_for_owner(
                prediction_id=prediction_id,
                owner_user_id=owner_user_id,
            )
            for explanation in explanations:
                explanation_id = explanation.get("id")
                if not isinstance(explanation_id, str) or not explanation_id.strip():
                    raise ValueError("XAI explanation record has no valid identifier.")
                self._xai_artifact_store.delete_explanation_artifacts(explanation_id)
            await self._xai_repository.delete_explanations_for_prediction_for_owner(
                prediction_id=prediction_id,
                owner_user_id=owner_user_id,
            )
        except Exception:
            logger.exception(
                "Prediction explanation deletion failed.",
                extra={
                    "prediction_id": prediction_id,
                    "owner_user_id": owner_user_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction explanation deletion is unavailable.",
            )

    async def _get_visible_prediction(
        self,
        principal: AuthPrincipal,
        prediction_id: str,
    ) -> dict[str, Any]:
        document = await self._repository.get_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=_owner_user_id(principal),
        )
        if document is None:
            _raise_not_found()
        return document


def _history_item(document: dict[str, Any]) -> PredictionHistoryItem:
    fusion = _document_fusion(document)
    return PredictionHistoryItem(
        prediction_id=str(document["id"]),
        filename=document.get("original_filename"),
        source_type=_source_type(document),
        status=_prediction_status(document),
        duration_seconds=document.get("duration_seconds"),
        final_prediction=_prediction_label(fusion.get("prediction")),
        confidence=fusion.get("confidence"),
        mode_summary=_mode_summary(document),
        created_at=_datetime_value(document.get("created_at")),
    )


def _detail_response(document: dict[str, Any]) -> PredictionDetailResponse:
    fusion = _fusion_result(document)
    warnings = _warnings(document, fusion)
    return PredictionDetailResponse(
        prediction_id=str(document["id"]),
        request_id=str(document["request_id"]),
        source_type=_source_type(document),
        status=_prediction_status(document),
        audio=PredictionHistoryAudioMetadata(
            original_filename=document.get("original_filename"),
            original_extension=document.get("original_extension"),
            detected_container=document.get("detected_container"),
            detected_codec=document.get("detected_codec"),
            duration_seconds=document.get("duration_seconds"),
            sample_rate=document.get("sample_rate"),
            channels=document.get("channels"),
            size_bytes=document.get("size_bytes"),
            storage_status=document.get("storage_status"),
            playback_available=_cloudinary_public_id(document) is not None,
        ),
        branches=_branch_predictions(document),
        fusion=fusion,
        preprocessing=document.get("preprocessing") or {},
        total_processing_time_ms=document.get("total_processing_time_ms"),
        warnings=warnings,
        research_eligible=bool(document.get("research_eligible", False)),
        created_at=_datetime_value(document.get("created_at")),
        updated_at=_datetime_value(document.get("updated_at")),
        completed_at=_optional_datetime(document.get("completed_at")),
    )


def _branch_predictions(document: dict[str, Any]) -> list[BranchPrediction]:
    branches = document.get("branches") or []
    return [BranchPrediction.model_validate(branch) for branch in branches]


def _fusion_result(document: dict[str, Any]) -> FusionResult | None:
    fusion = document.get("fusion")
    if not fusion:
        return None
    return FusionResult.model_validate(fusion)


def _mode_summary(document: dict[str, Any]) -> PredictionModeSummary:
    dummy = 0
    real = 0
    for branch in document.get("branches") or []:
        mode = branch.get("mode")
        if mode == ModelMode.dummy or mode == ModelMode.dummy.value:
            dummy += 1
        elif mode == ModelMode.real or mode == ModelMode.real.value:
            real += 1
    return PredictionModeSummary(
        dummy=dummy,
        real=real,
        contains_dummy=dummy > 0,
    )


def _warnings(
    document: dict[str, Any],
    fusion: FusionResult | None,
) -> list[str]:
    warnings = []
    if fusion is not None and fusion.warning:
        warnings.append(fusion.warning)
    if _mode_summary(document).contains_dummy:
        warnings.append(
            "Dummy branch outputs are development placeholders, not research results."
        )
    if not bool(document.get("research_eligible", False)):
        warnings.append("This prediction is not eligible for research evaluation.")
    return list(dict.fromkeys(warnings))


def _cloudinary_public_id(document: dict[str, Any]) -> str | None:
    cloudinary_asset = document.get("cloudinary_asset") or {}
    public_id = cloudinary_asset.get("public_id")
    return public_id if isinstance(public_id, str) and public_id.strip() else None


def _document_fusion(document: dict[str, Any]) -> dict[str, Any]:
    fusion = document.get("fusion")
    return fusion if isinstance(fusion, dict) else {}


def _prediction_label(value: Any) -> PredictionLabel | None:
    if value is None:
        return None
    return PredictionLabel(value)


def _prediction_status(document: dict[str, Any]) -> PredictionStatus:
    return PredictionStatus(document["status"])


def _source_type(document: dict[str, Any]) -> SourceType:
    return SourceType(document["source_type"])


def _datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    return datetime.now(UTC)


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return _datetime_value(value)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validate_date_range(
    created_from: datetime | None,
    created_to: datetime | None,
) -> None:
    if created_from is None or created_to is None:
        return
    if _ensure_utc(created_from) > _ensure_utc(created_to):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="created_from must be before or equal to created_to.",
        )


def _owner_user_id(principal: AuthPrincipal) -> str:
    return principal.user_id or principal.subject


def _raise_not_found() -> None:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Prediction was not found.",
    )
