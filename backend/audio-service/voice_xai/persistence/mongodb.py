"""MongoDB persistence for independent, versioned Voice XAI runs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from app.database.collections import XAI_EXPLANATIONS_COLLECTION
from app.database.mongodb import get_mongodb_database
from app.schemas.common import SourceType
from app.schemas.xai import (
    XAI_SCHEMA_VERSION,
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationComponent,
    ExplanationArtifactReference,
    ExplanationQuality,
    ExplanationComponentStatuses,
    ExplanationError,
    ExplanationProvenance,
    ExplanationStatus,
    NarrativeExplanation,
    SemanticExplanation,
    TemporalExplanation,
)
from app.voice_xai.contracts import CLASSIFIER_XAI_CONTRACT_VERSION
from app.voice_xai.persistence.protocols import ExplanationComponentResult

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None


VALID_EXPLANATION_STATUS_TRANSITIONS: dict[
    ExplanationStatus,
    set[ExplanationStatus],
] = {
    ExplanationStatus.queued: {
        ExplanationStatus.running,
        ExplanationStatus.blocked,
        ExplanationStatus.failed,
    },
    ExplanationStatus.running: {
        ExplanationStatus.partial,
        ExplanationStatus.completed,
        ExplanationStatus.blocked,
        ExplanationStatus.failed,
    },
    ExplanationStatus.partial: {
        ExplanationStatus.running,
        ExplanationStatus.completed,
        ExplanationStatus.blocked,
        ExplanationStatus.failed,
    },
    ExplanationStatus.blocked: {
        ExplanationStatus.queued,
        ExplanationStatus.failed,
    },
    ExplanationStatus.completed: set(),
    ExplanationStatus.failed: set(),
}

VALID_COMPONENT_STATUS_TRANSITIONS: dict[ComponentStatus, set[ComponentStatus]] = {
    ComponentStatus.queued: {
        ComponentStatus.running,
        ComponentStatus.blocked,
        ComponentStatus.failed,
        ComponentStatus.not_available,
    },
    ComponentStatus.running: {
        ComponentStatus.completed,
        ComponentStatus.blocked,
        ComponentStatus.failed,
        ComponentStatus.not_available,
    },
    ComponentStatus.blocked: {
        ComponentStatus.queued,
        ComponentStatus.failed,
        ComponentStatus.not_available,
    },
    ComponentStatus.completed: set(),
    ComponentStatus.failed: set(),
    ComponentStatus.not_available: set(),
}

_COMPONENT_RESULT_TYPES: dict[ExplanationComponent, type] = {
    ExplanationComponent.temporal: TemporalExplanation,
    ExplanationComponent.semantic: SemanticExplanation,
    ExplanationComponent.report: CombinedExplanationReport,
    ExplanationComponent.narrative: NarrativeExplanation,
}

_COMPONENT_DOCUMENT_FIELDS: dict[ExplanationComponent, str] = {
    ExplanationComponent.temporal: "temporal",
    ExplanationComponent.semantic: "semantic",
    ExplanationComponent.report: "combined_report",
    ExplanationComponent.narrative: "narrative",
}


class XaiPersistenceConsistencyError(RuntimeError):
    """Raised when an XAI persistence operation violates its state contract."""


class MongoXaiExplanationRepository:
    def __init__(self, database: Any | None = None) -> None:
        self._database = database

    @property
    def collection(self):
        # PyMongo AsyncDatabase deliberately rejects boolean coercion. The XAI
        # worker injects its own loop-bound AsyncDatabase, so select the
        # fallback explicitly rather than using ``or``.
        database = (
            self._database
            if self._database is not None
            else get_mongodb_database()
        )
        return database[XAI_EXPLANATIONS_COLLECTION]

    async def create_explanation(
        self,
        *,
        prediction_id: str,
        request_id: str,
        owner_user_id: str,
        source_type: SourceType,
        classifier_snapshot: ClassifierSnapshot,
        pipeline_version: str,
        provenance: ExplanationProvenance | None = None,
    ) -> str:
        _require_identifier("prediction_id", prediction_id)
        _require_identifier("request_id", request_id)
        _require_identifier("owner_user_id", owner_user_id)
        _require_identifier("pipeline_version", pipeline_version)

        explanation_id = str(uuid4())
        now = _utc_now()
        explanation_provenance = provenance or ExplanationProvenance(
            pipeline_version=pipeline_version,
            classifier_contract_version=CLASSIFIER_XAI_CONTRACT_VERSION,
        )
        if explanation_provenance.pipeline_version != pipeline_version:
            raise ValueError("Provenance pipeline_version must match the XAI run.")
        if explanation_provenance.schema_version != XAI_SCHEMA_VERSION:
            raise ValueError("Unsupported XAI schema version in provenance.")
        if (
            explanation_provenance.classifier_contract_version
            != CLASSIFIER_XAI_CONTRACT_VERSION
        ):
            raise ValueError("Unsupported classifier-to-XAI contract version.")

        # New runs queue narration explicitly. The schema's not_available
        # default exists only to preserve reads of immutable legacy documents
        # that predate this optional component.
        component_statuses = ExplanationComponentStatuses(
            narrative=ComponentStatus.queued
        )
        document = {
            "id": explanation_id,
            "prediction_id": prediction_id,
            "request_id": request_id,
            "owner_user_id": owner_user_id,
            "source_type": source_type.value,
            "schema_version": XAI_SCHEMA_VERSION,
            "pipeline_version": pipeline_version,
            "development_placeholder": True,
            "research_eligible": False,
            "status": ExplanationStatus.queued.value,
            "component_statuses": _model_document(component_statuses),
            "component_errors": {
                "temporal": None,
                "semantic": None,
                "report": None,
                "narrative": None,
            },
            "classifier_snapshot": _model_document(classifier_snapshot),
            "temporal": None,
            "semantic": None,
            "combined_report": None,
            "narrative": None,
            # Narrative retries are independent from the immutable deterministic
            # evidence. The counter is scoped to this explanation run, not to
            # a user or prediction globally.
            "narrative_retry_count": 0,
            "narrative_retry_history": [],
            "quality": {},
            "provenance": _model_document(explanation_provenance),
            "artifacts": [],
            "warnings": [],
            "error": None,
            "status_history": [
                _status_history_entry(ExplanationStatus.queued, now),
            ],
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        await self.collection.insert_one(document)
        return explanation_id

    async def get_explanation_for_owner(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
    ) -> dict[str, Any] | None:
        document = await self.collection.find_one(
            {"id": explanation_id, "owner_user_id": owner_user_id}
        )
        return _serialize_document(document)

    async def get_latest_explanation_for_prediction(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> dict[str, Any] | None:
        cursor = self.collection.find(
            {"prediction_id": prediction_id, "owner_user_id": owner_user_id}
        ).sort("created_at", -1).limit(1)
        documents = await cursor.to_list(length=1)
        return _serialize_document(documents[0]) if documents else None

    async def list_explanations_for_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> list[dict[str, Any]]:
        cursor = self.collection.find(
            {"prediction_id": prediction_id, "owner_user_id": owner_user_id}
        ).sort("created_at", -1)
        documents = await cursor.to_list(length=None)
        return [_serialize_document(document) for document in documents]

    async def delete_explanations_for_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> int:
        result = await self.collection.delete_many(
            {"prediction_id": prediction_id, "owner_user_id": owner_user_id}
        )
        return int(result.deleted_count)

    async def transition_status(
        self,
        explanation_id: str,
        status: ExplanationStatus,
        *,
        error: ExplanationError | None = None,
    ) -> None:
        document = await self.collection.find_one({"id": explanation_id})
        if document is None:
            raise XaiPersistenceConsistencyError(
                "XAI explanation does not exist for status transition."
            )
        current_status = ExplanationStatus(document["status"])
        _validate_explanation_status_transition(current_status, status)
        if status == ExplanationStatus.failed and error is None:
            raise ValueError("Failed XAI explanations require a safe error.")

        now = _utc_now()
        set_fields: dict[str, Any] = {
            "status": status.value,
            "updated_at": now,
            "error": _model_document(error) if error is not None else None,
        }
        if status in {ExplanationStatus.completed, ExplanationStatus.failed}:
            set_fields["completed_at"] = now

        result = await self.collection.update_one(
            {"id": explanation_id, "status": current_status.value},
            {
                "$set": set_fields,
                "$push": {"status_history": _status_history_entry(status, now)},
            },
        )
        _require_matched_one(result, "transition_status")

    async def update_terminal_error(
        self,
        explanation_id: str,
        *,
        error: ExplanationError,
    ) -> None:
        result = await self.collection.update_one(
            {"id": explanation_id, "status": ExplanationStatus.failed.value},
            {"$set": {"error": _model_document(error), "updated_at": _utc_now()}},
        )
        _require_matched_one(result, "update_terminal_error")

    async def update_component_status(
        self,
        explanation_id: str,
        component: ExplanationComponent,
        status: ComponentStatus,
        *,
        error: ExplanationError | None = None,
    ) -> None:
        document = await self._mutable_document(explanation_id)
        current_status = ComponentStatus(
            document["component_statuses"][component.value]
        )
        _validate_component_status_transition(current_status, status)
        failure_like = status in {
            ComponentStatus.failed,
            ComponentStatus.blocked,
            ComponentStatus.not_available,
        }
        if failure_like and error is None:
            raise ValueError(
                f"Component status {status.value} requires a safe error."
            )
        if not failure_like and error is not None:
            raise ValueError(
                f"Component status {status.value} cannot include an error."
            )
        if (
            error is not None
            and error.component is not None
            and error.component != component
        ):
            raise ValueError("Component error must identify the updated component.")
        result = await self.collection.update_one(
            {
                "id": explanation_id,
                f"component_statuses.{component.value}": current_status.value,
            },
            {
                "$set": {
                    f"component_statuses.{component.value}": status.value,
                    f"component_errors.{component.value}": (
                        _model_document(error) if error is not None else None
                    ),
                    "updated_at": _utc_now(),
                }
            },
        )
        _require_matched_one(result, "update_component_status")

    async def save_component_result(
        self,
        explanation_id: str,
        component: ExplanationComponent,
        result: ExplanationComponentResult,
    ) -> None:
        document = await self._mutable_document(explanation_id)
        expected_type = _COMPONENT_RESULT_TYPES[component]
        if not isinstance(result, expected_type):
            raise TypeError(
                f"{component.value} result must be {expected_type.__name__}."
            )
        if result.status != ComponentStatus.completed:
            raise ValueError(
                "Saved component results must have completed status."
            )

        current_status = ComponentStatus(
            document["component_statuses"][component.value]
        )
        _validate_component_status_transition(current_status, result.status)
        field_name = _COMPONENT_DOCUMENT_FIELDS[component]
        update_result = await self.collection.update_one(
            {
                "id": explanation_id,
                f"component_statuses.{component.value}": current_status.value,
            },
            {
                "$set": {
                    field_name: _model_document(result),
                    f"component_statuses.{component.value}": result.status.value,
                    f"component_errors.{component.value}": None,
                    "updated_at": _utc_now(),
                }
            },
        )
        _require_matched_one(update_result, "save_component_result")

    async def append_artifact(
        self,
        explanation_id: str,
        artifact: ExplanationArtifactReference,
    ) -> None:
        await self._mutable_document(explanation_id)
        result = await self.collection.update_one(
            {"id": explanation_id},
            {
                "$push": {"artifacts": _model_document(artifact)},
                "$set": {"updated_at": _utc_now()},
            },
        )
        _require_matched_one(result, "append_artifact")

    async def set_private_temporal_evidence_artifact(
        self,
        explanation_id: str,
        artifact: ExplanationArtifactReference,
    ) -> None:
        """Save compact retry evidence without exposing it in API artifacts."""

        await self._mutable_document(explanation_id)
        result = await self.collection.update_one(
            {"id": explanation_id},
            {
                "$set": {
                    "private_temporal_evidence_artifact": _model_document(artifact),
                    "updated_at": _utc_now(),
                }
            },
        )
        _require_matched_one(result, "set_private_temporal_evidence_artifact")

    async def update_run_evidence(
        self,
        explanation_id: str,
        *,
        development_placeholder: bool,
        research_eligible: bool,
        warnings: list[str],
        quality: ExplanationQuality,
        provenance: ExplanationProvenance,
    ) -> None:
        await self._mutable_document(explanation_id)
        result = await self.collection.update_one(
            {"id": explanation_id},
            {
                "$set": {
                    "development_placeholder": development_placeholder,
                    "research_eligible": research_eligible,
                    "warnings": list(dict.fromkeys(warnings)),
                    "quality": _model_document(quality),
                    "provenance": _model_document(provenance),
                    "updated_at": _utc_now(),
                }
            },
        )
        _require_matched_one(result, "update_run_evidence")

    async def begin_narrative_retry(
        self,
        explanation_id: str,
        *,
        max_attempts: int,
    ) -> bool:
        """Atomically change a failed optional narrative to ``running``.

        Completed XAI runs are otherwise immutable. This narrow operation is
        intentionally the only exception: it cannot update classifier output,
        temporal/semantic evidence, the report, or any public artifacts.
        """

        if max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        document = await self.collection.find_one({"id": explanation_id})
        if document is None:
            raise XaiPersistenceConsistencyError("XAI explanation does not exist.")
        current = document.get("component_statuses", {}).get("narrative")
        retry_count = document.get("narrative_retry_count", 0)
        if not isinstance(retry_count, int) or retry_count < 0:
            raise XaiPersistenceConsistencyError("Invalid narrative retry count.")
        if current not in {
            ComponentStatus.failed.value,
            ComponentStatus.not_available.value,
        } or retry_count >= max_attempts:
            return False

        now = _utc_now()
        # Existing completed runs predate this field. MongoDB matches a missing
        # field with ``None`` here, while later attempts use their stored count.
        retry_count_filter = (
            retry_count if "narrative_retry_count" in document else None
        )
        result = await self.collection.update_one(
            {
                "id": explanation_id,
                "status": ExplanationStatus.completed.value,
                "component_statuses.temporal": {
                    "$in": [
                        ComponentStatus.completed.value,
                        ComponentStatus.failed.value,
                        ComponentStatus.not_available.value,
                    ]
                },
                "component_statuses.semantic": {
                    "$in": [
                        ComponentStatus.completed.value,
                        ComponentStatus.failed.value,
                        ComponentStatus.not_available.value,
                    ]
                },
                "component_statuses.report": ComponentStatus.completed.value,
                "component_statuses.narrative": current,
                "narrative_retry_count": retry_count_filter,
            },
            {
                "$set": {
                    "component_statuses.narrative": ComponentStatus.running.value,
                    "component_errors.narrative": None,
                    "narrative": None,
                    "narrative_retry_count": retry_count + 1,
                    "updated_at": now,
                },
                "$push": {
                    "narrative_retry_history": _narrative_retry_event(
                        attempt=retry_count + 1,
                        status=ComponentStatus.running,
                        created_at=now,
                    )
                },
            },
        )
        return getattr(result, "matched_count", 0) == 1

    async def finish_narrative_retry(
        self,
        explanation_id: str,
        *,
        result: NarrativeExplanation | None = None,
        error: ExplanationError | None = None,
    ) -> None:
        """Finish a reserved narrative retry without reopening XAI evidence."""

        if (result is None) == (error is None):
            raise ValueError("Exactly one narrative retry result or error is required.")
        document = await self.collection.find_one({"id": explanation_id})
        if document is None:
            raise XaiPersistenceConsistencyError("XAI explanation does not exist.")
        retry_count = document.get("narrative_retry_count", 0)
        if not isinstance(retry_count, int) or retry_count < 1:
            raise XaiPersistenceConsistencyError("Narrative retry was not reserved.")
        now = _utc_now()
        if result is not None:
            if result.status != ComponentStatus.completed:
                raise ValueError("Narrative retry result must be completed.")
            fields = {
                "narrative": _model_document(result),
                "component_statuses.narrative": ComponentStatus.completed.value,
                "component_errors.narrative": None,
                "updated_at": now,
            }
            event_status = ComponentStatus.completed
            event_error = None
        else:
            if error is None or error.component != ExplanationComponent.narrative:
                raise ValueError("Narrative retry error must identify the narrative component.")
            fields = {
                "narrative": None,
                "component_statuses.narrative": ComponentStatus.failed.value,
                "component_errors.narrative": _model_document(error),
                "updated_at": now,
            }
            event_status = ComponentStatus.failed
            event_error = error
        update_result = await self.collection.update_one(
            {
                "id": explanation_id,
                "status": ExplanationStatus.completed.value,
                "component_statuses.narrative": ComponentStatus.running.value,
                "narrative_retry_count": retry_count,
            },
            {
                "$set": fields,
                "$push": {
                    "narrative_retry_history": _narrative_retry_event(
                        attempt=retry_count,
                        status=event_status,
                        created_at=now,
                        error=event_error,
                    )
                },
            },
        )
        _require_matched_one(update_result, "finish_narrative_retry")

    async def find_stale_active_explanations(
        self,
        *,
        updated_before: datetime,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        cursor = self.collection.find(
            {
                "status": {
                    "$in": [
                        ExplanationStatus.queued.value,
                        ExplanationStatus.running.value,
                        ExplanationStatus.partial.value,
                        ExplanationStatus.blocked.value,
                    ]
                },
                "updated_at": {"$lt": updated_before},
            }
        )
        documents = await cursor.limit(limit).to_list(length=limit)
        return [_serialize_document(document) for document in documents]

    async def _mutable_document(self, explanation_id: str) -> dict[str, Any]:
        document = await self.collection.find_one({"id": explanation_id})
        if document is None:
            raise XaiPersistenceConsistencyError("XAI explanation does not exist.")
        status = ExplanationStatus(document["status"])
        if status in {ExplanationStatus.completed, ExplanationStatus.failed}:
            raise XaiPersistenceConsistencyError(
                f"XAI explanation in terminal state {status.value} is immutable."
            )
        return document


def _model_document(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="python")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _status_history_entry(
    status: ExplanationStatus,
    created_at: datetime,
) -> dict[str, Any]:
    return {"status": status.value, "created_at": created_at}


def _narrative_retry_event(
    *,
    attempt: int,
    status: ComponentStatus,
    created_at: datetime,
    error: ExplanationError | None = None,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "status": status.value,
        "error": _model_document(error) if error is not None else None,
        "created_at": created_at,
    }


def _validate_explanation_status_transition(
    current_status: ExplanationStatus,
    next_status: ExplanationStatus,
) -> None:
    if next_status not in VALID_EXPLANATION_STATUS_TRANSITIONS[current_status]:
        raise XaiPersistenceConsistencyError(
            "Invalid XAI explanation status transition: "
            f"{current_status.value} -> {next_status.value}."
        )


def _validate_component_status_transition(
    current_status: ComponentStatus,
    next_status: ComponentStatus,
) -> None:
    if next_status not in VALID_COMPONENT_STATUS_TRANSITIONS[current_status]:
        raise XaiPersistenceConsistencyError(
            "Invalid XAI component status transition: "
            f"{current_status.value} -> {next_status.value}."
        )


def _require_identifier(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} is required.")


def _require_matched_one(result: Any, operation: str) -> None:
    if getattr(result, "matched_count", None) != 1:
        raise XaiPersistenceConsistencyError(
            f"MongoDB XAI update did not match exactly one record: {operation}."
        )


def _serialize_document(value: Any) -> Any:
    if value is None:
        return None
    if ObjectId is not None and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_serialize_document(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_document(item) for key, item in value.items()}
    return value
