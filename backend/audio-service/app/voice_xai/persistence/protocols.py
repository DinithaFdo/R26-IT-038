"""Persistence protocol for versioned Voice XAI explanation runs."""

from datetime import datetime
from typing import Any, Protocol

from app.schemas.common import SourceType
from app.schemas.xai import (
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationComponent,
    ExplanationError,
    ExplanationArtifactReference,
    ExplanationQuality,
    ExplanationProvenance,
    ExplanationStatus,
    NarrativeExplanation,
    SemanticExplanation,
    TemporalExplanation,
)

ExplanationComponentResult = (
    TemporalExplanation
    | SemanticExplanation
    | CombinedExplanationReport
    | NarrativeExplanation
)


class XaiExplanationRepository(Protocol):
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
        raise NotImplementedError

    async def get_explanation_for_owner(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def get_latest_explanation_for_prediction(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def list_explanations_for_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> list[dict[str, Any]]:
        """Return every owner-scoped run associated with one prediction."""

        raise NotImplementedError

    async def delete_explanations_for_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> int:
        """Hard-delete every owner-scoped run associated with one prediction."""

        raise NotImplementedError

    async def transition_status(
        self,
        explanation_id: str,
        status: ExplanationStatus,
        *,
        error: ExplanationError | None = None,
    ) -> None:
        raise NotImplementedError

    async def update_terminal_error(
        self,
        explanation_id: str,
        *,
        error: ExplanationError,
    ) -> None:
        """Replace the safe top-level error of an already failed run."""

        raise NotImplementedError

    async def update_component_status(
        self,
        explanation_id: str,
        component: ExplanationComponent,
        status: ComponentStatus,
        *,
        error: ExplanationError | None = None,
    ) -> None:
        raise NotImplementedError

    async def save_component_result(
        self,
        explanation_id: str,
        component: ExplanationComponent,
        result: ExplanationComponentResult,
    ) -> None:
        raise NotImplementedError

    async def append_artifact(
        self,
        explanation_id: str,
        artifact: ExplanationArtifactReference,
    ) -> None:
        """Persist reference metadata only; artifact bytes stay in private storage."""

    async def set_private_temporal_evidence_artifact(
        self,
        explanation_id: str,
        artifact: ExplanationArtifactReference,
    ) -> None:
        """Persist a retry-only compact XLS-R evidence reference.

        This is intentionally separate from ``artifacts``: it is input for a
        future retry, not a user-facing explanation download.
        """

        raise NotImplementedError

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
        """Persist calculated run metadata after component services execute."""

        raise NotImplementedError

    async def begin_narrative_retry(
        self,
        explanation_id: str,
        *,
        max_attempts: int,
    ) -> bool:
        """Atomically reserve an optional narrative-only retry.

        This intentionally does not reopen the deterministic XAI run. A
        ``True`` result means the narrative component is now running; ``False``
        means another request changed retry eligibility first.
        """

        raise NotImplementedError

    async def finish_narrative_retry(
        self,
        explanation_id: str,
        *,
        result: NarrativeExplanation | None = None,
        error: ExplanationError | None = None,
    ) -> None:
        """Record the terminal result of a reserved narrative-only retry."""

        raise NotImplementedError

    async def find_stale_active_explanations(
        self,
        *,
        updated_before: datetime,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return non-terminal runs last updated before ``updated_before``.

        Used only for restart-recovery reconciliation: a run left in
        ``queued``/``running``/``partial``/``blocked`` from before the
        current process started.
        """

        raise NotImplementedError
