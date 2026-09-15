"""Application-level lifecycle policy for independent Voice XAI state."""

from __future__ import annotations

from typing import Any

from app.schemas.xai import (
    ComponentStatus,
    ExplanationComponent,
    ExplanationComponentStatuses,
    ExplanationError,
    ExplanationStatus,
)
from app.voice_xai.persistence.protocols import (
    ExplanationComponentResult,
    XaiExplanationRepository,
)

_TERMINAL_INPUT_COMPONENT_STATUSES = {
    ComponentStatus.completed,
    ComponentStatus.failed,
    ComponentStatus.not_available,
}


class XaiStatusLifecycleError(RuntimeError):
    """Raised when application orchestration violates the XAI lifecycle."""


class XaiExplanationNotFoundError(XaiStatusLifecycleError):
    """Raised without revealing whether another owner has the explanation."""


def derive_explanation_status(
    statuses: ExplanationComponentStatuses,
) -> ExplanationStatus:
    """Derive overall state without coupling it to classifier status."""

    # Narrative is optional and must not turn an otherwise untouched queued,
    # running, or blocked XAI run into ``partial`` merely because legacy runs
    # default it to not_available. It only participates once the deterministic
    # report and both evidence components are terminal.
    values = (statuses.temporal, statuses.semantic, statuses.report)
    if statuses.report in {
        ComponentStatus.failed,
        ComponentStatus.not_available,
    }:
        return ExplanationStatus.failed
    if (
        statuses.report == ComponentStatus.completed
        and statuses.temporal in _TERMINAL_INPUT_COMPONENT_STATUSES
        and statuses.semantic in _TERMINAL_INPUT_COMPONENT_STATUSES
        and statuses.narrative in _TERMINAL_INPUT_COMPONENT_STATUSES
    ):
        return ExplanationStatus.completed
    if any(
        value
        in {
            ComponentStatus.completed,
            ComponentStatus.failed,
            ComponentStatus.not_available,
        }
        for value in values
    ):
        return ExplanationStatus.partial
    if any(value == ComponentStatus.running for value in values):
        return ExplanationStatus.running
    if any(value == ComponentStatus.blocked for value in values):
        return ExplanationStatus.blocked
    return ExplanationStatus.queued


class XaiStatusService:
    """Coordinates legal run/component transitions through a repository port."""

    def __init__(self, repository: XaiExplanationRepository) -> None:
        self._repository = repository

    async def start_run(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
    ) -> None:
        document = await self._get_owned(explanation_id, owner_user_id)
        current = ExplanationStatus(document["status"])
        if current in {ExplanationStatus.running, ExplanationStatus.partial}:
            return
        if current != ExplanationStatus.queued:
            raise XaiStatusLifecycleError(
                f"Cannot start XAI run from {current.value}."
            )
        await self._repository.transition_status(
            explanation_id,
            ExplanationStatus.running,
        )

    async def component_statuses(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
    ) -> ExplanationComponentStatuses:
        """Return the current owner-scoped component states for orchestration."""
        return _component_statuses(
            await self._get_owned(explanation_id, owner_user_id)
        )

    async def start_component(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
        component: ExplanationComponent,
    ) -> None:
        document = await self._ensure_run_mutable(explanation_id, owner_user_id)
        statuses = _component_statuses(document)
        if component == ExplanationComponent.report:
            _require_report_inputs_terminal(statuses)
        current = _component_status(statuses, component)
        if current == ComponentStatus.running:
            return
        if current != ComponentStatus.queued:
            raise XaiStatusLifecycleError(
                f"Cannot start {component.value} component from {current.value}."
            )
        await self._repository.update_component_status(
            explanation_id,
            component,
            ComponentStatus.running,
        )
        await self._synchronize_overall(explanation_id, owner_user_id)

    async def complete_component(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
        component: ExplanationComponent,
        result: ExplanationComponentResult,
    ) -> None:
        document = await self._ensure_run_mutable(explanation_id, owner_user_id)
        statuses = _component_statuses(document)
        if component == ExplanationComponent.report:
            _require_report_inputs_terminal(statuses)
        current = _component_status(statuses, component)
        if current == ComponentStatus.queued:
            await self._repository.update_component_status(
                explanation_id,
                component,
                ComponentStatus.running,
            )
        elif current != ComponentStatus.running:
            raise XaiStatusLifecycleError(
                f"Cannot complete {component.value} component from {current.value}."
            )
        await self._repository.save_component_result(
            explanation_id,
            component,
            result,
        )
        await self._synchronize_overall(explanation_id, owner_user_id)

    async def fail_component(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
        component: ExplanationComponent,
        error: ExplanationError,
    ) -> None:
        await self._set_failure_like_component_status(
            explanation_id=explanation_id,
            owner_user_id=owner_user_id,
            component=component,
            status=ComponentStatus.failed,
            error=error,
        )

    async def block_component(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
        component: ExplanationComponent,
        error: ExplanationError,
    ) -> None:
        await self._set_failure_like_component_status(
            explanation_id=explanation_id,
            owner_user_id=owner_user_id,
            component=component,
            status=ComponentStatus.blocked,
            error=error,
        )

    async def mark_component_not_available(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
        component: ExplanationComponent,
        error: ExplanationError,
    ) -> None:
        await self._set_failure_like_component_status(
            explanation_id=explanation_id,
            owner_user_id=owner_user_id,
            component=component,
            status=ComponentStatus.not_available,
            error=error,
        )

    async def requeue_component(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
        component: ExplanationComponent,
    ) -> None:
        document = await self._get_owned(explanation_id, owner_user_id)
        overall = ExplanationStatus(document["status"])
        if overall in {ExplanationStatus.completed, ExplanationStatus.failed}:
            raise XaiStatusLifecycleError(
                f"Cannot requeue a component for terminal run {overall.value}."
            )
        statuses = _component_statuses(document)
        current = _component_status(statuses, component)
        if current != ComponentStatus.blocked:
            raise XaiStatusLifecycleError(
                f"Cannot requeue {component.value} component from {current.value}."
            )
        await self._repository.update_component_status(
            explanation_id,
            component,
            ComponentStatus.queued,
        )
        await self._synchronize_overall(explanation_id, owner_user_id)

    async def _set_failure_like_component_status(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
        component: ExplanationComponent,
        status: ComponentStatus,
        error: ExplanationError,
    ) -> None:
        document = await self._ensure_run_mutable(explanation_id, owner_user_id)
        current = _component_status(_component_statuses(document), component)
        if current not in {
            ComponentStatus.queued,
            ComponentStatus.running,
            ComponentStatus.blocked,
        }:
            raise XaiStatusLifecycleError(
                f"Cannot set {component.value} component from {current.value} "
                f"to {status.value}."
            )
        await self._repository.update_component_status(
            explanation_id,
            component,
            status,
            error=error,
        )
        await self._synchronize_overall(explanation_id, owner_user_id)

    async def _ensure_run_mutable(
        self,
        explanation_id: str,
        owner_user_id: str,
    ) -> dict[str, Any]:
        document = await self._get_owned(explanation_id, owner_user_id)
        current = ExplanationStatus(document["status"])
        if current == ExplanationStatus.queued:
            await self._repository.transition_status(
                explanation_id,
                ExplanationStatus.running,
            )
            return await self._get_owned(explanation_id, owner_user_id)
        if current in {ExplanationStatus.running, ExplanationStatus.partial}:
            return document
        if current == ExplanationStatus.blocked:
            raise XaiStatusLifecycleError(
                "Blocked XAI runs must requeue blocked components before work resumes."
            )
        raise XaiStatusLifecycleError(
            f"XAI run in terminal state {current.value} is immutable."
        )

    async def _synchronize_overall(
        self,
        explanation_id: str,
        owner_user_id: str,
    ) -> None:
        document = await self._get_owned(explanation_id, owner_user_id)
        current = ExplanationStatus(document["status"])
        desired = derive_explanation_status(_component_statuses(document))
        if current == desired:
            return
        error = None
        if desired == ExplanationStatus.failed:
            raw_error = document.get("component_errors", {}).get("report")
            if raw_error is None:
                raise XaiStatusLifecycleError(
                    "Failed report component is missing its safe error."
                )
            error = ExplanationError.model_validate(raw_error)
        await self._repository.transition_status(
            explanation_id,
            desired,
            error=error,
        )

    async def fail_run(
        self,
        *,
        explanation_id: str,
        owner_user_id: str,
        error: ExplanationError,
    ) -> None:
        """Fail all unfinished components before recording a terminal run error.

        Recovery and the outer orchestration boundary must never leave a
        terminal overall status alongside ``queued`` or ``running`` component
        statuses.  Each mutable component receives the same safe error code,
        with its own component identifier, so clients can explain the failure
        without treating the run as still in progress.
        """

        document = await self._get_owned(explanation_id, owner_user_id)
        current = ExplanationStatus(document["status"])
        if current == ExplanationStatus.completed:
            return
        if current == ExplanationStatus.failed:
            await self._repository.update_terminal_error(explanation_id, error=error)
            return
        if current == ExplanationStatus.blocked:
            # A blocked run must become mutable before its blocked/queued
            # component statuses can be made terminal. This is a legal
            # recovery-only transition; no work is resumed.
            await self._repository.transition_status(
                explanation_id,
                ExplanationStatus.queued,
            )
        # Report is last: a failed report makes the whole run terminal, so all
        # other unfinished components (including optional narrative) must be
        # made terminal before that transition occurs.
        component_order = (
            ExplanationComponent.temporal,
            ExplanationComponent.semantic,
            ExplanationComponent.narrative,
            ExplanationComponent.report,
        )
        for component in component_order:
            document = await self._get_owned(explanation_id, owner_user_id)
            status = _component_status(_component_statuses(document), component)
            if status not in {
                ComponentStatus.queued,
                ComponentStatus.running,
                ComponentStatus.blocked,
            }:
                continue
            await self.fail_component(
                explanation_id=explanation_id,
                owner_user_id=owner_user_id,
                component=component,
                error=ExplanationError(
                    component=component,
                    code=error.code,
                    message=error.message,
                ),
            )
        document = await self._get_owned(explanation_id, owner_user_id)
        current = ExplanationStatus(document["status"])
        if current == ExplanationStatus.completed:
            return
        if current == ExplanationStatus.failed:
            await self._repository.update_terminal_error(explanation_id, error=error)
            return
        await self._repository.transition_status(
            explanation_id,
            ExplanationStatus.failed,
            error=error,
        )

    async def _get_owned(
        self,
        explanation_id: str,
        owner_user_id: str,
    ) -> dict[str, Any]:
        document = await self._repository.get_explanation_for_owner(
            explanation_id=explanation_id,
            owner_user_id=owner_user_id,
        )
        if document is None:
            raise XaiExplanationNotFoundError("XAI explanation was not found.")
        return document


def _component_statuses(document: dict[str, Any]) -> ExplanationComponentStatuses:
    return ExplanationComponentStatuses.model_validate(
        document["component_statuses"]
    )


def _component_status(
    statuses: ExplanationComponentStatuses,
    component: ExplanationComponent,
) -> ComponentStatus:
    return getattr(statuses, component.value)


def _require_report_inputs_terminal(
    statuses: ExplanationComponentStatuses,
) -> None:
    if (
        statuses.temporal not in _TERMINAL_INPUT_COMPONENT_STATUSES
        or statuses.semantic not in _TERMINAL_INPUT_COMPONENT_STATUSES
    ):
        raise XaiStatusLifecycleError(
            "Report component requires terminal temporal and semantic components."
        )
