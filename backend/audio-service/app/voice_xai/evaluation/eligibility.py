"""Centralized, versioned, fail-closed Voice XAI research-eligibility policy.

Research eligibility answers a stricter question than "did the XAI pipeline
complete without error": can this explanation's evidence support a research
claim (e.g. in the MULTI-SCOPE thesis/paper)? A run can be ``completed`` and
still not be research eligible.

This module is the single place that decision is made -- previously it was
an inline boolean expression inside ``VoiceXaiOrchestrator._run_internal``.
Consolidating it here does not change today's outcome (every requirement
below is still unmet, so evaluation still returns ``eligible=False``); it
exists so future changes flip individual, named, documented requirements
instead of loosening one combined expression. Every criterion must be met;
an unmet criterion fails the decision closed, never open. See
``CLASSIFIER_XAI_CONTRACT.md`` and ``VOICE_XAI_INTEGRATION_AUDIT_2026-08-08.md``
for the evidence behind each requirement below.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.xai import ClassifierSnapshot, SemanticExplanation, TemporalExplanation

RESEARCH_ELIGIBILITY_POLICY_VERSION = "voice-xai-research-eligibility-v1"


@dataclass(frozen=True, slots=True)
class ResearchEligibilityDecision:
    eligible: bool
    policy_version: str
    unmet_requirements: tuple[str, ...]


def evaluate_research_eligibility(
    *,
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> ResearchEligibilityDecision:
    """Fail-closed research-eligibility evaluation for one completed run.

    Requirements, all mandatory:

    Classifier
      - No dummy branch contributed to the fused result. Independently
        re-enforced by ``ClassifierSnapshot``'s own model validator, which
        refuses ``research_eligible=True`` together with
        ``contains_dummy_branches=True`` -- this check is deliberately
        redundant with that guard rather than trusting it alone.
      - The classifier snapshot must already consider itself eligible
        (real branch provenance recorded, not a placeholder run).

    Temporal
      - A temporal component result must exist for this run.
      - ``temporal.research_eligible`` must be True. The real XLS-R temporal
        service currently reports False until its classifier and calibration
        verification evidence is explicitly attested.

    Semantic
      - A semantic component result must exist for this run.
      - ``semantic.research_eligible`` must be True. Even
        ``ProductionSemanticExplanationService`` (a real loaded XGBoost/SHAP
        artifact) hardcodes this False today: whole-clip evidence lacks time
        localization, and the 28-feature acoustic recipe (including GCI) has
        not been parity-validated against the training pipeline.

    Combined
      - Not yet evaluated here pending the above. A versioned rule for
        temporal/semantic overlap agreement (e.g. an IoU threshold) must be
        defined before it can factor into eligibility; ``quality.py``
        computes the raw interval IoU today but no eligibility threshold is
        defined on it yet.

    Returns a decision that is always ``eligible=False`` until every
    requirement above is genuinely satisfied -- not merely engineered around
    -- with the specific unmet requirement names for diagnostics/logging.
    """

    unmet: list[str] = []
    if classifier.contains_dummy_branches:
        unmet.append("classifier_contains_dummy_branches")
    if not classifier.research_eligible:
        unmet.append("classifier_not_research_eligible")
    if temporal is None:
        unmet.append("temporal_missing")
    elif not temporal.research_eligible:
        unmet.append("temporal_not_research_eligible")
    if semantic is None:
        unmet.append("semantic_missing")
    elif not semantic.research_eligible:
        unmet.append("semantic_not_research_eligible")

    return ResearchEligibilityDecision(
        eligible=not unmet,
        policy_version=RESEARCH_ELIGIBILITY_POLICY_VERSION,
        unmet_requirements=tuple(unmet),
    )


__all__ = [
    "RESEARCH_ELIGIBILITY_POLICY_VERSION",
    "ResearchEligibilityDecision",
    "evaluate_research_eligibility",
]
