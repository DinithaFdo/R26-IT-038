"""Voice XAI eligibility and selected, fixed offline AOPC reference metrics.

Eligibility describes a completed real classifier/temporal/clip-semantic run.
It does not certify window localization accuracy or an AOPC pass threshold.
The selected single-audio measurements below are offline reference values;
they are not recomputed for each upload and are never labeled per-analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.xai import (
    ClassifierSnapshot, ComponentStatus, MetricEvidence, MetricStatus,
    SemanticExplanation, TemporalExplanation,
)

RESEARCH_ELIGIBILITY_POLICY_VERSION = "voice-xai-research-eligibility-v2-clip-semantic"


@dataclass(frozen=True, slots=True)
class OfflineAopcReference:
    name: str
    value: float
    random_mean: float
    source_run: str
    protocol: str
    model_fingerprint: str
    audio_id: str = "LA_D_2302632.wav"
    audio_sha256: str = "7b070c05c229f4a1af2709ddc0f35fa373ac202e3dbff7c1a39f80691403c2d2"
    scope: str = "offline_validation"
    n_audio: int = 1
    fixed_reference: bool = True

    @property
    def advantage_over_random(self) -> float:
        return self.value - self.random_mean

    def metric_evidence(self) -> MetricEvidence:
        return MetricEvidence(
            status=MetricStatus.available, value=self.value, scope="offline_validation",
            dataset_version=f"Selected single-audio reference: {self.audio_id}; n=1; {self.source_run}",
            reason=(f"Fixed offline reference, not measured on the current upload. {self.protocol}. "
                    f"Random mean={self.random_mean:.6f}; advantage={self.advantage_over_random:+.6f}. "
                    "Eligibility does not imply that this ranking beats random."),
        )


# Measured results from the same recording. Keep the actual random comparators,
# including the negative temporal advantage; these are not dataset averages.
OFFLINE_SEMANTIC_AOPC = OfflineAopcReference(
    name="semantic_aopc", value=0.517231721941027, random_mean=0.024964636611560034,
    source_run="semantic_aopc/LA_D_2302632_20261017T211001893554Z",
    protocol="Raw-score SHAP MoRF; K=20/165; saved median replacement; denominator=21; 30 random orders",
    model_fingerprint="541e4e6e8ddd57a4235c0fe0cdd09d4806b7c1e4d77407dcdc0be205008d5575",
)
OFFLINE_TEMPORAL_AOPC = OfflineAopcReference(
    name="temporal_aopc", value=0.1838843630005916, random_mean=0.2708303292786392,
    source_run="temporal_aopc/LA_D_2302632_20261017T210648071052Z",
    protocol="Attention MoRF; 100-ms blocks; 10/20/30/40/50% silence; denominator=6; 10 matched random orders",
    model_fingerprint="b684111dc07643e1307f493352f4051664f1edb1b0e3f7f1db76f81f410b0c8e",
)
OFFLINE_AOPC_REFERENCES = (OFFLINE_SEMANTIC_AOPC, OFFLINE_TEMPORAL_AOPC)


@dataclass(frozen=True, slots=True)
class ResearchEligibilityDecision:
    eligible: bool
    policy_version: str
    unmet_requirements: tuple[str, ...]
    offline_metrics: tuple[OfflineAopcReference, ...] = OFFLINE_AOPC_REFERENCES


def evaluate_research_eligibility(
    *,
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> ResearchEligibilityDecision:
    """Require an eligible classifier and completed, non-placeholder components.

    Real semantic inference with recorded held-out surrogate fidelity is eligible
    for clip-level feature explanation. Experimental window localization remains
    separately labeled and does not block this clip-level eligibility. Temporal
    evidence still requires the original real SSL verification metadata.
    Fixed offline AOPC references are reported independently of this decision:
    they neither replace missing evidence nor grant eligibility by their size.
    """

    unmet: list[str] = []
    if classifier.contains_dummy_branches:
        unmet.append("classifier_contains_dummy_branches")
    if not classifier.research_eligible:
        unmet.append("classifier_not_research_eligible")
    if temporal is None:
        unmet.append("temporal_missing")
    elif (not temporal.research_eligible or temporal.development_placeholder
          or temporal.status != ComponentStatus.completed):
        unmet.append("temporal_not_research_eligible")
    if semantic is None:
        unmet.append("semantic_missing")
    elif (not semantic.research_eligible or semantic.development_placeholder
          or semantic.status != ComponentStatus.completed):
        unmet.append("semantic_not_research_eligible")

    return ResearchEligibilityDecision(
        eligible=not unmet,
        policy_version=RESEARCH_ELIGIBILITY_POLICY_VERSION,
        unmet_requirements=tuple(unmet),
    )


__all__ = [
    "RESEARCH_ELIGIBILITY_POLICY_VERSION",
    "ResearchEligibilityDecision",
    "OfflineAopcReference",
    "OFFLINE_SEMANTIC_AOPC",
    "OFFLINE_TEMPORAL_AOPC",
    "OFFLINE_AOPC_REFERENCES",
    "evaluate_research_eligibility",
]
