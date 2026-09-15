import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.config.settings import Settings
from app.models.runtime import canonical_branch_name
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.prediction import BranchPrediction, FusionResult, ProbabilityScores

logger = logging.getLogger(__name__)

FusionMethod = Literal["weighted_average", "simple_average", "majority_vote"]

RECOGNIZED_BRANCHES = frozenset(
    {
        "cnn",
        "cnn_acoustic",
        "lfcc_cnn_tcn",
        "aasist",
        "ssl",
        "ssl_wavlm_xlsr",
        "ssl_sequence",
        "glottal",
        "glottal_features",
    }
)
DEFAULT_DEVELOPMENT_WEIGHTS = {
    "cnn": 0.25,
    "aasist": 0.25,
    "ssl": 0.25,
    "glottal": 0.25,
}
DUMMY_FUSION_WARNING = (
    "Fusion includes deterministic dummy branch outputs for system development only; "
    "this is not a research result."
)
PARTIAL_SYSTEM_WARNING = (
    "This result combines only the detection branches currently available. The "
    "full MULTI-SCOPE branch set did not contribute, so it must not be reported "
    "as full-system performance."
)
UNVERIFIED_BRANCH_WARNING = (
    "At least one contributing branch ran a trained checkpoint whose feature "
    "pipeline or class mapping is unverified. Scores are unvalidated."
)

#: The branch set the finished MULTI-SCOPE primary research detector is
#: designed around. A result that does not include all of them is a
#: partial-stage (degraded) result, however good the branches that did run
#: were, and must not be reported as the final research detector's output.
#:
#: Glottal is deliberately NOT in this set. Per the final detector contract
#: (``model_artifacts/fusion/final_detector_v1.json``), Glottal is auxiliary
#: physiological/XAI evidence only -- it must never gate primary research
#: eligibility, and it never contributes to the fusion score below (it is
#: excluded from ``successful_branches`` entirely whenever it is disabled, as
#: it is today; even if it is enabled in the future, it must not be added
#: back to this tuple).
FULL_SYSTEM_BRANCHES: tuple[str, ...] = (
    "lfcc_cnn_tcn",
    "aasist",
    "ssl_sequence",
)

#: Frozen, DEV-selected decision threshold for the final research detector.
#: Not derived, not retuned here -- this constant exists only to catch a
#: corrupted or hand-edited ``final_detector_v1.json`` that has drifted from
#: the actual frozen value, not to be a second source of truth for it.
FROZEN_RESEARCH_THRESHOLD = 0.5519237850482265
DEFAULT_FUSION_CONTRACT_FILENAME = "fusion/final_detector_v1.json"

#: Human-facing identity of the legacy 3-branch frozen detector when it is
#: resolved from ``final_detector_v1.json`` (see ``FusionEngine.from_settings``).
#: Distinct from ``FusionContract.contract_version`` (="final-detector-v1",
#: the value literally stored in that file) -- this is the name Fusion V3's
#: explicit-fallback contract (see below) requires the fallback result to be
#: reported under, so a fallback response is never mistaken for "Fusion V3".
LEGACY_FUSION_VERSION = "legacy-3branch-frozen-v1"
LEGACY_FUSION_MODE = "legacy_average"
#: Development/no-contract default identity -- unchanged from the pre-V3
#: behaviour, kept as the ``FusionEngine`` default so every existing caller
#: that never asked about ``fusion_version``/``fusion_mode`` keeps seeing the
#: same values it always has.
DEVELOPMENT_FUSION_VERSION = "score-level-fusion-v1"
DEVELOPMENT_FUSION_MODE = "development_average"

# ---------------------------------------------------------------------------
# Fusion V3: constrained 4-branch convex weighted fusion
# ---------------------------------------------------------------------------
#
# A second, independent fusion path alongside the legacy ``FusionEngine``
# above. Deliberately NOT folded into ``FusionEngine`` itself: V3's contract
# (exact weighted dot product, all four branches required, no renormalisation
# ever) is simple enough that reusing ``FusionEngine``'s partial-branch
# renormalising machinery would only make the "no renormalisation" guarantee
# implicit (true only because weights happen to sum to 1) rather than
# structural. ``VoiceService`` decides, per request, whether to use this or
# fall back to the ``FusionEngine`` instance built from the legacy contract.

CONVEX_V3_VERSION = "fusion-convex-4branch-v3"
CONVEX_V3_ARCHITECTURE = "constrained_convex_weighted_fusion"
CONVEX_V3_MODE = "learned_constrained"
#: Exact input order the frozen V3 artifact was trained/exported with.
CONVEX_V3_FEATURE_ORDER: tuple[str, ...] = (
    "cnn_spoof_probability",
    "aasist_spoof_probability",
    "ssl_spoof_probability",
    "glottal_spoof_probability",
)
#: Canonical branch name for each entry in ``CONVEX_V3_FEATURE_ORDER``, same
#: position-for-position order.
CONVEX_V3_BRANCH_ORDER: tuple[str, ...] = (
    "lfcc_cnn_tcn",
    "aasist",
    "ssl_sequence",
    "glottal",
)
CONVEX_V3_FEATURE_TO_BRANCH: dict[str, str] = dict(
    zip(CONVEX_V3_FEATURE_ORDER, CONVEX_V3_BRANCH_ORDER, strict=True)
)
CONVEX_V3_THRESHOLD = 0.5


class ConvexFusionContractError(ValueError):
    """The frozen Fusion V3 artifact/config is missing data or invalid."""


@dataclass(frozen=True)
class ConvexFusionContract:
    """The frozen Fusion V3 convex-weighted-fusion contract, loaded from disk."""

    version: str
    architecture: str
    #: Raw feature-name order as declared by the artifact (kept for audit --
    #: always equal to ``CONVEX_V3_FEATURE_ORDER`` once validated).
    feature_order: tuple[str, ...]
    #: Frozen weights keyed by CANONICAL branch name (``lfcc_cnn_tcn`` etc.),
    #: not by the raw feature name -- convenient for both fusion math and
    #: provenance reporting, matching how ``FusionEngine.branch_weights`` is
    #: keyed elsewhere in this module.
    weights: dict[str, float]
    threshold: float
    source_json_path: Path
    source_joblib_path: Path | None


@dataclass(frozen=True)
class FusionContract:
    """The final research detector's fusion configuration, loaded from disk.

    ``primary_branches`` is always exactly ``FULL_SYSTEM_BRANCHES`` -- the
    contract file states it explicitly rather than importing the constant, so
    a future edit to one is caught by the other via ``load_fusion_contract``,
    not silently trusted.
    """

    contract_version: str
    fusion_method: FusionMethod
    decision_threshold: float
    primary_branches: tuple[str, ...]
    source_path: Path


class FusionContractError(ValueError):
    """The frozen fusion contract file is missing required data or invalid."""


def load_fusion_contract(path: Path) -> FusionContract | None:
    """Load and validate the frozen research-detector fusion contract.

    Returns ``None`` when ``path`` does not exist (backward compatible: a
    deployment without the contract file falls back to plain
    ``Settings``-driven fusion configuration). Raises ``FusionContractError``
    for anything that *does* exist but is invalid -- a present-but-broken
    contract file must fail loudly, never be silently ignored.
    """

    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FusionContractError(f"Fusion contract at {path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise FusionContractError(f"Fusion contract at {path} must be a JSON object.")

    method = raw.get("fusion_method")
    if method not in {"weighted_average", "simple_average", "majority_vote"}:
        raise FusionContractError(
            f"Fusion contract at {path} has an invalid fusion_method: {method!r}."
        )
    threshold = raw.get("decision_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise FusionContractError(
            f"Fusion contract at {path} has an invalid decision_threshold: {threshold!r}."
        )
    threshold = float(threshold)
    if not math.isclose(threshold, FROZEN_RESEARCH_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise FusionContractError(
            f"Fusion contract at {path} declares decision_threshold={threshold!r}, "
            f"which does not match the frozen research threshold "
            f"{FROZEN_RESEARCH_THRESHOLD!r}. The threshold must never be derived "
            "or retuned -- fix the contract file, do not edit this check."
        )

    raw_branches = raw.get("primary_branches")
    if not isinstance(raw_branches, list) or not raw_branches:
        raise FusionContractError(
            f"Fusion contract at {path} is missing a non-empty primary_branches list."
        )
    try:
        canonical_branches = tuple(
            sorted({canonical_branch_name(branch) for branch in raw_branches})
        )
    except ValueError as error:
        raise FusionContractError(
            f"Fusion contract at {path} has an unrecognized branch name: {error}"
        ) from error
    if canonical_branches != tuple(sorted(FULL_SYSTEM_BRANCHES)):
        raise FusionContractError(
            f"Fusion contract at {path} declares primary_branches={canonical_branches}, "
            f"which does not match the required primary detector branch set "
            f"{tuple(sorted(FULL_SYSTEM_BRANCHES))}. Glottal must never appear here."
        )
    if raw.get("glottal_used_for_primary_decision") not in (False, None):
        raise FusionContractError(
            f"Fusion contract at {path} sets glottal_used_for_primary_decision to "
            "something other than false -- Glottal must never contribute to the "
            "primary fusion decision."
        )

    return FusionContract(
        contract_version=str(raw.get("contract_version", "")),
        fusion_method=method,
        decision_threshold=threshold,
        primary_branches=canonical_branches,
        source_path=path,
    )


def load_convex_fusion_contract(
    json_path: Path,
    joblib_path: Path | None = None,
) -> ConvexFusionContract | None:
    """Load and validate the frozen Fusion V3 convex-fusion contract.

    Mirrors ``load_fusion_contract``'s convention exactly: ``None`` when
    ``json_path`` does not exist (V3 is simply unavailable -- callers decide
    whether that means "fall back to the legacy detector" or "fail"), raises
    ``ConvexFusionContractError`` for anything that does exist but fails any
    of the section-4 checks (exact version/architecture/feature order/weight
    count/non-negativity/sum-to-one/threshold/label mapping). The joblib
    artifact is cross-validated too when present and ``joblib`` is
    importable -- best-effort, since ``joblib``/``scikit-learn`` are only a
    hard dependency for the ``glottal`` extra, not the base install.
    """

    if not json_path.is_file():
        return None
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} must be a JSON object."
        )

    # The JSON contract's own "version" field is a short artifact-revision
    # tag ("v3"), not the full identity string -- the joblib artifact's
    # "version" field is where "fusion-convex-4branch-v3" (CONVEX_V3_VERSION,
    # matching the exact string the task's own contract names) actually
    # lives, and is cross-validated against it below when available. The
    # resulting ``ConvexFusionContract.version`` always reports the full
    # CONVEX_V3_VERSION string, not this short tag.
    json_short_version = raw.get("version")
    if json_short_version != "v3":
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} has version={json_short_version!r}, "
            "expected 'v3'."
        )
    architecture = raw.get("architecture")
    if architecture != CONVEX_V3_ARCHITECTURE:
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} has architecture={architecture!r}, "
            f"expected {CONVEX_V3_ARCHITECTURE!r}."
        )

    feature_order_raw = raw.get("feature_order")
    if (
        not isinstance(feature_order_raw, list)
        or tuple(feature_order_raw) != CONVEX_V3_FEATURE_ORDER
    ):
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} has feature_order="
            f"{feature_order_raw!r}, expected exactly {CONVEX_V3_FEATURE_ORDER!r} "
            "in that order."
        )

    weights_raw = raw.get("weights")
    if not isinstance(weights_raw, dict) or set(weights_raw) != set(CONVEX_V3_FEATURE_ORDER):
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} must declare exactly the 4 "
            f"weights {CONVEX_V3_FEATURE_ORDER!r}."
        )
    weights_by_branch: dict[str, float] = {}
    for feature_name, branch_name in CONVEX_V3_FEATURE_TO_BRANCH.items():
        weight = weights_raw[feature_name]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ConvexFusionContractError(
                f"Fusion V3 contract at {json_path} has a non-numeric weight "
                f"for {feature_name!r}: {weight!r}."
            )
        weight = float(weight)
        if not math.isfinite(weight):
            raise ConvexFusionContractError(
                f"Fusion V3 contract at {json_path} has a non-finite weight "
                f"for {feature_name!r}: {weight!r}."
            )
        if weight < 0:
            raise ConvexFusionContractError(
                f"Fusion V3 contract at {json_path} has a negative weight for "
                f"{feature_name!r}: {weight!r}."
            )
        weights_by_branch[branch_name] = weight

    weight_sum = sum(weights_by_branch.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} weights sum to {weight_sum!r}, "
            "expected 1.0."
        )

    constraints = raw.get("constraints")
    if not isinstance(constraints, dict) or (
        constraints.get("non_negative_weights") is not True
        or constraints.get("weights_sum_to_one") is not True
        or constraints.get("intercept") is not False
        or constraints.get("post_sigmoid_calibration") is not False
    ):
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} has an invalid or missing "
            "constraints block (must declare non_negative_weights=true, "
            "weights_sum_to_one=true, intercept=false, "
            "post_sigmoid_calibration=false)."
        )

    decision = raw.get("decision")
    threshold = decision.get("threshold") if isinstance(decision, dict) else None
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} has an invalid decision.threshold: "
            f"{threshold!r}."
        )
    threshold = float(threshold)
    if not math.isfinite(threshold) or not math.isclose(
        threshold, CONVEX_V3_THRESHOLD, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} declares decision.threshold="
            f"{threshold!r}, which does not match the frozen V3 threshold "
            f"{CONVEX_V3_THRESHOLD!r}."
        )

    label_mapping = raw.get("label_mapping")
    if label_mapping is not None and label_mapping != {"bonafide": 0, "spoof": 1}:
        raise ConvexFusionContractError(
            f"Fusion V3 contract at {json_path} declares label_mapping="
            f"{label_mapping!r}, expected {{'bonafide': 0, 'spoof': 1}}."
        )

    resolved_joblib_path: Path | None = None
    if joblib_path is not None and joblib_path.is_file():
        resolved_joblib_path = joblib_path
        _cross_validate_convex_v3_joblib(
            joblib_path,
            json_path=json_path,
            expected_weights_by_branch=weights_by_branch,
            expected_threshold=threshold,
        )

    return ConvexFusionContract(
        version=CONVEX_V3_VERSION,
        architecture=architecture,
        feature_order=tuple(feature_order_raw),
        weights=weights_by_branch,
        threshold=threshold,
        source_json_path=json_path,
        source_joblib_path=resolved_joblib_path,
    )


def _cross_validate_convex_v3_joblib(
    joblib_path: Path,
    *,
    json_path: Path,
    expected_weights_by_branch: dict[str, float],
    expected_threshold: float,
) -> None:
    """Best-effort: cross-check the portable joblib dict against the JSON
    contract already validated above. Skipped (not failed) when ``joblib``
    itself is not importable -- it is only guaranteed present alongside the
    ``glottal`` extra, not in a base install that never enables real Glottal.
    """

    try:
        import joblib
    except ImportError:
        logger.debug(
            "joblib is not installed; skipping Fusion V3 joblib artifact "
            "cross-validation for %s.",
            joblib_path,
        )
        return

    try:
        payload = joblib.load(joblib_path)
    except Exception as error:
        raise ConvexFusionContractError(
            f"Fusion V3 joblib artifact at {joblib_path} could not be loaded: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ConvexFusionContractError(
            f"Fusion V3 joblib artifact at {joblib_path} must be a portable "
            f"dict, not a {type(payload).__name__}."
        )
    if payload.get("version") != CONVEX_V3_VERSION:
        raise ConvexFusionContractError(
            f"Fusion V3 joblib artifact at {joblib_path} has version="
            f"{payload.get('version')!r}, which disagrees with {json_path}."
        )
    feature_names = payload.get("feature_names")
    if tuple(feature_names or ()) != CONVEX_V3_FEATURE_ORDER:
        raise ConvexFusionContractError(
            f"Fusion V3 joblib artifact at {joblib_path} has feature_names="
            f"{feature_names!r}, which disagrees with the JSON contract's "
            "feature_order."
        )
    joblib_weights = payload.get("weights")
    if not isinstance(joblib_weights, dict):
        raise ConvexFusionContractError(
            f"Fusion V3 joblib artifact at {joblib_path} is missing a weights dict."
        )
    for feature_name, branch_name in CONVEX_V3_FEATURE_TO_BRANCH.items():
        joblib_weight = joblib_weights.get(feature_name)
        expected = expected_weights_by_branch[branch_name]
        if isinstance(joblib_weight, bool) or not isinstance(joblib_weight, (int, float)):
            raise ConvexFusionContractError(
                f"Fusion V3 joblib artifact at {joblib_path} has a non-numeric "
                f"weight for {feature_name!r}: {joblib_weight!r}."
            )
        if not math.isclose(float(joblib_weight), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ConvexFusionContractError(
                f"Fusion V3 joblib artifact at {joblib_path} weight for "
                f"{feature_name!r} ({joblib_weight!r}) disagrees with the JSON "
                f"contract ({expected!r})."
            )
    joblib_threshold = payload.get("threshold")
    if isinstance(joblib_threshold, bool) or not isinstance(joblib_threshold, (int, float)):
        raise ConvexFusionContractError(
            f"Fusion V3 joblib artifact at {joblib_path} has an invalid "
            f"threshold: {joblib_threshold!r}."
        )
    if not math.isclose(float(joblib_threshold), expected_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ConvexFusionContractError(
            f"Fusion V3 joblib artifact at {joblib_path} threshold="
            f"{joblib_threshold!r} disagrees with the JSON contract "
            f"({expected_threshold!r})."
        )


@dataclass(frozen=True)
class FusionEngine:
    """Score-level fusion over completed branch predictions.

    The default branch weights are development defaults for exercising the
    pipeline. They are not research-validated and must not be reported as
    experimental findings.
    """

    method: FusionMethod = "weighted_average"
    branch_weights: dict[str, float] = field(
        default_factory=lambda: DEFAULT_DEVELOPMENT_WEIGHTS.copy()
    )
    spoof_threshold: float = 0.5
    minimum_successful_branches: int = 2
    required_branches: tuple[str, ...] = ()
    config_version: str = "fusion-config-v1"
    allow_equal_weight_fallback: bool = True
    #: Branches that must all contribute before a result may claim research
    #: eligibility. Defaults to the full designed system.
    full_system_branches: tuple[str, ...] = FULL_SYSTEM_BRANCHES
    #: Set when this engine's method/threshold came from the frozen research
    #: contract file rather than plain ``Settings`` fields. Informational
    #: (surfaced in provenance); does not change fusion behaviour itself.
    contract_version: str | None = None
    #: Human-facing fusion identity surfaced on every ``FusionResult`` this
    #: engine produces. Defaults preserve pre-V3 behaviour for any caller
    #: that never asked about this; ``from_settings`` overrides both to
    #: ``LEGACY_FUSION_VERSION``/``LEGACY_FUSION_MODE`` when the frozen
    #: 3-branch contract is active, since that is specifically the identity
    #: Fusion V3's explicit fallback must be reported under.
    fusion_version: str = DEVELOPMENT_FUSION_VERSION
    fusion_mode: str = DEVELOPMENT_FUSION_MODE

    def __post_init__(self) -> None:
        if self.method not in {"weighted_average", "simple_average", "majority_vote"}:
            raise ValueError(f"Unsupported fusion method: {self.method}")
        if not math.isfinite(self.spoof_threshold) or not 0.0 <= self.spoof_threshold <= 1.0:
            raise ValueError("Spoof decision threshold must be between 0 and 1.")
        if self.minimum_successful_branches < 1:
            raise ValueError("At least one successful branch must be required.")
        if not self.config_version.strip():
            raise ValueError("Fusion config version must be configured.")
        self._validate_weights(self.branch_weights)
        for branch in self.required_branches:
            canonical_branch_name(branch)

    @classmethod
    def from_settings(cls, app_settings: Settings) -> "FusionEngine":
        contract = app_settings.frozen_fusion_contract
        method = contract.fusion_method if contract is not None else app_settings.fusion_method
        threshold = (
            contract.decision_threshold
            if contract is not None
            else app_settings.fusion_decision_threshold
        )
        return cls(
            method=method,
            branch_weights=app_settings.fusion_weight_map,
            spoof_threshold=threshold,
            minimum_successful_branches=app_settings.fusion_min_successful_branches,
            required_branches=tuple(app_settings.fusion_required_branch_list),
            config_version=app_settings.fusion_config_version,
            allow_equal_weight_fallback=app_settings.fusion_allow_equal_weight_fallback,
            contract_version=contract.contract_version if contract is not None else None,
            fusion_version=(
                LEGACY_FUSION_VERSION if contract is not None else DEVELOPMENT_FUSION_VERSION
            ),
            fusion_mode=LEGACY_FUSION_MODE if contract is not None else DEVELOPMENT_FUSION_MODE,
        )

    def fuse(self, branches: list[BranchPrediction]) -> FusionResult:
        successful_branches = [
            branch
            for branch in branches
            if (
                branch.status == BranchStatus.success
                and branch.probabilities is not None
            )
        ]
        if self.contract_version is not None:
            # Belt-and-braces: when this engine was built from the frozen
            # research-detector contract, Glottal must never enter the
            # primary fusion arithmetic even if it were ever set to a real,
            # succeeding branch in the future -- auxiliary evidence only.
            successful_branches = [
                branch
                for branch in successful_branches
                if canonical_branch_name(branch.model_name) != "glottal"
            ]
        contains_dummy_branches = any(
            branch.mode == ModelMode.dummy for branch in successful_branches
        )
        excluded_branches = {
            branch.model_name: _branch_exclusion_reason(branch)
            for branch in branches
            if branch not in successful_branches
        }
        present_required = {
            canonical_branch_name(branch.model_name)
            for branch in successful_branches
        }
        missing_required = [
            branch for branch in self.required_branches if branch not in present_required
        ]
        if missing_required:
            return self._failed_result(
                contains_dummy_branches=contains_dummy_branches,
                warning="Required fusion branches are unavailable.",
                excluded_branches={
                    **excluded_branches,
                    **{branch: "required_branch_missing" for branch in missing_required},
                },
            )

        if len(successful_branches) < self.minimum_successful_branches:
            warning = (
                "At least two successful branches are required for fusion."
                if self.minimum_successful_branches == 2
                else (
                    f"At least {self.minimum_successful_branches} successful "
                    "branches are required for fusion."
                )
            )
            return self._failed_result(
                contains_dummy_branches=contains_dummy_branches,
                warning=warning,
                excluded_branches=excluded_branches,
            )

        if self.method == "majority_vote":
            probabilities, effective_weights = self._majority_vote(successful_branches)
        elif self.method == "simple_average":
            probabilities, effective_weights = self._simple_average(successful_branches)
        else:
            probabilities, effective_weights = self._weighted_average(
                successful_branches
            )
        if not effective_weights:
            return self._failed_result(
                contains_dummy_branches=contains_dummy_branches,
                warning="No positive fusion weights are available.",
                excluded_branches=excluded_branches,
            )

        prediction = (
            PredictionLabel.spoof
            if probabilities.spoof >= self.spoof_threshold
            else PredictionLabel.bonafide
        )
        confidence = (
            probabilities.spoof
            if prediction == PredictionLabel.spoof
            else probabilities.bonafide
        )

        blockers = self._research_blockers(
            successful_branches,
            contains_dummy_branches=contains_dummy_branches,
        )
        is_partial = "incomplete_branch_set" in blockers

        return FusionResult(
            status=BranchStatus.success,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            method=self.method,
            decision_threshold=self.spoof_threshold,
            branch_weights=effective_weights,
            contains_dummy_branches=contains_dummy_branches,
            eligible_for_research_evaluation=not blockers,
            warning=_fusion_warning(blockers),
            config_version=self.config_version,
            minimum_successful_branches=self.minimum_successful_branches,
            contributing_branches=[
                branch.model_name for branch in successful_branches
            ],
            excluded_branches=excluded_branches,
            system_stage="partial" if is_partial else "full",
            research_blockers=blockers,
            fusion_version=self.fusion_version,
            fusion_mode=self.fusion_mode,
            fallback_used=False,
        )

    def _research_blockers(
        self,
        successful_branches: list[BranchPrediction],
        *,
        contains_dummy_branches: bool,
    ) -> list[str]:
        """Every reason this result may not be reported as a research finding.

        Fail-closed: a real checkpoint is necessary but not sufficient. The
        branch set must be complete, no placeholder may have contributed, and
        each contributing branch must be attested (correct feature pipeline and
        correct label map). Any one of those missing invalidates the claim.
        """

        blockers: list[str] = []
        if contains_dummy_branches:
            blockers.append("dummy_branch_contributed")

        contributing = {
            canonical_branch_name(branch.model_name) for branch in successful_branches
        }
        if set(self.full_system_branches) - contributing:
            blockers.append("incomplete_branch_set")

        if any(
            branch.metadata.get("research_result") is not True
            for branch in successful_branches
        ):
            blockers.append("unverified_branch_contributed")
        return blockers

    def _weighted_average(
        self,
        branches: list[BranchPrediction],
    ) -> tuple[ProbabilityScores, dict[str, float]]:
        raw_weights = {
            branch.model_name: self._weight_for_branch(branch.model_name)
            for branch in branches
        }
        effective_weights = self._normalize_weights(raw_weights)
        if not effective_weights:
            return (
                ProbabilityScores(bonafide=0.5, spoof=0.5),
                {},
            )
        probabilities = self._average_probabilities(branches, effective_weights)
        return probabilities, effective_weights

    def _simple_average(
        self,
        branches: list[BranchPrediction],
    ) -> tuple[ProbabilityScores, dict[str, float]]:
        equal_weight = 1.0 / len(branches)
        effective_weights = {branch.model_name: equal_weight for branch in branches}
        probabilities = self._average_probabilities(branches, effective_weights)
        return probabilities, effective_weights

    def _majority_vote(
        self,
        branches: list[BranchPrediction],
    ) -> tuple[ProbabilityScores, dict[str, float]]:
        equal_weight = 1.0 / len(branches)
        effective_weights = {branch.model_name: equal_weight for branch in branches}
        votes = Counter(branch.prediction for branch in branches)
        spoof_votes = votes[PredictionLabel.spoof]
        bonafide_votes = votes[PredictionLabel.bonafide]

        if spoof_votes == bonafide_votes:
            probabilities = self._average_probabilities(branches, effective_weights)
            return probabilities, effective_weights

        spoof_probability = spoof_votes / len(branches)
        return (
            ProbabilityScores(
                bonafide=1.0 - spoof_probability,
                spoof=spoof_probability,
            ),
            effective_weights,
        )

    def _average_probabilities(
        self,
        branches: list[BranchPrediction],
        effective_weights: dict[str, float],
    ) -> ProbabilityScores:
        spoof = 0.0
        bonafide = 0.0
        for branch in branches:
            weight = effective_weights[branch.model_name]
            probabilities = branch.probabilities
            if probabilities is None:
                continue
            spoof += probabilities.spoof * weight
            bonafide += probabilities.bonafide * weight
        total = spoof + bonafide
        if total <= 0:
            return ProbabilityScores(bonafide=0.5, spoof=0.5)
        return ProbabilityScores(bonafide=bonafide / total, spoof=spoof / total)

    def _failed_result(
        self,
        *,
        contains_dummy_branches: bool,
        warning: str,
        excluded_branches: dict[str, str] | None = None,
    ) -> FusionResult:
        return FusionResult(
            status=BranchStatus.failed,
            prediction=None,
            confidence=None,
            probabilities=None,
            method=self.method,
            decision_threshold=self.spoof_threshold,
            branch_weights={},
            contains_dummy_branches=contains_dummy_branches,
            eligible_for_research_evaluation=False,
            warning=warning,
            config_version=self.config_version,
            minimum_successful_branches=self.minimum_successful_branches,
            contributing_branches=[],
            excluded_branches=excluded_branches or {},
            fusion_version=self.fusion_version,
            fusion_mode=self.fusion_mode,
            fallback_used=False,
        )

    def _weight_for_branch(self, model_name: str) -> float:
        aliases = _branch_aliases(model_name)
        for alias in aliases:
            if alias in self.branch_weights:
                return self.branch_weights[alias]
        return 0.0

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        total_weight = sum(weights.values())
        if total_weight <= 0:
            if not self.allow_equal_weight_fallback:
                return {}
            equal_weight = 1.0 / len(weights)
            return {branch: equal_weight for branch in weights}
        return {branch: weight / total_weight for branch, weight in weights.items()}

    @staticmethod
    def _validate_weights(weights: dict[str, float]) -> None:
        if not weights:
            raise ValueError("At least one branch weight must be configured.")
        for branch, weight in weights.items():
            if branch not in RECOGNIZED_BRANCHES:
                raise ValueError(f"Unrecognized branch weight: {branch}")
            if not math.isfinite(weight):
                raise ValueError(f"Branch weight must be finite: {branch}")
            if weight < 0:
                raise ValueError(f"Branch weight cannot be negative: {branch}")


def _fusion_warning(blockers: list[str]) -> str | None:
    """Most specific applicable warning; dummy output is the worst case."""

    if "dummy_branch_contributed" in blockers:
        return DUMMY_FUSION_WARNING
    if "unverified_branch_contributed" in blockers:
        return UNVERIFIED_BRANCH_WARNING
    if "incomplete_branch_set" in blockers:
        return PARTIAL_SYSTEM_WARNING
    return None


def _branch_aliases(model_name: str) -> tuple[str, ...]:
    if model_name in {"cnn", "cnn_acoustic"}:
        return ("lfcc_cnn_tcn", "cnn_acoustic", "cnn")
    if model_name == "aasist":
        return ("aasist",)
    if model_name in {"ssl", "ssl_wavlm_xlsr", "ssl_sequence"}:
        return ("ssl_sequence", "ssl_wavlm_xlsr", "ssl")
    if model_name in {"glottal", "glottal_features"}:
        return ("glottal", "glottal_features")
    return (model_name,)


def _branch_exclusion_reason(branch: BranchPrediction) -> str:
    if branch.status != BranchStatus.success:
        if canonical_branch_name(branch.model_name) == "glottal":
            return branch.metadata.get("error_code") or "auxiliary_evidence_only"
        return branch.metadata.get("error_code") or "branch_failed"
    if branch.probabilities is None:
        return "missing_probabilities"
    if canonical_branch_name(branch.model_name) == "glottal":
        return "auxiliary_not_used_in_primary_fusion"
    return "excluded"


@dataclass(frozen=True)
class ConstrainedFourBranchFusion:
    """Fusion V3: frozen constrained convex weighted fusion over all four
    branches (CNN + AASIST + SSL + Glottal).

    Deliberately requires every one of the four branches to be present,
    ``success``, and in ``ModelMode.real`` before it computes anything -- a
    request missing any of them (branch failed, branch disabled, or a
    development ``dummy`` placeholder standing in for a real branch) gets an
    honest ``status=failed`` result naming exactly what is missing, never a
    fabricated 3-branch (or fewer) substitute under the V3 name. Requiring
    real mode specifically (not just ``success``) keeps this frozen,
    research-calibrated contract from ever being exercised over deterministic
    placeholder scores and reported as if it were the fitted detector.

    ``VoiceService`` is the only caller: it tries this first and, only on a
    failed result, falls back to the legacy ``FusionEngine`` -- see
    ``VoiceService._fuse``.
    """

    contract: ConvexFusionContract

    @property
    def branch_weights(self) -> dict[str, float]:
        return dict(self.contract.weights)

    def fuse(self, branches: list[BranchPrediction]) -> FusionResult:
        by_canonical: dict[str, BranchPrediction] = {}
        for branch in branches:
            try:
                canonical = canonical_branch_name(branch.model_name)
            except ValueError:
                continue
            by_canonical[canonical] = branch

        excluded_branches: dict[str, str] = {}
        ordered_present: list[BranchPrediction] = []
        for canonical in CONVEX_V3_BRANCH_ORDER:
            candidate = by_canonical.get(canonical)
            if candidate is None:
                excluded_branches[canonical] = "branch_missing"
                continue
            if candidate.status != BranchStatus.success or candidate.probabilities is None:
                excluded_branches[candidate.model_name] = _v3_branch_exclusion_reason(candidate)
                continue
            if candidate.mode != ModelMode.real:
                excluded_branches[candidate.model_name] = "not_real_mode"
                continue
            ordered_present.append(candidate)

        if len(ordered_present) < len(CONVEX_V3_BRANCH_ORDER):
            return self._failed_result(excluded_branches)

        spoof = sum(
            self.contract.weights[canonical] * prediction.probabilities.spoof
            for canonical, prediction in zip(
                CONVEX_V3_BRANCH_ORDER, ordered_present, strict=True
            )
        )
        # Numerical-safety clamp only (weights sum to 1, each branch score is
        # already in [0, 1], so this only ever corrects sub-ulp float noise
        # that would otherwise fail ProbabilityScores' [0, 1] bounds) -- not
        # renormalisation and not a calibration step.
        spoof = min(1.0, max(0.0, spoof))
        bonafide = 1.0 - spoof

        # Strict `>`, not `>=`: `FusionResult`'s own validator independently
        # derives a "winning label" as argmax(bonafide, spoof) and requires
        # `prediction` to match it. Since the frozen V3 threshold is exactly
        # 0.5, `spoof >= threshold` and that argmax agree everywhere except
        # the exact spoof == bonafide == 0.5 tie, where argmax favours
        # bonafide (Python `max` keeps the first-seen maximum, and bonafide
        # is listed first) -- `>=` would build an internally-inconsistent
        # `FusionResult` there and crash the request with a validation error
        # instead of returning a verdict. `>` matches the validator exactly
        # at every input, including that tie.
        prediction_label = (
            PredictionLabel.spoof if spoof > self.contract.threshold else PredictionLabel.bonafide
        )
        confidence = spoof if prediction_label == PredictionLabel.spoof else bonafide

        blockers: list[str] = []
        if any(
            prediction.metadata.get("research_result") is not True
            for prediction in ordered_present
        ):
            blockers.append("unverified_branch_contributed")

        return FusionResult(
            status=BranchStatus.success,
            prediction=prediction_label,
            confidence=confidence,
            probabilities=ProbabilityScores(bonafide=bonafide, spoof=spoof),
            method="convex_weighted",
            decision_threshold=self.contract.threshold,
            branch_weights={
                prediction.model_name: self.contract.weights[canonical]
                for canonical, prediction in zip(
                    CONVEX_V3_BRANCH_ORDER, ordered_present, strict=True
                )
            },
            contains_dummy_branches=False,
            eligible_for_research_evaluation=not blockers,
            warning=_fusion_warning(blockers),
            config_version=self.contract.version,
            minimum_successful_branches=len(CONVEX_V3_BRANCH_ORDER),
            contributing_branches=[prediction.model_name for prediction in ordered_present],
            excluded_branches={},
            system_stage="full",
            research_blockers=blockers,
            fusion_version=CONVEX_V3_VERSION,
            fusion_mode=CONVEX_V3_MODE,
            fallback_used=False,
        )

    def _failed_result(self, excluded_branches: dict[str, str]) -> FusionResult:
        return FusionResult(
            status=BranchStatus.failed,
            prediction=None,
            confidence=None,
            probabilities=None,
            method="convex_weighted",
            decision_threshold=self.contract.threshold,
            branch_weights={},
            contains_dummy_branches=False,
            eligible_for_research_evaluation=False,
            warning=(
                "Fusion V3 requires all four branches (CNN, AASIST, SSL, "
                "Glottal) to succeed in real mode; at least one did not. "
                "No partial/renormalised V3 score is computed."
            ),
            config_version=self.contract.version,
            minimum_successful_branches=len(CONVEX_V3_BRANCH_ORDER),
            contributing_branches=[],
            excluded_branches=excluded_branches,
            system_stage="partial",
            research_blockers=["incomplete_branch_set"],
            fusion_version=CONVEX_V3_VERSION,
            fusion_mode=CONVEX_V3_MODE,
            fallback_used=False,
        )


def _v3_branch_exclusion_reason(branch: BranchPrediction) -> str:
    if branch.status != BranchStatus.success:
        return branch.metadata.get("error_code") or "branch_failed"
    if branch.probabilities is None:
        return "missing_probabilities"
    return "excluded"
