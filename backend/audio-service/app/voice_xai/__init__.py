"""Voice XAI components kept independent from classifier runtime adapters.

The package intentionally has no FastAPI routes or persistence side effects.
Those are added only after the classifier-to-XAI integration contract is frozen.
"""

from app.voice_xai.semantic.features import (
    FEATURE_NAMES,
    AcousticFeatureExtractor,
    FeatureExtractionConfig,
    FeatureExtractionResult,
)
from app.voice_xai.jobs.queue import AsynchronousExplanationQueue
from app.voice_xai.integration.contracts import (
    CANONICAL_XAI_BRANCHES,
    CLASSIFIER_XAI_CONTRACT_VERSION,
    PUBLIC_MODEL_NAME_BY_XAI_BRANCH,
    ClassifierInferenceBundle,
)
from app.voice_xai.workflow.status import (
    XaiExplanationNotFoundError,
    XaiStatusLifecycleError,
    XaiStatusService,
    derive_explanation_status,
)

__all__ = [
    "FEATURE_NAMES",
    "AcousticFeatureExtractor",
    "FeatureExtractionConfig",
    "FeatureExtractionResult",
    "AsynchronousExplanationQueue",
    "CANONICAL_XAI_BRANCHES",
    "CLASSIFIER_XAI_CONTRACT_VERSION",
    "PUBLIC_MODEL_NAME_BY_XAI_BRANCH",
    "ClassifierInferenceBundle",
    "XaiExplanationNotFoundError",
    "XaiStatusLifecycleError",
    "XaiStatusService",
    "derive_explanation_status",
]
