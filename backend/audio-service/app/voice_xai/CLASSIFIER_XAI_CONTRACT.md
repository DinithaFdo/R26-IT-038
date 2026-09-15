# Classifier-to-XAI Contract

Contract version: `classifier-xai-v1`

This document freezes the boundary between the MULTI-SCOPE classifier and the
Voice XAI component. Changes to the fields or meanings marked as frozen require
an explicit contract-version change and coordinated classifier/XAI review.

## Ownership boundary

- The classifier owns audio preprocessing, branch model loading and inference,
  score-level fusion, classifier provenance, and the classifier result.
- Voice XAI owns capture plans, extracted representations, temporal and semantic
  explanations, explanation-quality evidence, explanation artifacts, and the
  combined report.
- Voice XAI consumes classifier-level contracts. It must not import concrete
  dummy or real model adapter classes.
- The classifier result is persisted before explanation work begins.
- Explanation failure never changes a completed classifier result to failed.

## Frozen identifiers

Canonical branch IDs are used by internal integration and XAI:

| Canonical branch ID | Public `model_name` compatibility alias |
| --- | --- |
| `lfcc_cnn_tcn` | `cnn_acoustic` |
| `aasist` | `aasist` |
| `ssl_sequence` | `ssl_wavlm_xlsr` |
| `glottal` | `glottal_features` |

`ssl_sequence` is deliberately architecture-neutral. A future XLSR, WavLM,
BiLSTM, Mamba, or other sequence-head implementation does not rename the XAI
branch contract.

## Frozen classifier meanings

- Labels are `bonafide` and `spoof`.
- `probabilities.spoof` always means the classifier score/probability for the
  spoof class.
- `probabilities.bonafide` always means the classifier score/probability for the
  bonafide class.
- Branch statuses are `success`, `failed`, and `skipped`.
- Model modes are `dummy` and `real`.
- Dummy outputs are development placeholders and are never research results.
- Mode-specific details belong in metadata and do not change the stable result
  shape.
- Existing `BranchPrediction`, `FusionResult`, `VoicePredictionResponse`, and
  `PredictionSubmissionResponse` fields retain their current meanings.

## Internal handoff bundle

The XAI orchestration boundary is `ClassifierInferenceBundle`, containing:

- `contract_version`
- `prediction_id`
- `request_id`
- `owner_user_id`
- `source_type`
- classifier-level `VoicePredictionResponse`
- an optional Phase 1 `ExtractionBundle`
- optional reduced XLS-R temporal evidence created in the classifier worker;
  clips longer than the trained 6-second input use the calibrated overlapping
  window pipeline

When the internal prediction caller explicitly enables capture, the bundle is
populated from request-scoped PyTorch hooks in the same branch worker that ran
classifier inference. Hook handles are removed before the worker returns.
Captured tensors remain bounded by the configured total-element budget and are
serialized by the asynchronous XAI worker as a private NPZ artifact only after
the classifier result has been persisted. They are never part of a public
prediction or explanation JSON payload.

Reduced XLS-R temporal evidence is separate from the generic tensor-capture
bundle. It contains only the compact time-aligned density needed by temporal
XAI; raw per-head attention matrices are reduced and discarded in the same
classifier worker that created the XLS-R branch prediction. The asynchronous
XAI worker must consume this evidence and must not rerun XLS-R.

After the classifier result is persisted, the orchestrator stores available
reduced temporal evidence as a hash-verified private retry artifact associated
with its XAI run. Its reference is hidden from public artifact lists. A manual
retry may load this compact evidence to repeat temporal analysis without
rerunning XLS-R; if it is expired, missing, or fails integrity validation,
temporal analysis is unavailable while other XAI components may still run.

The bundle is internal and must not expose raw tensors through a public API.
Large tensor artifacts must be stored separately and referenced by private,
authorized artifact metadata.

## Additive XAI contract

XAI state is independent of `PredictionStatus`. Its top-level statuses are:

- `queued`
- `running`
- `partial`
- `completed`
- `failed`
- `blocked`

Temporal, semantic, and report component states are stored separately. Public
XAI responses use schema version `voice-xai-api-v1`; stored explanation runs
record both schema and pipeline versions.

## Status transition policy

- Every explanation and component starts as `queued`.
- Starting component work moves the explanation to `running`.
- A completed, failed, or unavailable input component makes the explanation
  `partial` while a report can still be generated.
- A blocked component makes the explanation `blocked` only when no component
  has produced terminal output and no other component is running.
- Blocked components must be explicitly requeued before work resumes.
- Temporal or semantic failure does not fail the classifier or the explanation
  run; the combined report may complete with limitations.
- The explanation becomes `completed` only when temporal and semantic inputs
  are terminal and the required combined report is completed.
- The explanation becomes `failed` when the required report fails or is not
  available.
- Failed, blocked, and unavailable component states require a sanitized error
  code and message. Raw exceptions and stack traces are prohibited.
- Completed and failed explanation runs are immutable. A retry creates a new
  versioned run rather than rewriting terminal evidence.

The following are additive and do not change the classifier contract:

- an XAI explanation identifier and status summary on a prediction;
- an independent `xai_explanations` persistence record;
- temporal, semantic, report, quality, artifact, and XAI provenance fields;
- future versioned XAI endpoints.

## Prohibited exposure

Public XAI contracts must not expose full checkpoint paths, local upload paths,
raw exceptions or stack traces, secret configuration, or raw attention tensors
unless a separately authorized artifact endpoint is deliberately implemented.

## Compatibility test requirements

Contract tests must verify:

- canonical branch order and public alias mapping;
- spoof/bonafide probability meanings;
- dummy results remain research-ineligible;
- the handoff bundle rejects mismatched request IDs;
- explanation status is independent from prediction status;
- persisted XAI records retain schema and pipeline versions.
