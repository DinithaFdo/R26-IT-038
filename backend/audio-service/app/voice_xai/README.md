# Voice XAI module

`app.voice_xai` is the backend's feature-owned explainability module. It is a
downstream consumer of classifier evidence: it must not influence a classifier
prediction or import a concrete classifier adapter.

## Ownership boundaries

```text
app/api                 HTTP authentication and owner-scoped routes
app/services            prediction workflow and XAI enqueue integration
app/models              model-specific inference and validated capture targets
app/voice_xai           explanation capture, analysis, reporting, and storage
```

The permitted runtime direction is:

```text
API -> prediction services -> models + voice_xai -> persistence/artifacts
```

`voice_xai` receives the versioned `ClassifierInferenceBundle`. It must not
reach into the model registry or change prediction state.

## Package map

```text
voice_xai/
  contracts.py          classifier-to-XAI handoff contract
  orchestrator.py       explanation-run lifecycle and component orchestration
  status.py             independent XAI status transitions
  capture/              bounded request-scoped PyTorch representation capture
  jobs/                 bounded post-prediction XAI job scheduling
  temporal/             XLS-R temporal attention analysis and visualisation
  semantic/             acoustic feature schema and XGBoost/SHAP explanations
  evaluation/           consistency metrics and research-readiness policy
  report/               deterministic unified forensic report composition
  persistence/          XAI repository contracts and MongoDB implementation
  artifacts/            private, hash-verified evidence artifact storage
```

Research notebooks are intentionally outside the runtime package at
`backend/notebooks/voice_xai/research/`.

## Current request and response flow

1. `POST /api/v1/predictions` authenticates the user and calls
   `PredictionSubmissionService.submit`.
2. `PredictionSubmissionService.execute_prediction` validates/stores the
   upload, then calls `VoiceService`.
3. `VoiceService` preprocesses audio once and runs each classifier branch.
   If opt-in representation capture is enabled, it opens a request-scoped
   `capture.PyTorchExtractionInterface` session around the relevant real
   branch forward pass. During a real XLS-R prediction, the adapter also
   reduces its own attention to a compact time-aligned evidence object and
   discards the raw attention matrices before the branch worker returns.
4. `VoiceService` returns an internal `PredictionExecutionResult` containing
   the public classifier prediction, the same ephemeral `ProcessedAudio`, any
   bounded `ExtractionBundle`, and optional reduced XLS-R temporal evidence.
5. The prediction is persisted before `PredictionSubmissionService` calls
   `VoiceXaiOrchestrator.enqueue`. XAI errors cannot turn a successful
   prediction into a failed one.
6. When XLS-R temporal evidence is available, the orchestrator
   stores its compact timeline as a private, hash-verified retry artifact
   before queueing work. Its reference is hidden from public XAI artifact
   lists; raw attention matrices are never stored. A manual retry loads this
   compact artifact and therefore does not rerun XLS-R.
7. `jobs.AsynchronousExplanationQueue` runs the explanation in the
   background. The orchestrator records an independent explanation run.
8. The temporal service consumes only reduced XLS-R evidence and creates a
   private visualisation artifact; it never reruns XLS-R. For clips longer
   than six seconds, that evidence is created in the classifier worker from
   six-second windows with a three-second stride and cosine-weighted overlap
   fusion. The
   semantic service uses the v4 notebook-compatible
   148-feature extraction recipe and hash-verified XGBoost/TreeSHAP artifacts
   to produce whole-clip or windowed evidence. It uses ingestion's bounded
   pre-normalisation waveform so feature values match the training recipe.
   Evaluation calculates available consistency evidence. Research eligibility
   remains fail-closed.
9. The report service composes the combined report. Safe metadata goes to the
   `xai_explanations` repository; tensors and visual artefacts are stored only
   in the private artifact store.
10. Owner-scoped explanation endpoints return the status, report, temporal and
   semantic summaries. Artifact downloads verify ownership, expiry, size, and
   SHA-256 integrity before returning bytes.

## User-facing temporal evidence

The completed temporal response provides label-neutral `high_attention_regions`
whenever rollout density passes the calibrated threshold. Each includes exact
start/end times and its attention score. These regions are model-focus evidence,
not ground-truth fake-audio percentages.

When processed audio is available, the temporal artifacts include both the
attention time-series JSON and an `attention_spectrogram` JSON payload. The
latter contains a bounded log-mel matrix, frame-aligned continuous attention
density, and high-attention region markers for a frontend to draw as a heatmap.
It is derived from the same preprocessed mono waveform
consumed by the classifier.
