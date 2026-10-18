# Voice XAI module

`app.voice_xai` is the backend's feature-owned explainability module. It is a
downstream consumer of classifier evidence: it must not influence a classifier
prediction or import a concrete classifier adapter.

## Ownership boundaries

```text
app/api                 HTTP authentication and owner-scoped routes
app/services            prediction workflow and XAI enqueue integration
app/models              model-specific inference and original-pass temporal evidence
app/voice_xai           explanation analysis, reporting, and storage
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
  __init__.py           package initialization and public exports
  integration/
    contracts.py        classifier-to-XAI handoff contract
  workflow/
    orchestrator.py     explanation-run lifecycle and component orchestration
    queue.py            bounded post-prediction XAI job scheduling
    status.py           independent XAI status transitions
    recovery.py         startup reconciliation of interrupted explanation runs
    replay.py           stored source-audio preparation for manual runs/retries
  temporal/             XLS-R temporal attention analysis and visualisation
  semantic/             acoustic feature schema and XGBoost/SHAP explanations
  evaluation/           consistency metrics and evaluation eligibility policy
  report/               deterministic reports and optional Qwen AI narration
  persistence/          XAI repository contracts and MongoDB implementation
  artifacts/            private, hash-verified evidence artifact storage
```

Model training runs separately from the runtime package. Semantic inference
loads validated model exports from the configured fixed model directory.

Semantic audio extraction lives in `semantic/feature_extraction.py`.
`AcousticFeatureExtractor.extract_file()` decodes and cleans the original clip
once, then `extract_feature_mapping()` writes the 165 ordered model inputs through
Praat, cepstral, pitch, energy, spectral, CPP/pause and Glottal helpers. `schema.py` defines
the single ordered feature list. Glottal selection, validation and acoustic-value
precedence rules live in `feature_extraction.py`; DisVoice output names are
selected from the final schema. The trained calculations and error handling
are retained, including validation of the three shared acoustic names.
`service.py` then applies the fitted imputer, XGBoost regressor and Tree SHAP;
`explainer.py` formats the semantic response.

Optional semantic localization uses `SemanticWindowOptions` (defaults: 2-second
windows, 1-second hop) over the temporal visualization's complete `analyzed_spans`.
The extractor retains source trim offsets, decodes once for combined analysis,
and never trims, normalizes or pads a window again. End-anchored final windows
cover tails; spans shorter than the window use real samples only when at least
0.2 seconds remain. The frozen full-clip predictor stays separate from the
explicit window input contract.

Window activity is the sum of absolute SHAP values. Overlaps are averaged on
exact window boundaries, and the percentile uses duration weighting. Failed or
budget-skipped window supports are removed conservatively, including overlap
with successful windows. IoU intersects both region sets with common valid
coverage and remains unavailable for missing coverage or an empty region union.
Research eligibility is not granted by a high IoU or window-validation metadata.
Detailed contracts and validation procedure are in
`docs/semantic-window-localization.md` at the service root.

## Current request and response flow

1. `POST /api/v1/predictions` authenticates the user and calls
   `PredictionSubmissionService.submit`.
2. `PredictionSubmissionService.execute_prediction` validates/stores the
   upload, then calls `VoiceService`.
3. `VoiceService` preprocesses audio once and runs each classifier branch.
   During a real XLS-R prediction, the adapter
   reduces its own attention to a compact time-aligned evidence object and
   discards the raw attention matrices before the branch worker returns.
4. `VoiceService` returns an internal `PredictionExecutionResult` containing
   the public classifier prediction, the same ephemeral `ProcessedAudio`, and
   optional reduced XLS-R temporal evidence. CNN/AASIST intermediate layer
   outputs are no longer copied or persisted.
5. The prediction is persisted before `PredictionSubmissionService` calls
   `VoiceXaiOrchestrator.enqueue`. XAI errors cannot turn a successful
   prediction into a failed one.
6. When XLS-R temporal evidence is available, the orchestrator
   stores its compact timeline as a private, hash-verified retry artifact
   before queueing work. Its reference is hidden from public XAI artifact
   lists; raw attention matrices are never stored. A manual retry loads this
   compact artifact and therefore does not rerun XLS-R.
7. `workflow.queue.AsynchronousExplanationQueue` runs the explanation in the
   background. The orchestrator records an independent explanation run.
8. The temporal service consumes only reduced XLS-R evidence and creates a
   private visualisation artifact; it never reruns XLS-R. The classifier's
   six-second centre crop retains its offset in the original audio timeline.
   The
   semantic service runs the 165-feature teacher surrogate on
   a single full cleaned waveform decoded from the original upload bytes. The
   acoustic and Glottal measurements share that waveform without duration cropping.
   The fitted median imputer and regressor come
   from the fixed model directory at startup. Tree SHAP explains its raw regression
   score; the clip probability is clipped to [0,1].
   All 165 contributions are clip-level, without semantic windows. Quality
   reads this model's optional fidelity metadata; evaluation eligibility remains
   fail-closed. Replace matching model exports in the configured directory and
   restart to load a retrained model.
9. The report service composes the combined report. Safe metadata goes to the
   `xai_explanations` repository; visual artefacts are stored only
   in the private artifact store.
10. Owner-scoped explanation endpoints return the status, report, temporal and
   semantic summaries. Artifact downloads verify ownership, expiry, size, and
   SHA-256 integrity before returning bytes.

## Optional AI narrative

Optional AI narratives require `summary`, `detailed_explanation`, and a non-empty,
unique `evidence_references` array. The prompt includes the response JSON schema
and allowed reference strings; local validation still checks every response.
An invalid structured response gets at most one automatic correction attempt
using the same saved evidence. Provider errors and timeouts are not automatically
retried. Each request is bounded by `XAI_NARRATIVE_TIMEOUT_SECONDS`; both attempts
share a total budget of twice that value, capped by the remaining XAI job budget
for the initial run. The narrative-only retry endpoint receives the same bounded
generation budget. This does not repeat classification or acoustic extraction.
If both responses are invalid, narrative status remains failed and the
deterministic report stays available. Logs contain rejection reasons and retry
outcomes without generated content or evidence values.

## User-facing temporal evidence

The completed temporal response provides label-neutral `high_attention_regions`
using the pure per-clip 90th percentile of the aggregated rollout density. There
is no density floor or external threshold artifact. The cutoff is computed with
linear interpolation, rounded to float32, and compared using `>=`. Explanation
and visualization fields contain the same result. Ties and postprocessing mean
coverage is not necessarily 10%; flat maps can select all covered bins. Each region
includes start/end times, maximum and mean attention density, duration,
percentage of full audio, `start_sample`, and `end_sample_exclusive`.
Samples refer to the classifier-preprocessed mono waveform at `sample_rate_hz`,
not the original file's sampling rate. These regions are model-focus evidence,
not ground-truth fake-audio percentages.

The attention visualization JSON saves `raw_high_attention_mask` (also exposed
under the legacy `high_attention_mask` name) and `final_region_mask` separately.
The final mask includes merged gaps and excludes filtered short regions. Both
masks align with `time_seconds`; unavailable attention remains unselected.
`highlighted_samples` counts the union of final intervals rounded to sample
boundaries. `highlighted_duration_seconds` and `highlighted_coverage_percentage`
use that count and the full `audio_sample_count`, including unanalysed audio in
the denominator. `raw_high_attention_bin_percentage` instead uses only available
attention bins. `flat_attention_map` tests equality across those available scores.
These summaries are saved in the temporal API response as well.

Temporal `research_eligible` is true for validated original-pass evidence from a
successful real SSL branch whose original prediction metadata records
`preprocessing_verified`, `class_mapping_verified`, and `research_result` as true.
Otherwise `research_blockers` explains the missing evidence. Changing current
settings does not retrospectively attest an old prediction. Overall XAI research
eligibility still requires the classifier and semantic components to be eligible.
Policy `voice-xai-research-eligibility-v2-clip-semantic` accepts completed real
clip-level semantic inference with recorded held-out surrogate fidelity. The
semantic result wrapper carries the same eligibility as its explanation.
Experimental window localization keeps its own validation status and does not
block clip-level eligibility. This flag does not certify localization accuracy
or that attention/SHAP beats random under every perturbation protocol.

`evaluation/eligibility.py` records selected fixed offline AOPC references from
`LA_D_2302632.wav`: semantic raw SHAP MoRF AOPC at K=20 is
`0.517231721941027`; temporal attention AOPC over 10–50% silence is
`0.1838843630005916`. The source runs, audio hash, model fingerprints and random
comparators are retained. Temporal random mean `0.2708303292786392` exceeds
attention in this reference run; its negative advantage is preserved.
New explanation `quality.semantic_aopc` and `quality.temporal_aopc` fields label
these as `offline_validation` references, not measurements on the current upload.
They do not override failed, missing, placeholder or unverified component gates.

Previously the real temporal service hardcoded false even for verified evidence.
Restart the backend after updating SSL verification flags and create a new
prediction/explanation to record the verified provenance. Legacy explanations
remain readable with absent metadata represented as null; they are not rewritten.
Temporal model version is read from the original classifier provenance instead
of a fixed ASVspoof2019 label.

When processed audio is available, the temporal artifacts include both the
attention time-series JSON and an `attention_spectrogram` JSON payload. The
latter contains a bounded log-mel matrix, frame-aligned continuous attention
density, and high-attention region markers for a frontend to draw as a heatmap.
It is derived from the same preprocessed mono waveform
consumed by the classifier.

Classifier crops retain their offset in the original audio. Both graph artifacts
use source-audio coordinates and explicit analyzed spans; outside those spans,
attention values are `null`, not zero or repeated endpoints. The frontend draws
the full spectrogram normally and adds attention only inside analyzed spans.
Attention curves and their display smoothing do not connect across unknown gaps.
The pure-p90 cutoff is calculated only from covered evidence bins.
