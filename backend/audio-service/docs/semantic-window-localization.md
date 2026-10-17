# Semantic window localization

The full-clip explanation remains the output of the trained 165-feature
surrogate. A separate localization result describes the same surrogate's
responses to temporal-aligned windows. Window SHAP explains each window's raw
score; it is not an additive time decomposition of the full-clip prediction.

## Runtime flow

1. The orchestrator passes the temporal visualization's complete `analyzed_spans`
   and job deadline to the semantic service. High-attention regions do not
   determine which windows are measured.
2. Original upload bytes are decoded once when both full-clip and window
   extraction are needed. Shared full-clip cleaning retains the resampled
   source length and trim start/end sample offsets. Source coordinates remain
   anchored to the original recording, including leading silence.
3. Window planning intersects temporal coverage with the cleaned waveform.
   It uses integer sample bounds, never crosses evidence gaps, and never pads.
   The last complete window is anchored at each span's end. A shorter span
   contributes one real, unpadded window if it contains at least 0.2 seconds.
4. Each slice uses the existing feature calculations with no additional trim
   or normalization. The frozen imputer, regressor and Tree SHAP explain it
   through an explicit window extraction contract. If a slice is exactly the
   complete cleaned waveform measured in the same call, its raw feature vector
   is reused without repeating DSP. Arbitrary precomputed vectors are not reused
   for this optimization without a matching measured source.
5. A window's activity is `sum(abs(shap_values))`. Successful overlapping windows
   are averaged on exact window boundaries. Failed and unprocessed supports
   are excluded conservatively, including portions overlapping a success.
6. A duration-weighted inverse-CDF percentile (90 by default) selects positive
   activity bins. Only adjacent selected bins are merged. Unknown gaps remain
   gaps. Constant positive activity can select all valid coverage; the response
   reports `flat_activity_map` rather than inventing fine localization.
7. IoU is intersection duration divided by union duration after clipping both
   region sets to common valid coverage. It describes agreement with label-neutral
   XLS-R high attention. It is not fake-audio localization accuracy.

The API adds `semantic.localization` with source intervals, window contributions,
valid spans, coverage, activity threshold, regions, timing and validation status.
Its optional default keeps old stored explanations readable. Missing source
bytes or absent valid windows cannot fabricate a timeline. Window failures do
not replace a successful full-clip explanation with synthetic evidence.
Localization is partial whenever only part of the requested temporal coverage
has valid semantic support, including portions removed by full-clip trimming.

## Configuration

| Setting | Default | Meaning |
| --- | ---: | --- |
| `XAI_SEMANTIC_WINDOWS_ENABLED` | `true` | Add localization when temporal coverage is available |
| `XAI_SEMANTIC_WINDOW_SECONDS` | `2.0` | Window context duration |
| `XAI_SEMANTIC_HOP_SECONDS` | `1.0` | Distance between window starts; must not exceed duration |
| `XAI_SEMANTIC_MAX_WINDOWS` | `32` | Reject an oversized plan before expensive extraction |
| `XAI_SEMANTIC_WINDOW_BUDGET_SECONDS` | `120.0` | Cooperative additional-window stage limit |
| `XAI_SEMANTIC_ACTIVITY_PERCENTILE` | `90.0` | Independent semantic region-selection percentile |

All behavior settings, the window extraction contract and localization method
version participate in the explanation configuration hash. Change settings and
restart to construct the service with the new options. A job deadline is checked
between windows and after extraction/inference; an active DSP/library call is
not forcibly interrupted. The stage limit is not a hard end-to-end latency cap.

The requested performance target is 10–12 seconds for XAI. This remains a
measurement goal, not an established latency guarantee. Full-clip extraction,
temporal work, storage and reporting also consume time. For the current 6-second
temporal crop, 2s/1s creates five windows, regardless of source duration outside
that crop; the full-clip extraction still measures the complete cleaned source.

An initial real-audio smoke check used `LA_D_2130605.wav` (3.6218125 seconds),
the existing production model, initialized dependencies and one native thread.
After exact-vector reuse, one warm 2s/1s semantic-stage run took **32.302 seconds**;
the added window stage took **0.032 seconds**. The cleaned supported duration
was only 38.876% of the source-wide requested span, producing one shorter real
window and a partial localization result. This was not a full temporal-model/XAI
run, repeated latency study, localization accuracy measurement or winning-config
selection. The 10–12 second target was **not met**, even by this semantic stage.
Further performance work must address full-clip DSP cost as well as window count.

## Model validity

Existing model exports remain loadable without window metadata. Such window
results explicitly report `model_validation_status=experimental` and no
window-level fidelity value. Clip-level R² is never reused as window fidelity.

Optional `training_metadata.json.window_validation` must contain this structure:

```json
{
  "contract": {
    "version": "shared-clean-window-audio-v1",
    "preprocessing_scope": "shared_full_cleaned_clip",
    "window_trim": false,
    "window_normalization": false,
    "padding": false,
    "tail_policy": "end_anchored_short_unpadded",
    "time_reference": "source_audio",
    "window_seconds": 2.0,
    "hop_seconds": 1.0
  },
  "split_unit": "recording",
  "target": "teacher_window_spoof_probability",
  "n_recordings": 100,
  "n_windows": 500,
  "r2": 0.8,
  "mae": 0.1,
  "rmse": 0.15
}
```

The numeric values above illustrate the format, not measured results or pass
thresholds. Supply actual held-out metrics. `split_unit` may also be `speaker`.
Changing the window/hop recipe makes metadata for another recipe inapplicable.
Metadata does not grant research eligibility or turn overlap into accuracy.
If the existing model performs poorly on held-out windows, fit/refit the
surrogate and imputer on correctly scoped training data, then revalidate the
whole-clip use as well before replacing the shared model export.

## Selecting a configuration offline

Start with 1s/0.5s, 2s/1s, 3s/1.5s and 4s/2s. Their counts over exactly six seconds
are 11, 5, 3 and 2 windows. The same recordings, coverage, thread settings, model
identity, normalization and threshold methodology must be used for each.

- Run warmups, then repeated isolated timings. Measure full XAI latency plus
  per-stage and per-window median/p95, failed windows, imputation counts and
  valid coverage. Do not use contended concurrent runs to claim latency gains.
- Obtain the teacher's prediction for each actual source window using its
  declared preprocessing. Compare surrogate predictions with those targets
  using MAE, RMSE and R². Do not copy a whole-clip target onto every window.
- Split recordings/speakers before generating overlapping windows, and fit
  the imputer only on the training split. Select settings/thresholds on the
  validation split; reserve the test/eval split for final evaluation.
- Measure attention/activity IoU as descriptive agreement. For genuine spoof
  localization, also use independently verified segment annotations and a
  class-specific localization objective; absolute SHAP activity can legitimately
  be large for bonafide-supporting features too. Record annotation resolution
  and inspect bonafide false highlights and boundary error.
- Select a configuration that meets predeclared latency, coverage and fidelity
  requirements at the necessary temporal resolution. A maximum-IoU configuration
  alone is not an evidence-based winner. A single available window supplies no
  within-window localization even if its runtime is fast.

The supplied Drive label folder includes train/dev/eval `.npy` files at multiple
resolutions. A bounded dev/0.64 file was inspected using NumPy headers and pickle
opcodes without executing the object payload. It yielded 24,844 label vectors,
including 12,396 `CON_D_...` vectors containing both label values at this coarse
resolution. These counts describe the inspected vectors, not verified audio
availability or localization performance.

The [supplied dataset root](https://drive.google.com/drive/folders/1aHsAL7-3bi3W7trcyhzm4B3YF_a7rpa6)
contains `database/dev/con_wav`, which is the
[previously supplied audio folder](https://drive.google.com/drive/folders/11J-nra8jz43L7MZXhJkL9GphZmWGEPd6).
Its first 1,000 listed files were `LA_D_...` recordings; the initially inspected
100 matched bonafide label vectors. The connector's folder listing has a
1,000-item page limit without a continuation parameter, and search returned no
results even for a known accessible recording. Therefore this inspection does
not establish that `CON_D_...` audio is absent from the folder. The root also
contains `_archives/database_dev.tar.gz` (2,027,972,850 bytes), whose contents
were not downloaded or verified during this bounded inspection. Matching mixed
audio to its labels and running a representative held-out localization study
remain pending; no best window configuration has been established.

This offline experiment is separate from the production runtime. No legacy
extractor/comparison tool is reintroduced. Tests exercise all four window
configurations, offset preservation, gaps, tails, failures, deadlines, model
contracts, SHAP additivity and common-coverage IoU.
