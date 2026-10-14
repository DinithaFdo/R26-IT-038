# Semantic explanation pipeline

The semantic service uses a 165-feature XGBoost regressor to approximate the
teacher ensemble's spoof probability. Tree SHAP explains the regressor's raw
output through signed acoustic feature contributions for the whole clip.

## Replace a trained model

Export these files together from the **same finalized training run**:

- `surrogate_xgboost.json`
- `feature_columns.json` (the exact 165-feature input order)
- `imputer.joblib`
- `training_metadata.json`

The current extractor is `shared-clean-fullclip-audio-v2`, with schema
`shared-clean-fullclip-165-v2`. Training metadata must declare these versions
and the exact `preprocessing` object from `schema.py`. See
[the retraining steps](semantic-retraining.md) for dataset rebuilding and export.
The former crop-based model is incompatible; changing its metadata does not
make it a full-clip model. The existing v1 model bytes are preserved in the fixed
model directory.
With real XAI enabled, startup rejects that model until retrained v2 exports
replace it. Classification can run during retraining with `XAI_ENABLED=false`.

The loader reads only these four files. Only place a trusted imputer export in
the configured directory. The selected feature names are defined in `schema.py`.

The default model directory is:

```text
audio-service/app/voice_xai/semantic/artifacts/
  surrogate_xgboost.json
  feature_columns.json
  imputer.joblib
  training_metadata.json
```

Set `XAI_SEMANTIC_MODEL_DIR` only if you use another directory. Relative paths
resolve from `audio-service`.

To update the model, stop the backend, replace the four required files together
from the same finalized training run, and restart.
A downloaded imputer named `imputer (1).joblib` must be renamed to `imputer.joblib`.

Startup checks the feature order, preprocessing contract, regression objective,
fitted imputer, and prediction/SHAP reconstruction.
Invalid or incomplete exports reject startup. Model identity is calculated from
file contents automatically; no import command, version folder, manifest, or
active pointer is required. File bytes are preserved in Git.

The running process retains its loaded model until restart. If model files change
while it is running, explanations request a restart instead of mixing loaded
weights with newly replaced metadata. Keep backups outside this model directory
if you need to restore an earlier export.

## Feature extraction and inference

Original bounded upload bytes are retained privately in memory across upload
cleanup and manual replay. Decode the audio once with SoundFile, average stereo
channels, resample to 16 kHz with `kaiser_fast` when needed, replace non-finite
samples, remove DC offset, trim boundary silence at 35 dB, and peak-normalize.
All 148 acoustic inputs and 20 selected Glottal measurements use the same cleaned
float32 waveform for its full remaining duration. No centre crop, padding, or
CPP frame cap is applied. DisVoice's file interface receives a PCM16 encoding
of that waveform without another source decode or another preprocessing pass.
The three exact-name overlaps (`f0_std`, `spectral_centroid_mean`, and
`spectral_centroid_std`) retain the base acoustic value, producing 165 inputs.

Boundary trimming does not remove interior pauses. The cleaned signal must
contain at least 0.2 seconds and have a peak above `1e-8`. Shared decoding avoids
repeated preparation, but full-duration feature calculations can cost more than
the former six-second crop; this change does not promise lower total latency.

Inference applies the fitted median imputer, runs square-error regression, and
uses path-dependent Tree SHAP on the raw score. The reported probability is
`clip(raw_score, 0, 1)`. No sigmoid or probability calibrator is applied to this
inference output. All 165 signed contributions
are returned; positive, negative and zero values have distinct directions.
The teacher prediction remains separate from the surrogate approximation.

R² is read from the **current model's** `primary_test_fidelity.r2` metadata,
with its sample count. Missing metrics remain unavailable and never block
inference. Fidelity metadata must describe the active model's evaluation run.
Clip-level SHAP has no temporal intervals, so temporal/semantic IoU is unavailable.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `schema.py` | Define feature order, selected Glottal names, and extraction/schema versions. |
| `recipe.py` | Compute acoustic and Glottal measurements from source audio. |
| `runtime.py` | Load the DisVoice Glottal extractor with runtime compatibility patches. |
| `features.py` | Merge measurements into the ordered 165-feature input vector. |
| `dataset.py` | Rebuild all feature columns while preserving teacher targets and sample metadata. |
| `artifact_loader.py` | Validate fixed-directory model exports and load the imputer, regressor, and Tree SHAP explainer. |
| `contracts.py` | Validate prediction evidence and carry analysis results to the workflow. |
| `service.py` | Coordinate extraction, imputation, regression, and SHAP additivity checks. |
| `feature_labels.py` | Format supported feature identifiers for display. |
| `explainer.py` | Build ranked contributions, clip summaries, and fidelity metadata for the API. |
| `__init__.py` | Expose the package's feature contract, extractor, and service. |

The package supports startup model loading, audio extraction/explanation, and
offline training-data rebuilding. Portable DataFrame/dataset helpers are required
for rebuilding training inputs even though they are not called for each API
request. Dataclass validation hooks run when evidence is constructed.

## Verification and scope

From `audio-service`, run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_semantic_pipeline.py tests/test_semantic_model_updates.py tests/test_semantic_training_export.py
```

Tests verify full-duration processing beyond six seconds, removal of the CPP
frame cap, one source decode/preparation, and waveform identity across extractors.
They compare exported training features with backend features, validate real
XGBoost prediction/SHAP reconstruction using a small fitted test model, exercise
invalid model replacements, and verify preservation of sample targets and original bytes.
These tests do not establish the accuracy or fidelity of a retrained production
model. Extraction parity across machines also depends on matching dependency
versions; the portable export records the backend's extraction environment.

The `voice-xai-api-v2` response schema has no independent-classifier or window modes.
Persisted explanations made with the removed schema should be regenerated;
this change does not delete stored predictions or explanations.

