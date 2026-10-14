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

The optional `probability_calibrator.joblib`, `teacher_fusion.json`,
`teacher_assets_<fingerprint>.json`, and `selected_glottal_features_20.json`
are preserved and validated when supplied. Only import your trusted joblib files.

From `audio-service`, run:

```powershell
.\.venv\Scripts\python.exe -m app.voice_xai.semantic.import_bundle --source "D:\Models\new-surrogate"
```

Then restart the backend. For a model trained with the same feature contract,
no source, feature labels, hashes, model version, threshold, R² value, or settings
need editing.
`XAI_SEMANTIC_BUNDLE_DIR` selects the deployment's bundle directory; the command
uses that setting automatically. `--destination PATH` can prepare another deployment.
`--imputer "imputer (1).joblib"` accepts a differently named downloaded imputer.

Each import validates the input contract, objective, fitted median imputer,
teacher metadata when supplied, and a real prediction/SHAP reconstruction before
activation. An invalid import leaves `active.json` untouched. All file hashes
contribute to a unique version, so a retrained model cannot inherit an earlier
model's provenance or quality cache. Previous valid versions remain in `bundles/`.
The running service requires restart after activation to avoid mixing versions.
Bundle files retain their exact bytes in Git, including JSON line endings.

## Feature extraction and inference

Original bounded upload bytes are retained privately in memory across upload
cleanup and manual replay. The base acoustic path centre-crops to six seconds;
the selected-20 Glottal path uses the full original audio and its separate trim,
normalisation and DisVoice PCM16 path. The three exact-name overlaps retain the
base acoustic value, producing 165 features rather than 168.

Inference applies the fitted median imputer, runs square-error regression, and
uses path-dependent Tree SHAP on the raw score. The reported probability is
`clip(raw_score, 0, 1)`. No sigmoid or probability calibrator is applied to this
inference output. All 165 signed contributions
are returned; positive, negative and zero values have distinct directions.
The teacher prediction remains separate from the surrogate approximation.

R² is read from the **current bundle's** `primary_test_fidelity.r2` metadata,
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
| `artifact_loader.py` | Validate bundle integrity and load the imputer, regressor, and Tree SHAP explainer. |
| `import_bundle.py` | Validate model exports and atomically activate an immutable version. |
| `contracts.py` | Validate prediction evidence and carry analysis results to the workflow. |
| `service.py` | Coordinate extraction, imputation, regression, and SHAP additivity checks. |
| `feature_labels.py` | Format supported feature identifiers for display. |
| `explainer.py` | Build ranked contributions, clip summaries, and fidelity metadata for the API. |
| `__init__.py` | Expose the package's feature contract, extractor, and service. |

## Verification and scope

From `audio-service`, run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_semantic_pipeline.py tests/test_semantic_bundle_updates.py
```

Regression tests compare saved feature vectors, predictions, and all 165 SHAP
values against reference exports. They also exercise invalid imports, activate
retrained model versions with updated fidelity metadata, and verify that original
source bytes survive upload cleanup.

Saved feature vectors reproduce their reference predictions and SHAP values.
This does not establish bitwise waveform-level equivalence across operating
systems, dependency versions, or unverified copies of source audio. DisVoice
measurements can differ across those environments.

The `voice-xai-api-v2` response schema has no independent-classifier or window modes.
Persisted explanations made with the removed schema should be regenerated;
this change does not delete stored predictions or explanations.

