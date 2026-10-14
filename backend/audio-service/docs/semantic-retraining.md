# Retrain the surrogate with shared full-clip features

The new flow is:

`original audio -> mono / 16 kHz -> numeric cleanup -> DC removal -> boundary silence trim (35 dB) -> peak normalization -> full cleaned waveform -> all feature groups`

There is no six-second crop. All CPP frames are evaluated. The 148 base acoustic
features and selected 20 Glottal measurements use the same waveform. Only the
DisVoice file interface receives a PCM16 encoding of it. The three shared names
retain their base values, so the ordered input still contains 165 columns.

The existing fitted model/imputer and old parquet feature values are incompatible
with this preprocessing. Rebuilding only the final prediction helper does not
change what the model trains on. Do not relabel an existing model's metadata.

## 1. Use the backend extraction implementation

The prepared archive is `training/semantic_audio_v2.zip` beside this guide. To
regenerate it after any extraction-code change, run from `audio-service`:

```powershell
.\.venv\Scripts\python.exe scripts/export_semantic_extractor.py
```

Upload that ZIP to Colab as `/content/semantic_audio_v2.zip`. Add this setup cell
after mounting Drive and before any feature-table loading or extraction:

```python
from pathlib import Path
from zipfile import ZipFile
import sys

with ZipFile('/content/semantic_audio_v2.zip') as archive:
    archive.extractall('/content/semantic_extractor_v2')
sys.path.insert(0, '/content/semantic_extractor_v2')
```

The package includes the backend's actual extraction sources and compatibility
patches. `semantic_audio/export_manifest.json` records their hashes and installed
extraction-library versions. Use the packaged requirements for extraction, and
install DisVoice without its old transitive dependency pins:

```python
%pip install -q -r /content/semantic_extractor_v2/semantic_audio/requirements.txt
%pip install -q --no-deps disvoice==0.1.10
```

Restart the Colab runtime after dependency changes, rerun the ZIP setup cell,
and then import:

```python
import json
import numpy as np
import pandas as pd
from semantic_audio import (
    FEATURE_NAMES, SELECTED20, FEATURE_EXTRACTION_VERSION,
    FEATURE_SCHEMA_VERSION, PREPROCESSING_CONTRACT,
    extract_surrogate_165_features, rebuild_feature_frame,
)
assert len(FEATURE_NAMES) == 165
assert FEATURE_EXTRACTION_VERSION == 'shared-clean-fullclip-audio-v2'
```

Keep the training model's existing square-error XGBRegressor setup. The
extraction requirements do not replace the training/SHAP stack installation.

## 2. Rebuild the input tables before fitting anything

In `02_1_Train_Evaluate_Surrogate_XGBoost_v2_selected20_aligned.ipynb`, the
training section reads existing parquet tables in the cell beginning
`train_df = pd.read_parquet(TRAIN_PATH)`. Its extraction definitions appear
much later. Rebuild all three tables before running that read cell or any
imputer, hyperparameter search, calibration, or model-fitting cell.

Run the following once after the imports above. This writes a separate dataset
and leaves existing data intact. Every `audio_path` must resolve to the original
audio in Colab. If a row only points to an archive member, extract that audio
and update its `audio_path` first. Missing audio or failed extraction stops the
rebuild; old feature values are never used as a fallback.

```python
ROOT = Path('/content/drive/MyDrive/Research/Surrogate-v1/Dataset_Creation')
SOURCE_DATASET_ROOT = ROOT / 'datasets_asv2019_selected20_aligned'
NEW_DATASET_ROOT = ROOT / 'datasets_asv2019_shared_fullclip_v2'
NEW_META_ROOT = ROOT / 'metadata_asv2019_shared_fullclip_v2'
NEW_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
NEW_META_ROOT.mkdir(parents=True, exist_ok=True)

def progress(done, total):
    if done % 100 == 0 or done == total:
        print(f'{done}/{total} audio clips extracted')

for split in ['train', 'validation', 'test']:
    filename = f'surrogate_{split}.parquet'
    original = pd.read_parquet(SOURCE_DATASET_ROOT / filename)
    rebuilt = rebuild_feature_frame(original, progress=progress)
    assert len(rebuilt) == len(original)
    for column in [
        'sample_key', 'teacher_fused_spoof_probability',
        'label_int', 'teacher_fingerprint',
    ]:
        pd.testing.assert_series_equal(rebuilt[column], original[column])
    rebuilt.to_parquet(NEW_DATASET_ROOT / filename, index=False)

(NEW_META_ROOT / 'feature_columns_165.json').write_text(
    json.dumps(list(FEATURE_NAMES), indent=2), encoding='utf-8'
)
(NEW_META_ROOT / 'preprocessing_contract.json').write_text(
    json.dumps({
        'extractor_version': FEATURE_EXTRACTION_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'preprocessing': PREPROCESSING_CONTRACT,
    }, indent=2), encoding='utf-8'
)
```

Teacher predictions can remain unchanged when they came from the same audio
and unchanged teacher models/preprocessing/fusion. This change affects surrogate
inputs, not teacher inference. Keep the existing official splits and sample
identities; continue auditing cross-split overlap. If the teacher or source
audio changes, regenerate the corresponding teacher predictions as well.

## 3. Update both configuration cells in the training file

There are two path-configuration cells: one starts with `from pathlib import Path`
and defines `EXPERIMENT_NAME`; the following one repeats `RESEARCH_ROOT`,
`DATASET_ROOT`, and `TRAINING_ROOT`. In the provided file they are zero-based
cells 7 and 8. Update both, or keep one configuration cell and remove the
duplicate, so the second cannot reset the paths to the old dataset.

Use:

```python
DATASET_ROOT = SURROGATE_ROOT / 'Dataset_Creation' / 'datasets_asv2019_shared_fullclip_v2'
DATASET_META_ROOT = SURROGATE_ROOT / 'Dataset_Creation' / 'metadata_asv2019_shared_fullclip_v2'
EXPERIMENT_NAME = 'asv2019_shared_fullclip_reg50_v2'
TRAINING_ROOT = SURROGATE_ROOT / 'Training_experiments' / EXPERIMENT_NAME
```

Derive `MODEL_DIR`, `EVAL_DIR`, `PLOTS_DIR`, `TRAIN_PATH`, `VALID_PATH`, and
`TEST_PATH` from those updated roots as before. Retain the completed-experiment
overwrite check. In the feature-contract validation cell, also assert:

```python
assert tuple(feature_columns) == FEATURE_NAMES
```

Rerun from the new parquet-loading cell through final export. Fit fresh median
imputers on the designated training partitions; do not reuse the old imputer or
model. Keep `teacher_fused_spoof_probability` as the regression target and keep
`label_int` for diagnostics only. Rerun tuning, calibration, fidelity evaluation,
and test predictions with the rebuilt features. Leave the test set out of model
selection. A higher R² must be measured; this preprocessing change does not
guarantee a particular value.

## 4. Export the actual extraction contract with the new model

In the cell beginning `MODEL_PATH = MODEL_DIR / "surrogate_xgboost.json"`
(zero-based cell 54), add these entries to `training_metadata` before its
`TRAINING_METADATA_PATH.write_text(...)` call:

```python
training_metadata.update({
    'extractor_version': FEATURE_EXTRACTION_VERSION,
    'feature_schema_version': FEATURE_SCHEMA_VERSION,
    'preprocessing': dict(PREPROCESSING_CONTRACT),
})
```

Export a newly fitted `surrogate_xgboost.json`, `imputer.joblib`,
`feature_columns.json`, and `training_metadata.json` together. The existing
export cell already writes these. Its fidelity metadata must come from the new
evaluation, not an old result or an R² target. Keep the matching teacher-fusion
and teacher-assets files if you export them as optional provenance companions.

## 5. Replace the late raw-audio helper definitions

In the training file, remove the old extraction setup and function definitions
in zero-based cells 59, 61, and 63. They contain `MAX_AUDIO_SECONDS = 6.0`,
`center_crop`, separate Glottal loading, and the old DisVoice source decode.
Use the `semantic_audio` imports from step 1 instead.

In cell 65, remove only the locally defined `extract_surrogate_165_features`;
keep `load_saved_surrogate` and `predict_audio`. The imported function has the
same DataFrame signature, including `expected_feature_columns`, so their calls
remain valid. Do not execute older definitions afterward: they would override
the shared imported function. Copying only this late helper change, without
step 2, still trains on the old inputs.

## 6. Update the SHAP analysis file to the same extractor

For `03_SHAP_Analysis_10_Audio_Clips.ipynb`:

1. Set `EXPERIMENT_NAME = 'asv2019_shared_fullclip_reg50_v2'` in its path cell
   (zero-based cell 9), so both model inputs and SHAP outputs use the new run.
2. Run the ZIP/import setup from step 1 before loading/extracting audio.
3. Remove the old extraction setup/definitions in zero-based cells 16, 18,
   and 20. Remove the local `extract_surrogate_165_features` definition in
   cell 22, retaining its other helpers. Use the imported function everywhere.
4. After loading `training_metadata` and `feature_columns` in cell 11, check:

```python
assert tuple(feature_columns) == FEATURE_NAMES
assert training_metadata['extractor_version'] == FEATURE_EXTRACTION_VERSION
assert training_metadata['feature_schema_version'] == FEATURE_SCHEMA_VERSION
assert training_metadata['preprocessing'] == PREPROCESSING_CONTRACT
DEFAULT_THRESHOLD = float(training_metadata['decision_threshold'])
```

5. Rerun audio selection, raw-feature extraction, imputation, predictions, SHAP,
   and all exports. Do not reuse the previous raw/imputed feature CSVs or SHAP
   output directory. Keep Tree SHAP in raw regression space and keep the existing
   `np.clip(raw_score, 0, 1)` reporting transform.
6. Add the same three contract fields from step 4 to `run_metadata` before it
   is written in the analysis metadata cell (zero-based cell 46).

## 7. Import the retrained bundle into the backend

Copy the complete new model export folder to your local machine, then run from
`audio-service`:

```powershell
.\.venv\Scripts\python.exe -m app.voice_xai.semantic.import_bundle --source "D:\Models\asv2019_shared_fullclip_reg50_v2\model"
```

Restart the backend. The importer validates the new contract, model, fitted
imputer, and prediction/SHAP reconstruction before switching `active.json`.
It rejects unversioned or old preprocessing metadata. Existing immutable model
files are retained unchanged, and an invalid import does not replace the active
reference. With real XAI enabled, the old v1 bundle cannot start the new service.

For final parity verification, use the same original audio on both machines and
compare all 165 raw and imputed values, raw prediction, clipped probability, and
all SHAP contributions. Use the environment recorded in the portable export to
investigate dependency-specific differences. The old reference exports do not
validate the new extraction contract.
