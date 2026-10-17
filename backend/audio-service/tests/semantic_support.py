"""Build small, genuinely fitted model exports for full-clip inference tests."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import xgboost as xgb

from app.voice_xai.semantic.schema import (
    FEATURE_NAMES, FEATURE_EXTRACTION_VERSION, FEATURE_SCHEMA_VERSION,
    PREPROCESSING_CONTRACT,
)


def write_model_export(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(103)
    inputs = pd.DataFrame(rng.normal(size=(48, 165)), columns=FEATURE_NAMES)
    inputs.iloc[0, :2] = np.nan
    target = np.clip(0.5 + 0.2 * inputs.iloc[:, 2].to_numpy(), 0, 1)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(inputs.iloc[:36])
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2, n_jobs=1, objective="reg:squarederror")
    model.fit(pd.DataFrame(train, columns=FEATURE_NAMES), target[:36])
    test = imputer.transform(inputs.iloc[36:])
    fidelity = r2_score(target[36:], model.predict(test))
    model.save_model(directory / "surrogate_xgboost.json")
    joblib.dump(imputer, directory / "imputer.joblib")
    (directory / "feature_columns.json").write_text(json.dumps(FEATURE_NAMES), encoding="utf-8")
    metadata = {
        "experiment_name": "fullclip-test-v2", "model_type": "XGBRegressor",
        "target": "teacher_fused_spoof_probability",
        "ground_truth_used_as_training_target": False,
        "feature_count": 165, "decision_threshold": 0.5,
        "extractor_version": FEATURE_EXTRACTION_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "preprocessing": PREPROCESSING_CONTRACT,
        "test_rows": 12, "primary_test_fidelity": {"r2": fidelity, "n": 12},
    }
    (directory / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return inputs, model, imputer
