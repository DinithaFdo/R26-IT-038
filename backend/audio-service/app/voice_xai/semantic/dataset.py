"""Rebuild acoustic input columns while preserving sample and teacher metadata.

Re-extract every row from its original audio path using the current full-clip
feature contract. Stop on extraction failures instead of mixing old and new rows.
"""
from pathlib import Path

import pandas as pd

from .recipe import extract_surrogate_165_features
from .schema import FEATURE_NAMES


def rebuild_feature_frame(frame, *, audio_path_column="audio_path", progress=None):
    """Return a new table with all feature columns replaced in original row order."""
    if audio_path_column not in frame.columns:
        raise ValueError(f"Missing source audio column: {audio_path_column}.")
    paths = []
    for position, value in enumerate(frame[audio_path_column]):
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError(f"Invalid source audio path at row {position}.")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Original audio is unavailable at row {position}: {path}")
        paths.append(path)
    rows = []
    for position, path in enumerate(paths):
        try:
            rows.append(extract_surrogate_165_features(path))
        except Exception as error:
            raise ValueError(f"Feature extraction failed at row {position}: {path}") from error
        if progress is not None:
            progress(position + 1, len(paths))
    if rows:
        values = pd.concat(rows, ignore_index=True)
    else:
        values = pd.DataFrame(columns=FEATURE_NAMES, dtype=float)
    values.index = frame.index
    metadata = frame.drop(columns=[name for name in FEATURE_NAMES if name in frame.columns])
    result = pd.concat([metadata.copy(deep=True), values], axis=1)
    columns = list(frame.columns) + [name for name in FEATURE_NAMES if name not in frame.columns]
    return result.loc[:, columns]
