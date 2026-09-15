"""Validated immutable calibration for XLS-R temporal rollout density."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class TemporalCalibrationError(ValueError):
    """Raised when a saved threshold cannot be safely applied."""


REQUIRED_STATUS = "CALIBRATED_ON_PARTIALSPOOF_V1_2_DEV"


def load_fixed_density_threshold(
    path: Path, *, expected: dict[str, float]
) -> tuple[float, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemporalCalibrationError(
            "The temporal threshold artifact cannot be read."
        ) from error
    if not isinstance(raw, dict) or raw.get("mode") != "global_density":
        raise TemporalCalibrationError(
            "Temporal calibration must use global_density mode."
        )
    if raw.get("status") != REQUIRED_STATUS:
        raise TemporalCalibrationError(
            "Temporal calibration is not the locked PartialSpoof v1.2 DEV artifact."
        )
    try:
        threshold = float(raw["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise TemporalCalibrationError(
            "Temporal calibration has no finite threshold value."
        ) from error
    if not math.isfinite(threshold):
        raise TemporalCalibrationError("Temporal calibration has no finite threshold value.")
    pipeline = raw.get("pipeline")
    if not isinstance(pipeline, dict):
        raise TemporalCalibrationError("Temporal calibration is missing pipeline metadata.")
    for name, current in expected.items():
        calibrated = pipeline.get(name)
        if calibrated is None or not math.isclose(
            float(calibrated), current, rel_tol=0.0, abs_tol=1e-9
        ):
            raise TemporalCalibrationError(
                f"Temporal calibration does not match the configured {name}."
            )
    return threshold, raw
