"""Private, compact serialization for Phase 1 intermediate representations."""

from __future__ import annotations

from io import BytesIO
import json
from typing import Any

import numpy as np

from app.voice_xai.capture.contracts import ExtractionBundle


EXTRACTION_ARTIFACT_FORMAT = "voice-xai-extraction-npz-v1"
EXTRACTION_ARTIFACT_CONTENT_TYPE = "application/x-npz"


def serialize_extraction_bundle(bundle: ExtractionBundle) -> bytes:
    """Encode captures as a compressed NPZ plus a JSON manifest.

    Raw tensor values are deliberately binary rather than JSON: this avoids
    expanding a bounded multi-megabyte capture into a much larger response or
    database document. The caller stores these bytes only in the private XAI
    artifact store.
    """

    arrays: dict[str, np.ndarray[Any, Any]] = {}
    branches: dict[str, dict[str, dict[str, Any]]] = {}
    for branch_name, layers in sorted(bundle.branches.items()):
        serialized_layers: dict[str, dict[str, Any]] = {}
        for layer_index, capture in sorted(layers.items()):
            descriptor = {
                "category": capture.category,
                "normalization": capture.normalization,
                "status": capture.status,
                "reason": capture.reason,
                "array_key": None,
            }
            if capture.values is not None:
                array_key = f"capture_{len(arrays)}"
                arrays[array_key] = np.ascontiguousarray(
                    capture.values, dtype=np.float32
                )
                descriptor["array_key"] = array_key
            serialized_layers[str(layer_index)] = descriptor
        branches[branch_name] = serialized_layers

    manifest = {
        "format": EXTRACTION_ARTIFACT_FORMAT,
        "request_id": bundle.request_id,
        "metadata": bundle.metadata,
        "branches": branches,
        "acoustic_features": (
            bundle.acoustic_features.as_json_dict()
            if bundle.acoustic_features is not None
            else None
        ),
    }
    output = BytesIO()
    np.savez_compressed(
        output,
        manifest=np.asarray(json.dumps(manifest, sort_keys=True), dtype=np.str_),
        **arrays,
    )
    return output.getvalue()
