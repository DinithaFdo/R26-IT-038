"""Export the backend feature implementation as a portable training package.

Copy the canonical extraction code, feature schema, and compatibility shims into
a small ZIP so offline feature generation uses the same functions as inference.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

SERVICE_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_ROOT = SERVICE_ROOT / "app/voice_xai/semantic"


def export_extractor(destination: Path):
    files = {}
    for name in ["schema.py", "recipe.py", "runtime.py", "dataset.py"]:
        files[name] = (SEMANTIC_ROOT / name).read_text(encoding="utf-8")
    files["runtime.py"] = files["runtime.py"].replace(
        "from app.models.real.glottal_compat import (", "from .glottal_compat import ("
    )
    files["glottal_compat.py"] = (SERVICE_ROOT / "app/models/real/glottal_compat.py").read_text(encoding="utf-8")
    files["__init__.py"] = '''"""Portable full-clip feature extraction and dataset rebuilding API."""
from .schema import FEATURE_NAMES, SELECTED20, FEATURE_EXTRACTION_VERSION, FEATURE_SCHEMA_VERSION, PREPROCESSING_CONTRACT
from .recipe import extract_surrogate_165_features, load_audio_path, prepare_audio
from .dataset import rebuild_feature_frame
'''
    # Import the lightweight schema only; exporting must not initialize DisVoice.
    namespace = {}
    exec(compile(files["schema.py"], "schema.py", "exec"), namespace)
    packages = ["numpy", "scipy", "librosa", "soundfile", "praat-parselmouth",
                "pysptk", "resampy", "pandas", "setuptools", "matplotlib",
                "tqdm", "kaldi_io"]
    environment = {name: importlib.metadata.version(name) for name in packages}
    # DisVoice imports torch for optional output formats. Keep Colab's existing
    # compatible CPU/CUDA build instead of pinning a Windows-specific wheel tag.
    files["requirements.txt"] = "\n".join(f"{name}=={version}" for name, version in environment.items()) + "\ntorch>=2.6,<3\n"
    manifest = {
        "extractor_version": namespace["FEATURE_EXTRACTION_VERSION"],
        "feature_schema_version": namespace["FEATURE_SCHEMA_VERSION"],
        "preprocessing": namespace["PREPROCESSING_CONTRACT"],
        "source_hashes": {name: sha256(source.encode()).hexdigest() for name, source in files.items()},
        "extraction_environment": environment,
        "disvoice": importlib.metadata.version("disvoice"),
        "torch": importlib.metadata.version("torch"),
    }
    files["export_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.writestr(f"semantic_audio/{name}", source)
    return destination


def main():
    parser = argparse.ArgumentParser(description="Export the shared semantic feature extractor for offline training.")
    parser.add_argument("--output", type=Path, default=SERVICE_ROOT / "docs/training/semantic_audio_v2.zip")
    args = parser.parse_args()
    print(export_extractor(args.output))


if __name__ == "__main__":
    main()
