"""Load and cache the DisVoice Glottal extractor with compatibility patches.

Import the installed Glottal module directly to avoid initializing unrelated
TensorFlow extractors from the package entry point.
"""

from functools import lru_cache
import importlib.util
from pathlib import Path

from app.models.real.glottal_compat import (
    apply_disvoice_compatibility_patches,
    patch_disvoice_gci_numpy2_bug,
)


@lru_cache(maxsize=1)
def load_glottal_class():
    apply_disvoice_compatibility_patches()
    package = importlib.util.find_spec("disvoice")
    if package is None or package.origin is None:
        raise ImportError("Install the xai extra for DisVoice feature extraction.")
    # Package __init__ imports unrelated TensorFlow extractors. Execute the
    # real installed Glottal module directly; its calculations are unchanged.
    source = Path(package.origin).parent / "glottal/glottal.py"
    spec = importlib.util.spec_from_file_location("_semantic_disvoice_glottal", source)
    if spec is None or spec.loader is None:
        raise ImportError("DisVoice Glottal implementation is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patch_disvoice_gci_numpy2_bug()
    return module.Glottal
