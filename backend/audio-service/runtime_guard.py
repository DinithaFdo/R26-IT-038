import sys


MINIMUM_PYTHON = (3, 12)
MAXIMUM_PYTHON_EXCLUSIVE = (3, 13)


def ensure_supported_python(version_info=None) -> None:
    """Stop startup cleanly when the interpreter is outside the supported range."""

    detected = tuple(version_info or sys.version_info)
    detected_release = detected[:2]
    if MINIMUM_PYTHON <= detected_release < MAXIMUM_PYTHON_EXCLUSIVE:
        return

    detected_version = ".".join(str(part) for part in detected[:3])
    raise SystemExit(
        "MULTI-SCOPE backend requires Python 3.12 (>=3.12,<3.13).\n"
        f"Detected Python {detected_version}.\n"
        "Recreate backend/audio-service/.venv using Python 3.12."
    )
