import pytest

from runtime_guard import ensure_supported_python


@pytest.mark.parametrize(
    "version_info",
    [
        (3, 12, 0),
        (3, 12, 10),
        (3, 12, 99),
    ],
)
def test_supported_python_versions_pass(version_info) -> None:
    ensure_supported_python(version_info)


def test_python_39_exits_with_recreation_instructions() -> None:
    with pytest.raises(SystemExit) as exc_info:
        ensure_supported_python((3, 9, 6))

    assert str(exc_info.value) == (
        "MULTI-SCOPE backend requires Python 3.12 (>=3.12,<3.13).\n"
        "Detected Python 3.9.6.\n"
        "Recreate backend/audio-service/.venv using Python 3.12."
    )


def test_python_313_is_rejected_to_match_project_metadata() -> None:
    with pytest.raises(SystemExit, match="Detected Python 3.13.0"):
        ensure_supported_python((3, 13, 0))


@pytest.mark.parametrize("version_info", [(3, 10, 0), (3, 11, 0), (3, 11, 99), (4, 0, 0)])
def test_versions_outside_python_312_are_rejected(version_info) -> None:
    with pytest.raises(SystemExit, match="requires Python 3.12"):
        ensure_supported_python(version_info)
