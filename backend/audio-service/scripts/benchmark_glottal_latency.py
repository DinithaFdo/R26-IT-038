"""End-to-end latency benchmark for the real Glottal branch.

MEASUREMENT ONLY. This script does not modify, patch, or alter the behavior
of any production code: `app/models/preprocessing/glottal_v1.py` and
`app/models/real/glottal_inference.py` are called exactly as production
calls them. The only "monkeypatching" here wraps `soundfile.write` and the
DisVoice extractor's `extract_features_file` with pass-through timing
wrappers (call the original, measure, return the original's result
unchanged) purely to split the "temp WAV creation" and "DisVoice/IAIF
extraction" sub-times out of one function call -- the computation and its
result are identical to an uninstrumented run.

Usage:
    python scripts/benchmark_glottal_latency.py
    python scripts/benchmark_glottal_latency.py --audio /path/to/file.flac
    python scripts/benchmark_glottal_latency.py --skip-full-pipeline
    python scripts/benchmark_glottal_latency.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import subprocess
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings(
    "ignore",
    message='Field "model_.*" in Settings has conflict with protected namespace',
)

from app.config.settings import Settings
from app.ingestion.audio import inspect_audio_file, preprocess_audio_file
from app.models.factory import ModelFactory
from app.models.preprocessing import glottal_v1
from app.models.real.glottal_inference import LoadedGlottalBranch, build_glottal_loader

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent

WARMUP_RUNS = 2
MEASURED_RUNS = 10
PREFERRED_AUDIO_FILENAME = "LA_D_6330573.flac"
STAGE_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
STAGE_LABELS = {
    "A": "Input decode / shared audio prep",
    "B": "Glottal waveform cleaning (DC + trim + normalize)",
    "C": "Voice-quality extraction (F0/jitter/shimmer/HNR)",
    "D": "Spectral extraction (tilt + centroid)",
    "E": "Temporary WAV creation",
    "F": "DisVoice / IAIF extraction",
    "G": "Mapping 36 DisVoice features (+ tempfile mgmt remainder)",
    "H": "Selected-20 feature assembly + validation",
    "I": "sklearn predict_proba",
    "J": "Total Glottal branch latency (real end-to-end predictor call)",
}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class StageStats:
    samples_ms: list[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        self.samples_ms.append(value_ms)

    @property
    def n(self) -> int:
        return len(self.samples_ms)

    def summary(self) -> dict[str, float]:
        if not self.samples_ms:
            return {"min": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "stdev": 0.0}
        values = sorted(self.samples_ms)
        return {
            "min": values[0],
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": _percentile(values, 95),
            "max": values[-1],
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = k - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


# ---------------------------------------------------------------------------
# Environment report
# ---------------------------------------------------------------------------


def environment_report() -> dict[str, Any]:
    cpu_model = None
    physical_cores = None
    ram_bytes = None
    if platform.system() == "Darwin":
        cpu_model = _sysctl("machdep.cpu.brand_string")
        physical_cores = _sysctl("hw.physicalcpu")
        ram_bytes = _sysctl("hw.memsize")
    elif platform.system() == "Linux":
        cpu_model = _linux_cpu_model()
        physical_cores = _linux_physical_cores()
        ram_bytes = _linux_ram_bytes()

    gpu_available = False
    gpu_backend = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_backend = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            gpu_available = True
            gpu_backend = "mps"
    except Exception as error:  # noqa: BLE001 - environment probing must not crash the benchmark
        logging.getLogger(__name__).debug("GPU probe failed: %s", error)

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpu_count": _try_int(subprocess_or_os_cpu_count()),
        "physical_cpu_count": _try_int(physical_cores),
        "ram_gb": round(int(ram_bytes) / (1024**3), 2) if ram_bytes else None,
        "gpu_available": gpu_available,
        "gpu_backend": gpu_backend,
        "glottal_uses_gpu": False,
        "glottal_compute_note": (
            "DisVoice/parselmouth/scikit-learn are CPU-only in this pipeline; "
            "the Glottal branch never touches CUDA/MPS regardless of GPU availability."
        ),
    }


def subprocess_or_os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=2, check=False
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _linux_cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def _linux_physical_cores() -> str | None:
    try:
        result = subprocess.run(
            ["nproc", "--all"], capture_output=True, text=True, timeout=2, check=False
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _linux_ram_bytes() -> str | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return str(kb * 1024)
    except Exception:  # noqa: BLE001
        return None
    return None


def _try_int(value: str | int | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Audio resolution
# ---------------------------------------------------------------------------


def resolve_audio_path(explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"--audio path does not exist: {explicit}")
        return explicit, "explicit --audio argument"

    # Deliberately bounded to the repository itself -- NOT Path.home(): this
    # machine has a OneDrive-synced home directory, and an unbounded rglob()
    # there can hang for minutes (or hit a network-mount timeout, as observed)
    # crawling cloud-synced folders. The repo is the only place a real
    # research audio sample would legitimately be checked in anyway.
    search_roots = [_REPO_ROOT, _BACKEND_ROOT, _REPO_ROOT / "model_artifacts"]
    for root in search_roots:
        if not root.is_dir():
            continue
        for match in root.rglob(PREFERRED_AUDIO_FILENAME):
            return match, f"found under {root}"

    fallback = (
        _BACKEND_ROOT
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "pysptk"
        / "example_audio_data"
        / "arctic_a0007.wav"
    )
    if fallback.is_file():
        return fallback, (
            f"PREFERRED_AUDIO_FILENAME ({PREFERRED_AUDIO_FILENAME}) not found anywhere "
            "under the repo or home directory -- using a bundled real 16kHz mono speech "
            "sample (CMU ARCTIC, shipped with the pysptk dependency) instead"
        )

    raise SystemExit(
        f"{PREFERRED_AUDIO_FILENAME} not found and the fallback sample is unavailable. "
        "Pass --audio /path/to/real/audio.wav explicitly."
    )


def decode_audio(audio_path: Path, app_settings: Settings) -> tuple[Any, dict[str, Any], float]:
    extension = audio_path.suffix.lower().lstrip(".")
    start = perf_counter()
    inspection = inspect_audio_file(audio_path, extension, app_settings)
    processed = preprocess_audio_file(audio_path, extension, app_settings, inspection)
    elapsed_ms = max((perf_counter() - start) * 1000, 0.0)
    info = {
        "filename": audio_path.name,
        "source_format": extension,
        "detected_codec": inspection.detected_codec,
        "detected_container": inspection.detected_container,
        "duration_seconds": round(processed.duration_seconds, 4),
        "decoded_sample_rate": processed.sample_rate,
        "original_sample_rate": processed.original_sample_rate,
        "original_channels": processed.original_channels,
    }
    return processed, info, elapsed_ms


# ---------------------------------------------------------------------------
# Non-invasive DisVoice sub-stage instrumentation (pass-through wrappers)
# ---------------------------------------------------------------------------


class _PassThroughTimer:
    """Wraps a callable with a pass-through timer: calls the original
    function unchanged, records elapsed time, returns the original result
    untouched. Used to split E (temp WAV write) and F (DisVoice extraction)
    out of `extract_disvoice_glottal_features`'s single function boundary
    without editing that function.
    """

    def __init__(self, original: Callable[..., Any]) -> None:
        self._original = original
        self.last_elapsed_ms: float = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        result = self._original(*args, **kwargs)
        self.last_elapsed_ms = max((perf_counter() - start) * 1000, 0.0)
        return result


def instrument_disvoice_substages() -> tuple[_PassThroughTimer, _PassThroughTimer]:
    """Installs pass-through timing wrappers around `soundfile.write` and the
    DisVoice extractor's `extract_features_file`. Returns (write_timer,
    extract_timer); read `.last_elapsed_ms` after each
    `extract_disvoice_glottal_features` call. Caller must call the returned
    restore-safe teardown is handled by `uninstrument_disvoice_substages`.
    """

    import soundfile as sf

    write_timer = _PassThroughTimer(sf.write)
    sf.write = write_timer  # module-level monkeypatch; restored below

    extractor = glottal_v1._get_disvoice_extractor()
    extract_timer = _PassThroughTimer(extractor.extract_features_file)
    extractor.extract_features_file = extract_timer

    return write_timer, extract_timer


def uninstrument_disvoice_substages(write_timer: _PassThroughTimer, extract_timer: _PassThroughTimer) -> None:
    import soundfile as sf

    sf.write = write_timer._original
    extractor = glottal_v1._get_disvoice_extractor()
    extractor.extract_features_file = extract_timer._original


# ---------------------------------------------------------------------------
# Stage-instrumented single pass over the real (unmodified) pipeline
# functions -- A-I are timed by calling each real function directly; J is a
# separate, genuinely end-to-end call through the real production predictor.
# ---------------------------------------------------------------------------


def run_stage_instrumented(
    *,
    audio_path: Path,
    app_settings: Settings,
    runtime_model: LoadedGlottalBranch,
    write_timer: _PassThroughTimer,
    extract_timer: _PassThroughTimer,
) -> dict[str, float]:
    stages: dict[str, float] = {}

    processed_audio, _info, decode_ms = decode_audio(audio_path, app_settings)
    stages["A"] = decode_ms

    waveform = (
        processed_audio.unnormalised_waveform
        if processed_audio.unnormalised_waveform is not None
        else processed_audio.waveform
    )

    t0 = perf_counter()
    cleaned = glottal_v1.clean_glottal_waveform(waveform, processed_audio.sample_rate)
    stages["B"] = max((perf_counter() - t0) * 1000, 0.0)

    t0 = perf_counter()
    voice_quality = glottal_v1.extract_voice_quality_features(cleaned)
    stages["C"] = max((perf_counter() - t0) * 1000, 0.0)

    t0 = perf_counter()
    spectral = glottal_v1.extract_spectral_features(cleaned)
    stages["D"] = max((perf_counter() - t0) * 1000, 0.0)

    t0 = perf_counter()
    disvoice_features = glottal_v1.extract_disvoice_glottal_features(cleaned)
    efg_total_ms = max((perf_counter() - t0) * 1000, 0.0)
    stages["E"] = write_timer.last_elapsed_ms
    stages["F"] = extract_timer.last_elapsed_ms
    stages["G"] = max(efg_total_ms - stages["E"] - stages["F"], 0.0)

    all_features: dict[str, float] = {**voice_quality, **spectral, **disvoice_features}

    t0 = perf_counter()
    vector = glottal_v1.build_selected_feature_vector(all_features, list(runtime_model.selected_features))
    stages["H"] = max((perf_counter() - t0) * 1000, 0.0)

    t0 = perf_counter()
    probabilities = runtime_model.pipeline.predict_proba(vector)[0]
    stages["I"] = max((perf_counter() - t0) * 1000, 0.0)

    stages["_reconstructed_total"] = sum(stages[s] for s in "ABCDEFGHI")
    stages["_spoof_probability"] = float(probabilities[runtime_model.spoof_class_index])
    stages["_selected_feature_count"] = float(vector.shape[1])
    stages["_all_finite"] = float(bool(__import__("numpy").isfinite(vector).all()))

    return stages


# ---------------------------------------------------------------------------
# Real end-to-end predictor call (production predictor function, exactly as
# `RealModelAdapter.predict` invokes it) -- this is stage "J".
# ---------------------------------------------------------------------------


def run_real_predictor(
    *,
    audio_path: Path,
    app_settings: Settings,
    runtime_model: LoadedGlottalBranch,
    config: Any,
    predictor: Callable[..., Any],
) -> tuple[float, Any]:
    processed_audio, _info, _decode_ms = decode_audio(audio_path, app_settings)
    start = perf_counter()
    prediction = predictor(processed_audio, runtime_model, config)
    elapsed_ms = max((perf_counter() - start) * 1000, 0.0)
    return elapsed_ms, prediction


# ---------------------------------------------------------------------------
# Production-path benchmark (VoiceService/factory/loader/predictor, section 5)
# ---------------------------------------------------------------------------


def benchmark_production_path(
    *, audio_path: Path, app_settings: Settings
) -> dict[str, Any]:
    from app.models.real.glottal import GlottalRealAdapter

    factory = ModelFactory(app_settings)
    config = factory.branch_config("glottal")
    loader = build_glottal_loader(app_settings)
    from app.models.real.glottal_inference import build_glottal_predictor

    predictor = build_glottal_predictor()
    adapter = GlottalRealAdapter(config, loader=loader, predictor=predictor)

    assert not adapter.is_loaded
    cold_load_start = perf_counter()
    processed_audio, _info, _decode_ms = decode_audio(audio_path, app_settings)
    cold_prediction = adapter.predict_safe(processed_audio)
    cold_total_ms = max((perf_counter() - cold_load_start) * 1000, 0.0)
    assert adapter.is_loaded

    warm_stats = StageStats()
    last_prediction = cold_prediction
    for _ in range(WARMUP_RUNS):
        processed_audio, _info, _ = decode_audio(audio_path, app_settings)
        adapter.predict_safe(processed_audio)
    for _ in range(MEASURED_RUNS):
        processed_audio, _info, _ = decode_audio(audio_path, app_settings)
        start = perf_counter()
        last_prediction = adapter.predict_safe(processed_audio)
        warm_stats.add(max((perf_counter() - start) * 1000, 0.0))

    return {
        "cold_total_ms": cold_total_ms,
        "cold_prediction_status": cold_prediction.status.value,
        "warm_summary_ms": warm_stats.summary(),
        "warm_prediction_status": last_prediction.status.value,
        "warm_spoof_probability": (
            last_prediction.probabilities.spoof if last_prediction.probabilities else None
        ),
        "model_load_on_request_path": (
            "Loading happens on the FIRST predict() call only (lazy `RealModelAdapter.load()`, "
            "guarded by `is_loaded`); every subsequent request reuses the already-loaded joblib "
            "pipeline + manifest in-process. With MODEL_LOAD_STRATEGY=startup (the deployed "
            "default), loading instead happens once during application startup, so no request "
            "ever pays it."
        ),
    }


# ---------------------------------------------------------------------------
# Full VoiceService pipeline benchmark (section 6)
# ---------------------------------------------------------------------------


class _StageLogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[str, float]] = []

    def emit(self, record: logging.LogRecord) -> None:
        stage = getattr(record, "stage", None)
        duration_ms = getattr(record, "duration_ms", None)
        if stage is not None and duration_ms is not None:
            self.records.append((stage, float(duration_ms)))


def benchmark_full_pipeline(app_settings: Settings, audio_path: Path) -> dict[str, Any] | None:
    import shutil
    import tempfile

    from app.ingestion.audio import save_validated_local_audio_file
    from app.services.voice_service import VoiceService

    service = VoiceService(app_settings=app_settings)

    capture = _StageLogCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(capture)
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)

    # `save_validated_local_audio_file` finalises via `os.replace(temp_path,
    # saved_path)`, which POSIX `rename(2)` refuses across filesystems
    # (EXDEV). This repo lives on an external volume while the system temp
    # dir (`tempfile`'s default) is on the internal boot volume, so the temp
    # copy must be created on the SAME volume as the upload directory, not in
    # system temp.
    upload_dir = app_settings.resolved_upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_copy = Path(tempfile.mktemp(suffix=audio_path.suffix, dir=str(upload_dir)))
    shutil.copy(audio_path, tmp_copy)

    try:
        upload_metadata = save_validated_local_audio_file(
            tmp_copy,
            original_filename=audio_path.name,
            content_type="audio/wav",
            app_settings=app_settings,
        )
        start = perf_counter()
        response = service.predict_from_validated_upload(upload_metadata, cleanup_upload=True)
        total_ms = max((perf_counter() - start) * 1000, 0.0)
    finally:
        root_logger.removeHandler(capture)
        root_logger.setLevel(previous_level)

    stage_from_logs = {stage: duration for stage, duration in capture.records}
    branch_latencies = {branch.model_name: branch.processing_time_ms for branch in response.branches}
    sum_branches_ms = sum(branch_latencies.values())
    preprocessing_ms = stage_from_logs.get("audio_preprocessing")
    fusion_ms_logged = stage_from_logs.get("fusion")
    fusion_ms_derived = (
        total_ms - (preprocessing_ms or 0.0) - sum_branches_ms
        if preprocessing_ms is not None
        else None
    )

    return {
        "total_ms": total_ms,
        "response_total_processing_time_ms": response.total_processing_time_ms,
        "audio_preprocessing_ms_logged": preprocessing_ms,
        "branch_latencies_ms": branch_latencies,
        "branch_statuses": {b.model_name: b.status.value for b in response.branches},
        "fusion_ms_logged": fusion_ms_logged,
        "fusion_ms_derived_from_totals": fusion_ms_derived,
        "fusion_status": response.fusion.status.value,
        "fusion_contributing_branches": response.fusion.contributing_branches,
        "fusion_excluded_branches": response.fusion.excluded_branches,
        "max_branch_latency_ms": max(branch_latencies.values()) if branch_latencies else None,
        "execution_strategy": (
            "SEQUENTIAL. app/services/voice_service.py's "
            "_predict_from_validated_upload runs `for model in self._model_registry.models: "
            "... _predict_branch_with_timeout(...)` as a plain synchronous for-loop -- one "
            "branch completes before the next starts. The per-branch ThreadPoolExecutor(max_"
            "workers=1) inside _predict_branch_with_timeout exists only to enforce a timeout, "
            "not to run branches concurrently with each other."
        ),
        "ideal_parallel_lower_bound_ms": max(branch_latencies.values()) if branch_latencies else None,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_stage_table(stage_stats: dict[str, StageStats]) -> None:
    print(f"{'Stage':<45} {'Mean ms':>10} {'P50 ms':>10} {'P95 ms':>10} {'Max ms':>10}")
    print("-" * 87)
    for stage in STAGE_ORDER:
        stats = stage_stats.get(stage)
        if stats is None or stats.n == 0:
            continue
        summary = stats.summary()
        label = f"{stage}. {STAGE_LABELS[stage]}"
        print(
            f"{label:<45} {summary['mean']:>10.3f} {summary['median']:>10.3f} "
            f"{summary['p95']:>10.3f} {summary['max']:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, default=None, help=f"Real audio file (prefers {PREFERRED_AUDIO_FILENAME}).")
    parser.add_argument("--skip-full-pipeline", action="store_true", help="Skip section 6 (CNN+AASIST+SSL+Glottal).")
    parser.add_argument("--json", action="store_true", help="Also dump the full result as JSON at the end.")
    args = parser.parse_args()

    env = environment_report()
    audio_path, audio_source_note = resolve_audio_path(args.audio)

    glottal_settings = Settings(
        _env_file=None,
        glottal_model_mode="real",
        glottal_model_path="glottal/glottal_logreg_selected20_v1.joblib",
        glottal_selected_features_path="glottal/glottal_selected_features_v1.json",
        model_root_dir="../model_artifacts",
        model_device_policy="cpu",
        model_default_device="cpu",
        model_load_strategy="lazy",
    )

    print("=" * 87)
    print("GLOTTAL BRANCH LATENCY BENCHMARK -- measurement only, no behavior changes")
    print("=" * 87)
    print()
    print("-- Environment --")
    for key, value in env.items():
        print(f"  {key}: {value}")
    print()

    _processed_audio, audio_info, _decode_ms = decode_audio(audio_path, glottal_settings)
    print("-- Audio sample --")
    print(f"  path: {audio_path}")
    print(f"  resolution: {audio_source_note}")
    for key, value in audio_info.items():
        print(f"  {key}: {value}")
    print()

    factory = ModelFactory(glottal_settings)
    config = factory.branch_config("glottal")
    loader = build_glottal_loader(glottal_settings)
    runtime_model = loader(config)
    print(
        f"-- Glottal artifact loaded: {runtime_model.n_features_in} features, "
        f"spoof_class_index={runtime_model.spoof_class_index} --"
    )
    print()

    write_timer, extract_timer = instrument_disvoice_substages()
    try:
        print(f"Running {WARMUP_RUNS} warm-up pass(es) (stage-instrumented)...")
        for _ in range(WARMUP_RUNS):
            run_stage_instrumented(
                audio_path=audio_path,
                app_settings=glottal_settings,
                runtime_model=runtime_model,
                write_timer=write_timer,
                extract_timer=extract_timer,
            )

        print(f"Running {MEASURED_RUNS} measured pass(es) (stage-instrumented)...")
        stage_stats: dict[str, StageStats] = {s: StageStats() for s in "ABCDEFGHI"}
        reconstructed_totals = StageStats()
        validation_ok = True
        last_result: dict[str, float] = {}
        for _ in range(MEASURED_RUNS):
            result = run_stage_instrumented(
                audio_path=audio_path,
                app_settings=glottal_settings,
                runtime_model=runtime_model,
                write_timer=write_timer,
                extract_timer=extract_timer,
            )
            for stage in "ABCDEFGHI":
                stage_stats[stage].add(result[stage])
            reconstructed_totals.add(result["_reconstructed_total"])
            validation_ok = validation_ok and (
                int(result["_selected_feature_count"]) == 20 and bool(result["_all_finite"])
            )
            last_result = result
    finally:
        uninstrument_disvoice_substages(write_timer, extract_timer)

    print(f"Running {WARMUP_RUNS} warm-up + {MEASURED_RUNS} measured real end-to-end predictor call(s) (stage J)...")
    from app.models.real.glottal_inference import build_glottal_predictor

    predictor = build_glottal_predictor()
    j_stats = StageStats()
    last_prediction = None
    for _ in range(WARMUP_RUNS):
        run_real_predictor(
            audio_path=audio_path, app_settings=glottal_settings, runtime_model=runtime_model,
            config=config, predictor=predictor,
        )
    for _ in range(MEASURED_RUNS):
        elapsed_ms, last_prediction = run_real_predictor(
            audio_path=audio_path, app_settings=glottal_settings, runtime_model=runtime_model,
            config=config, predictor=predictor,
        )
        j_stats.add(elapsed_ms)
    stage_stats["J"] = j_stats

    print()
    print("-- Stage-level timings (2 warm-up + 10 measured runs) --")
    print_stage_table(stage_stats)
    print()

    j_summary = j_stats.summary()
    print("-- Glottal total (stage J: real end-to-end predictor call) --")
    print(f"  mean:  {j_summary['mean']:.3f} ms ({j_summary['mean'] / 1000:.4f} s)")
    print(f"  p50:   {j_summary['median']:.3f} ms")
    print(f"  p95:   {j_summary['p95']:.3f} ms")
    print(f"  min:   {j_summary['min']:.3f} ms   max: {j_summary['max']:.3f} ms   stdev: {j_summary['stdev']:.3f} ms")
    print(
        f"  cross-check: sum(A..I) reconstructed total mean = "
        f"{reconstructed_totals.summary()['mean']:.3f} ms (should be close to J; A/I overlap "
        "with J's own internal work, so these are independent measurements of the same pipeline, "
        "not double-counted)"
    )
    print()

    print("-- Production path (VoiceService -> factory -> Glottal loader -> Glottal predictor) --")
    production = benchmark_production_path(audio_path=audio_path, app_settings=glottal_settings)
    print(f"  cold first request (includes model load): {production['cold_total_ms']:.3f} ms")
    print(f"  warm request median (p50): {production['warm_summary_ms']['median']:.3f} ms")
    print(f"  warm request p95:          {production['warm_summary_ms']['p95']:.3f} ms")
    print(f"  {production['model_load_on_request_path']}")
    print()

    full_pipeline: dict[str, Any] | None = None
    if not args.skip_full_pipeline:
        print("-- Full VoiceService pipeline (CNN + AASIST + SSL + Glottal + fusion) --")
        full_settings = Settings(
            _env_file=None,
            cnn_model_mode="real",
            aasist_model_mode="real",
            ssl_model_mode="real",
            glottal_model_mode="real",
            glottal_model_path="glottal/glottal_logreg_selected20_v1.joblib",
            glottal_selected_features_path="glottal/glottal_selected_features_v1.json",
            model_device_policy="cpu",
            model_default_device="cpu",
            model_load_strategy="lazy",
        )
        try:
            full_pipeline = benchmark_full_pipeline(full_settings, audio_path)
        except Exception as error:  # noqa: BLE001 - report, don't crash the rest of the benchmark
            print(f"  SKIPPED (full pipeline run failed): {type(error).__name__}: {error}")
            full_pipeline = None

        if full_pipeline is not None:
            for branch, latency in full_pipeline["branch_latencies_ms"].items():
                status = full_pipeline["branch_statuses"][branch]
                print(f"  {branch:<20} {latency:>10.3f} ms   status={status}")
            fusion_ms = (
                full_pipeline["fusion_ms_logged"]
                if full_pipeline["fusion_ms_logged"] is not None
                else full_pipeline["fusion_ms_derived_from_totals"]
            )
            fusion_label = "logged" if full_pipeline["fusion_ms_logged"] is not None else "derived"
            print(f"  {'fusion (' + fusion_label + ')':<20} {fusion_ms:>10.3f} ms   status={full_pipeline['fusion_status']}")
            print(f"  {'TOTAL request':<20} {full_pipeline['total_ms']:>10.3f} ms")
            print()
            print(f"  execution strategy: {full_pipeline['execution_strategy']}")
            print(f"  ideal parallel lower bound (max branch latency): {full_pipeline['ideal_parallel_lower_bound_ms']:.3f} ms")
    print()

    print("-- Validation --")
    print(f"  selected feature count == 20: {int(last_result.get('_selected_feature_count', 0)) == 20}")
    print(f"  all selected features finite: {bool(last_result.get('_all_finite', 0.0))}")
    print(
        f"  real glottal_spoof_probability generated: {last_prediction.probabilities.spoof if last_prediction and last_prediction.probabilities else 'N/A'}"
    )
    print(f"  production prediction mode: {production['warm_prediction_status']} (real, non-dummy)")
    print(f"  all {MEASURED_RUNS} measured passes passed validation: {validation_ok}")
    print()

    print("-- Bottleneck analysis --")
    stage_means = {s: stage_stats[s].summary()["mean"] for s in "ABCDEFGHI"}
    j_mean = j_summary["mean"]
    biggest_stage = max(stage_means, key=stage_means.get)
    disvoice_pct = (
        (stage_means["E"] + stage_means["F"] + stage_means["G"]) / j_mean * 100 if j_mean else 0.0
    )
    voice_quality_pct = stage_means["C"] / j_mean * 100 if j_mean else 0.0
    sklearn_pct = stage_means["I"] / j_mean * 100 if j_mean else 0.0
    tempwav_pct = stage_means["E"] / j_mean * 100 if j_mean else 0.0
    print(f"  A. Biggest single stage: {biggest_stage} ({STAGE_LABELS[biggest_stage]}), mean {stage_means[biggest_stage]:.3f} ms")
    print(f"  B. DisVoice/IAIF (E+F+G) share of Glottal total: {disvoice_pct:.1f}%")
    print(f"  C. Temp WAV I/O (E) share of Glottal total: {tempwav_pct:.1f}% ({stage_means['E']:.3f} ms mean)")
    print(f"  D. Praat voice-quality (C) share of Glottal total: {voice_quality_pct:.1f}% ({stage_means['C']:.3f} ms mean)")
    print(f"  E. sklearn predict_proba (I) share of Glottal total: {sklearn_pct:.1f}% ({stage_means['I']:.3f} ms mean)")
    if full_pipeline is not None:
        glottal_full = full_pipeline["branch_latencies_ms"].get("glottal_features", 0.0)
        other_branches = {k: v for k, v in full_pipeline["branch_latencies_ms"].items() if k != "glottal_features"}
        glottal_share_of_total = glottal_full / full_pipeline["total_ms"] * 100 if full_pipeline["total_ms"] else 0.0
        slowest_other = max(other_branches, key=other_branches.get) if other_branches else None
        print(f"  F. Glottal share of full request latency: {glottal_share_of_total:.1f}% ({glottal_full:.3f} ms of {full_pipeline['total_ms']:.3f} ms)")
        if slowest_other:
            print(
                f"  G. Slowest non-Glottal branch: {slowest_other} "
                f"({other_branches[slowest_other]:.3f} ms) "
                f"{'>' if other_branches[slowest_other] > glottal_full else '<='} Glottal ({glottal_full:.3f} ms)"
            )
    else:
        print("  F/G. Full-pipeline comparison skipped (see above).")
    print()

    if args.json:
        payload = {
            "environment": env,
            "audio": audio_info,
            "stages": {s: stage_stats[s].summary() for s in STAGE_ORDER if s in stage_stats},
            "reconstructed_total_ms": reconstructed_totals.summary(),
            "production_path": production,
            "full_pipeline": full_pipeline,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
