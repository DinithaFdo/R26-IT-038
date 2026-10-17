"""Measure real temporal, semantic and combined post-classification Voice XAI.

Run from audio-service:
  .venv/Scripts/python -B -m app.voice_xai.benchmarks.run_latency_benchmark --mode all

Models load once. One genuine classifier pass prepares each recording's original
attention evidence, separately timed from XAI. Combined runs call the production
services in order, including visualization, consistency, report and JSON encoding.
HTTP, queueing, database/storage and external narrative generation are excluded.
"""

import argparse
import csv
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
from time import monotonic, perf_counter

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODES = ('temporal', 'semantic', 'combined')


# Keep help and argument validation independent of optional model/DSP libraries.
def parser():
    command = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    command.add_argument('--mode', choices=(*MODES, 'all'), default='all')
    command.add_argument('--audio', type=Path, nargs='+', default=None,
                         help='Recordings; defaults to the bundled audio/*.wav files.')
    command.add_argument('--windows', choices=('enabled', 'disabled', 'both'), default='enabled')
    command.add_argument('--threads', type=int, default=2)
    command.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    command.add_argument('--warmups', type=int, default=1, help='Includes the first call, reported separately.')
    command.add_argument('--repeats', type=int, default=3, help='Measured warm calls per recording/mode.')
    command.add_argument('--model-root', type=Path, help='Override MODEL_ROOT_DIR for this process only.')
    command.add_argument('--output', type=Path, default=HERE / 'results/voice-xai-latency.json',
                         help='JSON report path; defaults to benchmarks/results/voice-xai-latency.json.')
    return command


# Observe power at request boundaries without changing Windows power settings.
def power_state():
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes

    class Power(ctypes.Structure):
        _fields_ = [('ac', ctypes.c_ubyte), ('flags', ctypes.c_ubyte),
                    ('percent', ctypes.c_ubyte), ('saver', ctypes.c_ubyte),
                    ('life', wintypes.DWORD), ('full', wintypes.DWORD)]

    value = Power()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)):
        return None
    return {0: False, 1: True}.get(value.ac)


# Observe the existing extractor; all decoding, locking and DSP remain delegated.
class ExtractionObserver:
    def __init__(self, target):
        self.target = target
        self.reset()

    # Clear only observations, retaining the real extractor and its caches.
    def reset(self):
        self.cleaned_seconds = None
        self.full_seconds = self.window_seconds = 0.

    # Time public extraction calls and retain the measured full-clip duration.
    def __getattr__(self, name):
        method = getattr(self.target, name)
        if name not in {'extract_processed_audio', 'extract_prepared_audio', 'extract_prepared_window'}:
            return method

        def observed(*args, **kwargs):
            started = perf_counter()
            try:
                result = method(*args, **kwargs)
                if name != 'extract_prepared_window':
                    self.cleaned_seconds = result.duration_seconds
                return result
            finally:
                elapsed = perf_counter() - started
                if name == 'extract_prepared_window':
                    self.window_seconds += elapsed
                else:
                    self.full_seconds += elapsed

        return observed


# Own actual production services; never synthesize classifier scores or attention.
class Runtime:
    def __init__(self, args):
        import torch
        from app.config.settings import settings
        from app.models.registry import ModelRegistry
        from app.services.voice_service import VoiceService
        from app.voice_xai.temporal.xlsr_attention_service import XLSRTemporalAttentionService
        from app.voice_xai.semantic.service import ProductionSemanticExplanationService
        from app.voice_xai.semantic.contracts import SemanticWindowOptions
        from app.voice_xai.report.service import DeterministicReportComposer

        torch.set_num_threads(args.threads)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable.')
        self.torch, self.device = torch, args.device
        changes = {'xai_enabled': True, 'xai_mode': 'real', 'xai_narrative_enabled': False,
                   'model_device_policy': args.device, 'model_default_device': args.device,
                   'model_allow_cpu_fallback': False, 'ssl_xlsr_local_files_only': True}
        for branch in ('lfcc_cnn_tcn', 'aasist', 'ssl_sequence', 'glottal'):
            changes[f'{branch}_device'] = args.device
        if args.model_root:
            changes['model_root_dir'] = str(args.model_root.resolve())
        self.settings = settings.model_copy(update=changes)
        self.registry = ModelRegistry(app_settings=self.settings)
        if any(model.mode.value != 'real' for model in self.registry.models):
            raise RuntimeError('Set CNN_MODEL_MODE, AASIST_MODEL_MODE, SSL_MODEL_MODE and GLOTTAL_MODEL_MODE to real.')
        self.startup = {}
        for model in self.registry.models:
            print(f'Loading {model.branch_name} (local files only for XLS-R)...', flush=True)
            started = perf_counter()
            model.load()
            self.startup[f'{model.branch_name}_model_seconds'] = perf_counter() - started
            if model.branch_name == 'ssl_sequence':
                original = model.predict_safe_with_temporal_evidence

                def capture(audio, original=original):
                    self.synchronize()
                    started = perf_counter()
                    try:
                        return original(audio)
                    finally:
                        self.synchronize()
                        self.ssl_seconds = perf_counter() - started

                model.predict_safe_with_temporal_evidence = capture
        self.temporal = XLSRTemporalAttentionService(app_settings=self.settings)
        started = perf_counter()
        self.semantic = ProductionSemanticExplanationService.load_once(
            model_dir=self.settings.resolve_xai_artifact_path(self.settings.xai_semantic_model_dir),
            window_options=SemanticWindowOptions.from_settings(self.settings))
        self.startup['semantic_model_seconds'] = perf_counter() - started
        self.extractor = ExtractionObserver(self.semantic._feature_extractor)
        self.semantic._feature_extractor = self.extractor
        self.composer = DeterministicReportComposer()
        self.voice = VoiceService(model_registry=self.registry, app_settings=self.settings,
                                  preprocess_fn=self.preprocess)
        self.coverage = None

    # Synchronize asynchronous GPU work around measured boundaries when requested.
    def synchronize(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    # Use the production ffprobe/FFmpeg preprocessing with this process's settings.
    def preprocess(self, upload):
        from app.ingestion.audio import preprocess_audio_file
        started = perf_counter()
        try:
            return preprocess_audio_file(upload.saved_path, upload.original_extension,
                                         self.settings, inspection=upload.inspection)
        finally:
            self.preprocessing_seconds = perf_counter() - started

    # Prepare the exact classifier/XAI handoff once, outside the XAI timing loop.
    def prepare(self, path):
        from app.ingestion.audio import AudioUploadMetadata, inspect_audio_file
        from app.schemas.common import SourceType
        from app.voice_xai.evaluation.consistency import span_duration
        from app.voice_xai.integration.contracts import ClassifierInferenceBundle
        extension = path.suffix.lstrip('.').lower()
        inspection = inspect_audio_file(path, extension, self.settings)
        upload = AudioUploadMetadata(path.name, path.name, path.name, path, None,
                                     path.stat().st_size, inspection.duration_seconds,
                                     inspection.sample_rate, inspection.channels,
                                     original_extension=extension, inspection=inspection)
        self.ssl_seconds = self.preprocessing_seconds = 0.
        self.synchronize()
        started = perf_counter()
        execution = self.voice.predict_from_validated_upload_with_processed_audio(
            upload, cleanup_upload=False)
        self.synchronize()
        elapsed = perf_counter() - started
        failed = [branch.model_name for branch in execution.prediction.branches
                  if branch.status.value != 'success' or branch.mode.value != 'real']
        if failed or execution.prediction.fusion.status.value != 'success':
            raise RuntimeError(f'Genuine classifier context is incomplete: {failed or "fusion failed"}.')
        if not execution.temporal_evidence:
            raise RuntimeError('The original XLS-R pass returned no temporal evidence.')
        bundle = ClassifierInferenceBundle(path.stem, execution.prediction.request_id,
            'local-latency-benchmark', SourceType.dashboard_upload,
            execution.prediction, temporal_evidence=execution.temporal_evidence,
            processed_audio=execution.processed_audio)
        self.coverage = None
        return bundle, {'audio': path.name, 'sha256': sha256(path.read_bytes()).hexdigest(),
                        'source_seconds': inspection.duration_seconds,
                        'classifier_with_preprocessing_seconds': elapsed,
                        'preprocessing_seconds': self.preprocessing_seconds,
                        'ssl_forward_and_attention_reduction_seconds': self.ssl_seconds,
                        'temporal_evidence_seconds': span_duration(
                            (w.start_seconds, w.end_seconds) for w in execution.temporal_evidence)}


# Retain partial/failed timing rows, but never count incomplete windows as successful speed.
def window_status(localization, enabled):
    if not enabled:
        return {'status': 'completed', 'planned_windows': 0, 'completed_windows': 0, 'failed_windows': 0}
    if localization is None or not localization.windows:
        return {'status': 'partial', 'planned_windows': 0, 'completed_windows': 0,
                'failed_windows': 0, 'error': 'Requested localization produced no windows.'}
    completed = sum(getattr(window.status, 'value', window.status) == 'completed'
                    for window in localization.windows)
    count = len(localization.windows)
    result = {'status': 'completed' if completed == count else 'partial',
            'planned_windows': count, 'completed_windows': completed, 'failed_windows': count - completed,
            'semantic_coverage_percent': localization.coverage_percentage,
            'localization_seconds': localization.processing_seconds}
    if completed != count:
        reasons = {getattr(window, 'reason', None) or 'unavailable window' for window in localization.windows
                   if getattr(window.status, 'value', window.status) != 'completed'}
        result['error'] = 'Incomplete windows: ' + '; '.join(sorted(reasons))
    return result


# Time one actual core pipeline, including response construction and JSON encoding.
def measure(runtime, bundle, mode, windows_enabled):
    from app.voice_xai.evaluation.consistency import quality_for_analysis
    from app.voice_xai.evaluation.eligibility import evaluate_research_eligibility
    from app.voice_xai.temporal.spectrogram import mel_spectrogram_payload
    from app.voice_xai.workflow.orchestrator import classifier_snapshot_from_prediction
    runtime.extractor.reset()
    runtime.semantic.window_options = replace(runtime.semantic.window_options, enabled=windows_enabled)
    output = {'status': 'failed', 'ac_before': power_state()}
    timings, response = {}, {}
    runtime.synchronize()
    started = perf_counter()

    def stage(name, function):
        runtime.synchronize()
        before = perf_counter()
        try:
            return function()
        finally:
            runtime.synchronize()
            timings[name] = perf_counter() - before

    try:
        deadline = monotonic() + runtime.settings.xai_job_timeout_seconds
        temporal = semantic = None
        if mode in ('temporal', 'combined'):
            temporal = stage('temporal_seconds', lambda: runtime.temporal.analyze(bundle))
            runtime.coverage = temporal.visualization.analyzed_spans
            output['temporal_research_eligible'] = temporal.explanation.research_eligible
            response['temporal'] = temporal.explanation.model_dump(mode='json')
            response['visualization'] = temporal.visualization.as_json_dict()
            response['spectrogram'] = stage('spectrogram_seconds', lambda: mel_spectrogram_payload(
                bundle.processed_audio, temporal.explanation, temporal.visualization,
                n_mels=runtime.settings.xai_spectrogram_n_mels,
                n_fft=runtime.settings.xai_spectrogram_n_fft,
                hop_length=runtime.settings.xai_spectrogram_hop_length,
                max_frames=runtime.settings.xai_spectrogram_max_frames))
        if mode in ('semantic', 'combined'):
            if runtime.coverage is None:
                raise RuntimeError('Real validated temporal coverage was not prepared.')
            semantic = stage('semantic_seconds', lambda: runtime.semantic.analyze(
                bundle, analyzed_spans=runtime.coverage, deadline=deadline))
            response['semantic'] = semantic.explanation.model_dump(mode='json')
            output['semantic_research_eligible'] = semantic.explanation.research_eligible
            output.update(window_status(semantic.explanation.localization, windows_enabled))
        else:
            output['status'] = 'completed'
        if mode == 'combined':
            quality = stage('consistency_seconds', lambda: quality_for_analysis(
                temporal.explanation, semantic.explanation, temporal_spans=runtime.coverage))
            classifier = classifier_snapshot_from_prediction(bundle.prediction)
            response['quality'] = quality.model_dump(mode='json')
            response['eligibility'] = asdict(evaluate_research_eligibility(
                classifier=classifier, temporal=temporal.explanation,
                semantic=semantic.explanation))
            report = stage('report_seconds', lambda: runtime.composer.compose(
                classifier=classifier, temporal=temporal.explanation,
                semantic=semantic.explanation, quality=quality))
            response['report'] = report.model_dump(mode='json')
        stage('json_seconds', lambda: json.dumps(response, allow_nan=False))
        if monotonic() >= deadline:
            output.update(status='partial', error='The XAI processing deadline was exceeded.')
    except Exception as error:
        output.update(status='failed', error=f'{type(error).__name__}: {error}')
    runtime.synchronize()
    output.update(total_seconds=perf_counter() - started, **timings,
                  cleaned_seconds=runtime.extractor.cleaned_seconds,
                  full_clip_extraction_seconds=runtime.extractor.full_seconds,
                  window_extraction_seconds=runtime.extractor.window_seconds,
                  ac_after=power_state())
    return output


# Summarize successful warm calls only; keep failures/partials explicit in every group.
def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['audio'], row['mode'], row['windows'])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (audio, mode, windows), group in groups.items():
        times = sorted(row['total_seconds'] for row in group
                       if row['phase'] == 'warm' and row['status'] == 'completed')
        p95 = None
        if times:
            index = (len(times) - 1) * .95
            p95 = times[math.floor(index)] + (times[math.ceil(index)] - times[math.floor(index)]) * (index % 1)
        summaries.append({'audio': audio, 'mode': mode, 'windows': windows,
                          'first_call_seconds': next(row['total_seconds'] for row in group if row['phase'] == 'first'),
                          'first_call_status': next(row['status'] for row in group if row['phase'] == 'first'),
                          'warm_completed': len(times),
                          'median_seconds': statistics.median(times) if times else None,
                          'p95_seconds': p95, 'min_seconds': min(times) if times else None,
                          'max_seconds': max(times) if times else None,
                          'p95_is_small_sample': len(times) < 20,
                          'incomplete_calls': sum(row['status'] != 'completed' for row in group)})
    return summaries


# Persist progress atomically and export a flat table for spreadsheet demonstrations.
def save(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    rows = report['runs']
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['audio', 'mode', 'status']
    with output.with_suffix('.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# Run independent mode calls from genuine input; all-mode calls share warm model caches.
def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if min(args.threads, args.repeats, args.warmups) < 1:
        command.error('threads, warmups and repeats must be positive integers.')
    audio = [path.resolve() for path in (args.audio or sorted((HERE / 'audio').glob('*.wav')))]
    output = args.output.resolve()
    if not audio or any(not path.is_file() for path in audio) or len({p.name for p in audio}) != len(audio):
        command.error('Provide existing audio files with unique names.')
    if output.suffix.lower() != '.json' or any(p in {output, output.with_suffix('.csv'), output.with_suffix('.json.tmp')} for p in audio):
        command.error('Output must be a JSON report separate from input audio.')
    # Keep reports in the benchmark results directory or outside production code/model directories.
    protected = [ROOT / 'model_artifacts']
    if args.model_root:
        protected.append(args.model_root.resolve())
    inside_production_code = output.is_relative_to(ROOT / 'app') and not output.is_relative_to(HERE / 'results')
    if inside_production_code or any(output.is_relative_to(directory) for directory in protected):
        command.error('Write results under benchmarks/results or outside app/model directories.')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    report = {'version': 'voice-xai-latency-v1', 'passed': False, 'startup': {}, 'inputs': [], 'runs': [],
              'configuration': {'mode': args.mode, 'windows': args.windows, 'device': args.device,
                                'threads': args.threads, 'warmups': args.warmups, 'repeats': args.repeats},
              'scope': 'Post-classification XAI services, visualization/spectrogram, consistency, deterministic report and JSON; classifier/evidence preparation measured separately. HTTP, queue, DB/storage and external narrative excluded.',
              'notes': ['First call per mode follows model/classifier setup, not a process-cold measurement. All modes share in-memory caches; existing JIT/OS disk caches are retained.',
                        'Warm summaries exclude warmups, failed and partial calls; fewer than 20 completed warm samples gives only a descriptive p95.',
                        'passed means all measurements completed; it does not certify a 12-second target or research validity.',
                        'Coverage includes real analyzed_spans only; partial status is based on incomplete planned windows, not trimmed-silence percentage alone.']}
    runtime = None
    try:
        started = perf_counter()
        runtime = Runtime(args)
        report['startup'] = {**runtime.startup, 'initialization_seconds': perf_counter() - started}
        print(f"Model/runtime initialization (outside XAI timings): {report['startup']['initialization_seconds']:.3f}s", flush=True)
        report['semantic_model_fingerprint'] = runtime.semantic.artifacts.fingerprint
        report['semantic_window_configuration'] = asdict(runtime.semantic.window_options)
        report['temporal_configuration'] = {name: getattr(runtime.settings, name) for name in (
            'xai_temporal_global_hop_seconds', 'xai_temporal_minimum_region_duration_seconds',
            'xai_temporal_merge_gap_seconds', 'xai_spectrogram_n_mels', 'xai_spectrogram_n_fft',
            'xai_spectrogram_hop_length', 'xai_spectrogram_max_frames')}
        report['runner_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
        report['model_root'] = str(runtime.settings.resolved_model_root_dir)
        report['packages'] = {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'numba', 'scipy', 'librosa', 'xgboost', 'shap')}
        from threadpoolctl import threadpool_limits
        modes = MODES if args.mode == 'all' else (args.mode,)
        options = (False, True) if args.windows == 'both' else (args.windows == 'enabled',)
        with threadpool_limits(limits=args.threads):
            for path in audio:
                print(f'Preparing real classifier/attention context: {path.name}', flush=True)
                bundle, setup = runtime.prepare(path)
                report['inputs'].append(setup)
                print(f"Setup outside XAI timings: classifier+preprocessing={setup['classifier_with_preprocessing_seconds']:.3f}s; "
                      f"shared SSL forward+attention reduction={setup['ssl_forward_and_attention_reduction_seconds']:.3f}s; "
                      f"source={setup['source_seconds']:.3f}s; temporal evidence={setup['temporal_evidence_seconds']:.3f}s", flush=True)
                if modes == ('semantic',):
                    before = perf_counter()
                    runtime.coverage = runtime.temporal.analyze(bundle).visualization.analyzed_spans
                    setup['temporal_coverage_setup_seconds'] = perf_counter() - before
                for mode in modes:
                    for enabled in ((False,) if mode == 'temporal' else options):
                        for repeat in range(args.warmups + args.repeats):
                            phase = 'first' if repeat == 0 else 'warmup' if repeat < args.warmups else 'warm'
                            row = {'audio': path.name, 'mode': mode, 'windows': 'n/a' if mode == 'temporal' else 'enabled' if enabled else 'disabled',
                                   'phase': phase, 'repeat': repeat + 1, **measure(runtime, bundle, mode, enabled)}
                            report['runs'].append(row)
                            print(f"{path.name} {mode:8} {row['windows']:8} {phase:6} {row['status']:9} {row['total_seconds']:.3f}s" + (f" | {row['error']}" if row.get('error') else ''), flush=True)
                            save(output, report)
        report['summaries'] = summarize(report['runs'])
        report['passed'] = bool(report['runs']) and all(row['status'] == 'completed' for row in report['runs'])
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        print(report['error'], file=sys.stderr, flush=True)
    finally:
        save(output, report)
        if runtime is not None:
            runtime.registry.unload_all()
    print('\nWarm summary (successful calls only):', flush=True)
    for row in report.get('summaries', []):
        median = f"{row['median_seconds']:.3f}s" if row['median_seconds'] is not None else 'unavailable'
        p95 = f"{row['p95_seconds']:.3f}s" if row['p95_seconds'] is not None else 'unavailable'
        print(f"{row['audio']} {row['mode']:8} {row['windows']:8} median={median} p95={p95} incomplete={row['incomplete_calls']}", flush=True)
    if any(row['p95_is_small_sample'] for row in report.get('summaries', [])):
        print('p95 is descriptive for groups with fewer than 20 successful warm calls.', flush=True)
    print(f'JSON: {output}\nCSV:  {output.with_suffix(".csv")}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
