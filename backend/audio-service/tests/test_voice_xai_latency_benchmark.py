"""Latency measurement boundaries, incomplete evidence and safe report output."""
from types import SimpleNamespace
import json

import pytest

from app.schemas.xai import ComponentStatus
from app.voice_xai.semantic.contracts import SemanticWindowOptions
from app.voice_xai.benchmarks import run_latency_benchmark as benchmark


class Response:
    research_eligible = False

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def model_dump(self, **kwargs):
        return {'value': 'controlled unit-test response'}


@pytest.fixture
def runtime(monkeypatch):
    from app.voice_xai.evaluation import consistency
    from app.voice_xai.temporal import spectrogram
    from app.voice_xai.workflow import orchestrator
    calls = []
    extractor = SimpleNamespace(reset=lambda: None, cleaned_seconds=1., full_seconds=.1, window_seconds=.2)
    windows = [SimpleNamespace(status=ComponentStatus.completed)] * 2
    localization = SimpleNamespace(windows=windows, processing_seconds=.2, coverage_percentage=75.)
    spans = ((1., 3.),)
    visualization = SimpleNamespace(analyzed_spans=spans, as_json_dict=lambda: {'spans': spans})

    def temporal(bundle):
        calls.append('temporal')
        return SimpleNamespace(explanation=Response(), visualization=visualization)

    def semantic(bundle, *, analyzed_spans, deadline):
        calls.append('semantic')
        assert analyzed_spans == spans and deadline > 0
        return SimpleNamespace(explanation=Response(localization=localization))

    def quality(*args, **kwargs):
        calls.append('quality')
        assert kwargs['temporal_spans'] == spans
        return Response()

    def compose(**kwargs):
        calls.append('report')
        return Response()

    def mel(*args, **kwargs):
        calls.append('spectrogram')
        return {'values': []}

    monkeypatch.setattr(consistency, 'quality_for_analysis', quality)
    monkeypatch.setattr(spectrogram, 'mel_spectrogram_payload', mel)
    monkeypatch.setattr(orchestrator, 'classifier_snapshot_from_prediction', lambda _: SimpleNamespace(
        research_eligible=False, contains_dummy_branches=False))
    return SimpleNamespace(
        calls=calls, extractor=extractor, synchronize=lambda: None, coverage=spans,
        semantic=SimpleNamespace(window_options=SemanticWindowOptions(), analyze=semantic),
        temporal=SimpleNamespace(analyze=temporal), composer=SimpleNamespace(compose=compose),
        settings=SimpleNamespace(xai_job_timeout_seconds=600, xai_spectrogram_n_mels=64,
                                 xai_spectrogram_n_fft=512, xai_spectrogram_hop_length=160,
                                 xai_spectrogram_max_frames=900))


# Combined timing runs the actual sequence rather than adding independent results.
@pytest.mark.parametrize('mode,expected', [
    ('temporal', ['temporal', 'spectrogram']),
    ('semantic', ['semantic']),
    ('combined', ['temporal', 'spectrogram', 'semantic', 'quality', 'report']),
])
def test_modes_preserve_boundaries_and_real_coverage(runtime, mode, expected):
    bundle = SimpleNamespace(processed_audio=object(), prediction=object())
    result = benchmark.measure(runtime, bundle, mode, True)
    assert result['status'] == 'completed', result
    assert runtime.calls == expected
    assert result['total_seconds'] >= result['json_seconds'] >= 0
    if mode != 'temporal':
        assert result['completed_windows'] == 2
        assert result['semantic_coverage_percent'] == 75


# Incomplete windows cannot masquerade as fast successful localization.
@pytest.mark.parametrize('completed,count', [(0, 0), (0, 2), (1, 2)])
def test_incomplete_window_support_is_partial(completed, count):
    windows = [SimpleNamespace(status=ComponentStatus.completed)] * completed
    windows += [SimpleNamespace(status=ComponentStatus.not_available)] * (count - completed)
    loc = SimpleNamespace(windows=windows, processing_seconds=.1, coverage_percentage=25.)
    assert benchmark.window_status(loc, True)['status'] == 'partial'


# Production window status is a string literal; support it as well as component enums.
@pytest.mark.parametrize('status', ['completed', ComponentStatus.completed])
def test_completed_literal_or_enum_windows(status):
    loc = SimpleNamespace(windows=[SimpleNamespace(status=status)],
                          processing_seconds=.1, coverage_percentage=75.)
    assert benchmark.window_status(loc, True)['status'] == 'completed'


# A stage error stays visible, with elapsed failure time retained outside success statistics.
def test_stage_error_is_not_zero_or_success(runtime):
    def fail(*args):
        raise ValueError('controlled temporal failure')

    runtime.temporal.analyze = fail
    result = benchmark.measure(runtime, SimpleNamespace(), 'combined', True)
    assert result['status'] == 'failed' and 'controlled temporal failure' in result['error']
    assert result['total_seconds'] >= result['temporal_seconds'] > 0


# Partial/failed warm requests remain counted while their times do not lower the median.
def test_summary_excludes_incomplete_calls():
    rows = [dict(audio='a.wav', mode='combined', windows='enabled', phase=phase,
                 status=status, total_seconds=seconds)
            for phase, status, seconds in [('first', 'completed', 9.), ('warm', 'failed', .01),
                                           ('warm', 'partial', .02), ('warm', 'completed', 4.),
                                           ('warm', 'completed', 6.)]]
    summary = benchmark.summarize(rows)[0]
    assert summary['warm_completed'] == 2 and summary['incomplete_calls'] == 2
    assert summary['median_seconds'] == 5. and summary['p95_seconds'] == pytest.approx(5.9)
    assert summary['first_call_seconds'] == 9. and summary['p95_is_small_sample']


# CSV and strict JSON contain the same saved measurements, including failures.
def test_report_export_preserves_status_and_numeric_times(tmp_path):
    path = tmp_path / 'run.json'
    report = {'runs': [{'audio': 'a.wav', 'mode': 'temporal', 'status': 'failed', 'total_seconds': .2}]}
    benchmark.save(path, report)
    assert json.loads(path.read_text()) == report
    assert 'failed' in path.with_suffix('.csv').read_text()
    assert not path.with_suffix('.json.tmp').exists()


# Invalid arguments and paths fail before optional models load or files are overwritten.
@pytest.mark.parametrize('arguments', [
    ['--threads', '0'], ['--repeats', '0'], ['--warmups', '0'],
    ['--output', 'app/voice_xai/semantic/artifacts/training_metadata.json'],
])
def test_invalid_cli_arguments_do_not_start_models(arguments, tmp_path, monkeypatch):
    audio = tmp_path / 'a.wav'
    audio.write_bytes(b'input must survive')
    monkeypatch.setattr(benchmark, 'Runtime', lambda _: pytest.fail('Models must not load.'))
    with pytest.raises(SystemExit) as result:
        benchmark.main(['--audio', str(audio), *arguments])
    assert result.value.code == 2 and audio.read_bytes() == b'input must survive'
