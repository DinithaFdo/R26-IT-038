"""Single-audio temporal AOPC using the Temporal_AOPC_10Pilot_100Evaluation recipe.

Run from audio-service (same default audio as the semantic AOPC benchmark):
  .venv/Scripts/python -B -m app.voice_xai.benchmarks.benchmark_temporal_aopc
  .venv/Scripts/python -B -m app.voice_xai.benchmarks.benchmark_temporal_aopc --self-test

Rank 100-ms blocks by ORIGINAL continuous XLS-R attention, cumulatively silence
10/20/30/40/50% of decoded samples, and compare 10 length-matched random orders.
Keep the original predicted class and divide the five score drops by six.
Uses the current backend checkpoint and front end; short windows retain backend
six-second padding (the notebook pads only to a minimum of one second). Long
recordings use the notebook's six-second/three-second full-clip max-window rule.
Results: benchmarks/results/temporal_aopc/<audio>_<UTC>/; figures are PNG only.
This measures the XLS-R agent's ranking, not the fused ensemble or semantic SHAP.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_AUDIO = HERE / 'audio' / 'LA_D_2130605.wav'
BUDGETS = (.1, .2, .3, .4, .5)
SAMPLE_RATE = 16000


def parser():
    command = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    command.add_argument('--audio', type=Path, default=DEFAULT_AUDIO, help='Exactly one recording.')
    command.add_argument('--random-repeats', type=int, default=10)
    command.add_argument('--seed', type=int, default=42)
    command.add_argument('--threads', type=int, default=2)
    command.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    command.add_argument('--model-root', type=Path, help='Optional local MODEL_ROOT_DIR override.')
    command.add_argument('--output-dir', type=Path, help='New result directory; existing directories are rejected.')
    command.add_argument('--self-test', action='store_true', help='Run built-in protocol checks without loading XLS-R.')
    return command


def stable_seed(*parts):
    return int(sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16) % (2 ** 32)


def window_starts(n_samples, window_samples, stride_samples):
    if min(n_samples, window_samples, stride_samples) < 1 or stride_samples > window_samples:
        raise ValueError('Invalid waveform/window/stride length.')
    if n_samples <= window_samples:
        return [0]
    return sorted(set(range(0, n_samples - window_samples + 1, stride_samples)) | {n_samples - window_samples})


def ranked_blocks(times, density, n_samples, sample_rate, block_seconds=.1):
    times, density = np.asarray(times), np.asarray(density)
    block_samples = int(round(block_seconds * sample_rate))
    if (n_samples < 1 or block_samples < 1 or times.ndim != 1 or times.size == 0
            or times.shape != density.shape or not np.isfinite(times).all()
            or not np.isfinite(density).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid waveform or original attention timeline.')
    starts = np.arange(0, n_samples, block_samples, dtype=np.int64)
    ends = np.minimum(starts + block_samples, n_samples)
    means = []
    for start, end in zip(starts, ends):
        selected = (times >= start / sample_rate) & (times < end / sample_rate)
        means.append(float(np.mean(density[selected])) if selected.any() else
                     float(np.interp((start + end) / (2 * sample_rate), times, density)))
    means = np.asarray(means)
    return starts, ends, means, np.lexsort((starts, -means))


def matched_random_order(starts, ends, ranked, seed):
    lengths, rng = ends - starts, np.random.default_rng(seed)
    pools = {int(length): list(rng.permutation(np.flatnonzero(lengths == length)))
             for length in sorted(set(lengths.tolist()))}
    order = np.array([pools[int(lengths[block])].pop() for block in ranked], dtype=np.int64)
    if len(np.unique(order)) != len(starts) or not np.array_equal(lengths[ranked], lengths[order]):
        raise ValueError('Random order must preserve the ranked block-length sequence.')
    return order


def intervals_for_budget(starts, ends, order, budget):
    if not 0 <= budget <= int(ends[-1]):
        raise ValueError('Invalid masking sample budget.')
    remaining, intervals = int(budget), []
    for block in order:
        if remaining == 0:
            break
        start, end = int(starts[block]), int(ends[block])
        count = min(end - start, remaining)
        intervals.append([start, start + count])
        remaining -= count
    if remaining:
        raise ValueError('Masking budget was not reached.')
    return intervals


def silence(waveform, intervals):
    masked = np.asarray(waveform, dtype=np.float32).copy()
    for start, end in intervals:
        masked[start:end] = 0
    return masked


def target_score(p_spoof, target):
    if not np.isfinite(p_spoof) or not 0 <= p_spoof <= 1:
        raise ValueError('Invalid XLS-R probability.')
    return float(p_spoof if target == 1 else 1 - p_spoof)


def aopc(original_score, step_scores):
    scores = np.asarray(step_scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all() or not np.isfinite(original_score):
        raise ValueError('Need finite original and perturbed scores.')
    return float(np.sum(original_score - scores) / (len(scores) + 1))


def evaluate(waveform, times, density, original_p, predict, *, random_repeats, seed, audio_hash, progress=print):
    import pandas as pd

    target = int(original_p >= .5)
    original_q = target_score(original_p, target)
    starts, ends, means, ranked = ranked_blocks(times, density, len(waveform), SAMPLE_RATE)
    budgets = [int(round(len(waveform) * fraction)) for fraction in BUDGETS]
    if min(budgets) < 1 or len(set(budgets)) != len(budgets):
        raise ValueError('Audio is too short for the five distinct sample budgets.')
    orders = [('attention_morf', 0, ranked)] + [
        ('random', repeat, matched_random_order(starts, ends, ranked,
         stable_seed(seed, audio_hash, 'random_order', repeat))) for repeat in range(random_repeats)]
    curves, trials, geometry = [], [], []
    for method, repeat, order in orders:
        curves.append({'method': method, 'repeat': repeat, 'step': 0, 'masked_samples': 0,
                       'masked_fraction': 0., 'p_spoof': original_p, 'target_score': original_q, 'score_drop': 0.})
        scores = []
        for step, budget in enumerate(budgets, 1):
            intervals = intervals_for_budget(starts, ends, order, budget)
            reference = intervals_for_budget(starts, ends, ranked, budget)
            if [b - a for a, b in intervals] != [b - a for a, b in reference]:
                raise ValueError('Random masking geometry differs from attention masking.')
            masked = silence(waveform, intervals)
            p = float(predict(masked))
            q = target_score(p, target)
            scores.append(q)
            curves.append({'method': method, 'repeat': repeat, 'step': step, 'masked_samples': budget,
                           'masked_fraction': budget / len(waveform), 'p_spoof': p,
                           'target_score': q, 'score_drop': original_q - q})
            if method == 'attention_morf':
                progress(f'Attention: {step}/5 ({100 * budget / len(waveform):.1f}% silenced)', flush=True)
        trials.append({'method': method, 'repeat': repeat, 'aopc': aopc(original_q, scores)})
        geometry.append({'method': method, 'repeat': repeat, 'block_order': order.tolist()})
        if method == 'random':
            progress(f'Random order {repeat + 1}/{random_repeats} complete', flush=True)
    inverse_rank = np.empty(len(ranked), dtype=np.int64)
    inverse_rank[ranked] = np.arange(1, len(ranked) + 1)
    blocks = pd.DataFrame({'block_id': np.arange(len(starts)), 'start_sample': starts, 'end_sample_exclusive': ends,
                           'start_seconds': starts / SAMPLE_RATE, 'end_seconds': ends / SAMPLE_RATE,
                           'mean_attention_density': means, 'attention_rank': inverse_rank})
    trials = pd.DataFrame(trials)
    return {'target': 'SPOOF' if target else 'BONAFIDE', 'original_target_score': original_q,
            'curves': pd.DataFrame(curves), 'trials': trials, 'blocks': blocks,
            'summary': trials.groupby('method', as_index=False)['aopc'].mean(),
            'geometry': {'n_samples': len(waveform), 'sample_rate': SAMPLE_RATE,
                         'budget_samples': budgets, 'orders': geometry}}


class TemporalRuntime:
    """Load only the real SSL agent; reuse production inference and rollout."""

    def __init__(self, args):
        import torch
        from app.config.settings import settings
        from app.models.factory import ModelFactory
        from app.models.real.ssl_sequence_inference import build_ssl_loader, build_ssl_predictor
        from app.voice_xai.temporal.xlsr_attention_service import XLSRTemporalAttentionService

        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable. Use --device cpu.')
        changes = {'ssl_model_mode': 'real', 'model_device_policy': args.device,
                   'model_default_device': args.device, 'ssl_sequence_device': args.device,
                   'model_allow_cpu_fallback': False, 'ssl_xlsr_local_files_only': True,
                   'xai_temporal_global_hop_seconds': .02}
        if args.model_root:
            changes['model_root_dir'] = str(args.model_root.resolve())
        self.settings = settings.model_copy(update=changes)
        self.config = ModelFactory(self.settings).branch_config('ssl_sequence')
        if not self.config.checkpoint.valid:
            raise ValueError(f'SSL checkpoint unavailable: {self.config.checkpoint.error_code}')
        print(f'Loading local XLS-R + Mamba on {args.device}...', flush=True)
        self.model = build_ssl_loader(self.settings)(self.config)
        self.predictor = build_ssl_predictor()
        self.temporal = XLSRTemporalAttentionService(app_settings=self.settings)
        self.window_samples = self.model.front_end.config.target_samples
        if self.window_samples != 6 * SAMPLE_RATE:
            raise ValueError('This protocol requires the six-second XLS-R window.')

    @staticmethod
    def processed(waveform):
        from app.ingestion.audio import ProcessedAudio
        return ProcessedAudio(waveform=waveform, sample_rate=SAMPLE_RATE,
                              original_sample_rate=SAMPLE_RATE, original_channels=1,
                              duration_seconds=len(waveform) / SAMPLE_RATE,
                              was_resampled=False, was_converted_to_mono=False,
                              normalisation_applied=False, peak_amplitude=float(np.max(np.abs(waveform))),
                              rms_energy=float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))))

    def predict(self, waveform, *, attention=False):
        from app.models.real.ssl_sequence_inference import classify_ssl_with_temporal_evidence
        probabilities, evidence, branches = [], [], []
        for start in window_starts(len(waveform), self.window_samples, 3 * SAMPLE_RATE):
            audio = self.processed(waveform[start:start + self.window_samples])
            if attention:
                branch, windows = classify_ssl_with_temporal_evidence(audio, runtime_model=self.model, config=self.config)
                if not windows:
                    raise ValueError('Original XLS-R attention evidence is unavailable.')
                evidence.extend(replace(window, start_seconds=window.start_seconds + start / SAMPLE_RATE,
                                        end_seconds=window.end_seconds + start / SAMPLE_RATE) for window in windows)
                branches.append(branch)
            else:
                branch = self.predictor(audio, self.model, self.config)
            probabilities.append(branch.probabilities.spoof)
        p = float(max(probabilities))
        if not attention:
            return p
        # Only the real SSL branch's fields are required by the temporal service.
        inference = SimpleNamespace(temporal_evidence=tuple(evidence), processed_audio=self.processed(waveform),
                                    prediction=SimpleNamespace(provenance=None, branches=[branches[0]]))
        result = self.temporal.analyze(inference)
        return p, result


def save_plot(result, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    curves = result['curves'].groupby(['method', 'step'], as_index=False)[['masked_fraction', 'score_drop']].mean()
    names = {'attention_morf': 'Attention MoRF', 'random': 'Random mean'}
    colors = {'attention_morf': '#275dad', 'random': '#888888'}
    for method in ('attention_morf', 'random'):
        part = curves[curves['method'] == method]
        axes[0].plot(100 * part['masked_fraction'], part['score_drop'], marker='o',
                     label=names[method], color=colors[method])
    axes[0].set(xlabel='Audio samples silenced (%)', ylabel='Fixed-class probability drop')
    axes[0].legend()
    summary = result['summary'].set_index('method').loc[['attention_morf', 'random']]
    axes[1].bar(['Attention MoRF', 'Random mean'], summary['aopc'], color=[colors[m] for m in summary.index])
    axes[1].set(ylabel='AOPC (five drops / six)')
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.6)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / 'faithfulness_aopc.png', dpi=220)
    plt.close(fig)


def run(args):
    import pandas as pd
    import soundfile as sf

    audio = args.audio.resolve(strict=True)
    audio_hash = sha256(audio.read_bytes()).hexdigest()
    output = (args.output_dir or HERE / 'results' / 'temporal_aopc' /
              f'{audio.stem}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}').resolve()
    if output.exists():
        raise ValueError(f'Result directory already exists; choose a new directory: {output}')
    waveform, sample_rate = sf.read(audio, dtype='float32', always_2d=True)
    waveform = np.nan_to_num(waveform.mean(axis=1), nan=0., posinf=0., neginf=0.)
    if sample_rate != SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < 400 or not np.any(waveform):
        raise ValueError('Audio must contain at least 25 ms of non-silent samples.')
    print(f'Audio: {audio.name} | {len(waveform) / SAMPLE_RATE:.3f} seconds', flush=True)
    started = perf_counter()
    runtime = TemporalRuntime(args)
    print('Extracting ORIGINAL continuous attention...', flush=True)
    original_p, temporal = runtime.predict(waveform, attention=True)
    # Check that enabling attention did not change the predictor being evaluated.
    probability_only = runtime.predict(waveform)
    np.testing.assert_allclose(original_p, probability_only, atol=1e-5, rtol=0)
    times, density = temporal.visualization.time_seconds, temporal.visualization.attention_scores
    print(f'Original P(spoof): {original_p:.6f}; evaluating 5 attention masks + {args.random_repeats * 5} random masks...', flush=True)
    result = evaluate(waveform, times, density, original_p, runtime.predict,
                      random_repeats=args.random_repeats, seed=args.seed, audio_hash=audio_hash)
    if sha256(audio.read_bytes()).hexdigest() != audio_hash:
        raise ValueError('Audio changed during evaluation; retry with an unchanged recording.')
    values = result['summary'].set_index('method')['aopc']
    advantage = float(values['attention_morf'] - values['random'])
    note = ('Uses the current backend SSL checkpoint and six-second front-end padding. '
            'The notebook checkpoint/padding can differ; numeric results are not interchangeable.')
    report = (f'TEMPORAL AOPC - SINGLE AUDIO\nAudio: {audio.name}\n'
              f'Original P(spoof): {original_p:.6f}\nFixed original target: {result["target"]}\n'
              '100-ms blocks; cumulative sample budgets: 10%, 20%, 30%, 40%, 50%.\n'
              f'Length-matched random orders: {args.random_repeats}; seed: {args.seed}\n'
              'AOPC = sum(original target probability - masked target probability) / 6.\n\n'
              f'Attention MoRF AOPC: {values["attention_morf"]:.6f}\n'
              f'Random mean AOPC: {values["random"]:.6f}\nAttention-minus-random: {advantage:.6f}\n'
              + ('Attention beat the random mean for this audio.\n' if advantage > 0 else
                 'Attention did not beat the random mean for this audio; retain this result.\n')
              + 'Class-neutral attention can produce negative drops. Single-audio evidence; no population CI.\n'
              'Full continuous ranking is evaluated, including outside p90 highlights.\n'
              'This measures XLS-R agent ranking under silence replacement, not ensemble faithfulness or causality.\n'
              + note + f'\n\nResults: {output}\n')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('summary', 'curves', 'trials', 'blocks'):
        result[name].assign(audio_id=audio.stem).to_csv(output / f'{name}.csv', index=False)
    pd.DataFrame({'time_seconds': times, 'original_attention_density': density}).to_csv(output / 'original_attention.csv', index=False)
    (output / 'mask_geometry.json').write_text(json.dumps(result['geometry'], indent=2), encoding='utf-8')
    save_plot(result, output, f'{audio.name} | fixed target: {result["target"]}')
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'audio_path': str(audio), 'audio_sha256': audio_hash,
        'sample_rate': SAMPLE_RATE, 'n_samples': len(waveform), 'original_p_spoof': original_p,
        'fixed_target': result['target'], 'block_seconds': .1, 'budget_fractions': BUDGETS,
        'random_repeats': args.random_repeats, 'seed': args.seed, 'device': runtime.model.device,
        'threads': args.threads, 'aopc_denominator': 6, 'attention_minus_random': advantage,
        'random_aopc_repeat_sd': float(result['trials'].query("method == 'random'")['aopc'].std()),
        'checkpoint_path': str(runtime.config.checkpoint.safe_path), 'checkpoint_sha256': runtime.config.checkpoint.sha256,
        'backbone': runtime.model.xlsr_model_name,
        'backbone_commit': getattr(runtime.model.module.xlsr.config, '_commit_hash', None),
        'front_end': runtime.model.preprocessing, 'model_verification': runtime.model.verification,
        'temporal_method_version': temporal.explanation.method_version,
        'temporal_research_blockers': temporal.explanation.research_blockers,
        'window_seconds': 6., 'stride_seconds': 3., 'global_hop_seconds': .02,
        'ranking': 'original continuous mean block attention; ties earlier first; fixed once',
        'random_control': 'nested orders matching block length at every position; partial final-block prefix',
        'masking': 'silence raw decoded samples; unchanged production per-window normalization afterwards',
        'score': 'max window P(spoof), oriented to original predicted class at threshold 0.5',
        'notebook_compatibility_note': note, 'population_ci': None,
        'software': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'torch', 'transformers', 'matplotlib')},
        'total_seconds': perf_counter() - started,
    }
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'README_results.txt').write_text(report, encoding='utf-8')
    runtime.model.close()
    print('\n' + report, flush=True)
    return output


def run_self_tests():
    """Synthetic regression checks live here, alongside the actual evaluator."""
    import unittest

    class TemporalAopcChecks(unittest.TestCase):
        def test_blocks_rank_by_original_mean_density(self):
            _, _, means, order = ranked_blocks(np.array([.05, .15, .25]), np.array([1., 3., 2.]), 30, 100)
            np.testing.assert_allclose(means, [1, 3, 2])
            np.testing.assert_array_equal(order, [1, 2, 0])

        def test_exact_nested_masks_and_length_matched_random_tail(self):
            times = (np.arange(27) + .5) / 100
            starts, ends, _, ranked = ranked_blocks(times, np.ones(27), 27, 100, .1)
            np.testing.assert_array_equal(ranked, [0, 1, 2])  # ties: earlier first
            other = matched_random_order(starts, ends, ranked, 42)
            np.testing.assert_array_equal(ends[ranked] - starts[ranked], ends[other] - starts[other])
            np.testing.assert_array_equal(other, matched_random_order(starts, ends, ranked, 42))
            for order in (ranked, other):
                previous = set()
                for budget in (3, 5, 8, 11, 14):
                    intervals = intervals_for_budget(starts, ends, order, budget)
                    chosen = {sample for a, b in intervals for sample in range(a, b)}
                    self.assertEqual(len(chosen), budget)
                    self.assertTrue(previous.issubset(chosen))
                    previous = chosen
                    masked = silence(np.ones(27), intervals)
                    self.assertEqual(len(masked), 27)
                    self.assertEqual(np.count_nonzero(masked == 0), budget)
            with self.assertRaises(ValueError):
                intervals_for_budget(starts, ends, ranked, 28)

        def test_known_aopc_keeps_zero_step_and_negative_drops(self):
            self.assertAlmostEqual(aopc(.9, [.8, .7, .6, .5, .4]), .25)
            self.assertAlmostEqual(aopc(.5, [.5] * 5), 0)
            self.assertLess(aopc(.2, [.3] * 5), 0)
            self.assertAlmostEqual(target_score(.8, 0), .2)
            with self.assertRaises(ValueError):
                target_score(float('nan'), 1)

        def test_full_clip_windows_cover_tail(self):
            self.assertEqual(window_starts(100, 100, 50), [0])
            self.assertEqual(window_starts(230, 100, 50), [0, 50, 100, 130])

        def test_evaluation_preserves_target_after_flip_and_averages_random(self):
            waveform = np.ones(1000, dtype=np.float32)
            times = np.arange(.005, 1000 / SAMPLE_RATE, .01)
            # All samples occupy one 100-ms block, so both methods have known equal curves.
            result = evaluate(waveform, times, np.ones_like(times), .2,
                              lambda masked: .2 + .8 * np.count_nonzero(masked == 0) / len(masked),
                              random_repeats=2, seed=42, audio_hash='fixture', progress=lambda *a, **k: None)
            self.assertEqual(result['target'], 'BONAFIDE')
            self.assertTrue((result['curves'].query('step == 5')['p_spoof'] > .5).all())
            np.testing.assert_allclose(result['summary']['aopc'], .2)
            self.assertEqual(len(result['curves']), 18)
            random = result['trials'].query("method == 'random'")['aopc']
            self.assertAlmostEqual(result['summary'].set_index('method').loc['random', 'aopc'], random.mean())
            self.assertTrue(np.all(waveform == 1))

    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TemporalAopcChecks)).wasSuccessful()


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.self_test:
        command.exit(0 if run_self_tests() else 1)
    if args.random_repeats < 2 or args.threads < 1 or args.seed < 0:
        command.error('--random-repeats must be >=2, --threads positive, --seed nonnegative.')
    try:
        run(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        command.exit(1, f'Temporal AOPC failed: {error}\n')


if __name__ == '__main__':
    main()
