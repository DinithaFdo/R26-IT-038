"""Validate one recording's semantic SHAP ranking with cumulative median replacement.

From audio-service:
  .venv/Scripts/python -B -m app.voice_xai.benchmarks.benchmark_semantic_aopc
  .venv/Scripts/python -B -m app.voice_xai.benchmarks.benchmark_semantic_aopc --self-test

This follows Notebook 04's feature-space protocol, using the production extractor,
frozen model and imputer. It measures surrogate faithfulness, not audio deletion,
teacher-model faithfulness or acoustic causality. No classifier pass is needed.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE.parent / 'semantic' / 'artifacts'
METHODS = ('shap_morf', 'random', 'shap_lorf')
METRICS = ('aopc_raw', 'aopc_raw_removed_only', 'raw_drop_at_k',
           'aopc_probability', 'aopc_probability_removed_only', 'probability_drop_at_k')


def parser():
    command = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    command.add_argument('--audio', type=Path, help='One audio file; default: first audio/*.wav by filename.')
    command.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    command.add_argument('--calibrator', type=Path,
                         help='Optional matching Notebook 02 calibrator; otherwise look in --model-dir.')
    command.add_argument('--max-features', type=int, default=20)
    command.add_argument('--checkpoints', type=int, nargs='+',
                         help='Default: 5, 10, 20 up to max-features, plus max-features.')
    command.add_argument('--random-repeats', type=int, default=30)
    command.add_argument('--seed', type=int, default=42)
    command.add_argument('--threads', type=int, default=2)
    command.add_argument('--output-dir', type=Path,
                         help='New results directory; default: benchmarks/results/semantic_aopc/<audio>_<UTC>.')
    command.add_argument('--self-test', action='store_true',
                         help='Run the numerical and PNG-output checks built into this file, without audio extraction.')
    return command


def cumulative_feature_masks(x, order, replacement, k):
    """Keep step zero unchanged, then replace each prefix with frozen medians."""
    x = np.asarray(x, dtype=np.float64)
    replacement = np.asarray(replacement, dtype=np.float64)
    order = np.asarray(order)
    if (x.ndim != 1 or replacement.shape != x.shape or not np.isfinite(x).all()
            or not np.isfinite(replacement).all()):
        raise ValueError('Input and baseline must be equal-length finite vectors.')
    if (order.ndim != 1 or not np.issubdtype(order.dtype, np.integer)
            or k < 1 or k > len(order) or len(set(order[:k].tolist())) != k
            or np.any(order[:k] < 0) or np.any(order[:k] >= len(x))):
        raise ValueError('Need K distinct valid integer feature indices.')
    rows = np.repeat(x[None, :], k + 1, axis=0)
    for step in range(1, k + 1):
        chosen = order[:step]
        rows[step, chosen] = replacement[chosen]
    return rows


def fixed_class_score(spoof_scores, target_class):
    scores = np.asarray(spoof_scores, dtype=np.float64)
    return scores if int(target_class) == 1 else 1.0 - scores


def aopc_metrics(scores, k):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or k < 1 or len(scores) < k + 1 or not np.isfinite(scores).all():
        raise ValueError('Need the original score and K finite perturbed scores.')
    drops = scores[0] - scores[:k + 1]
    return {'paper': float(drops.sum() / (k + 1)),
            'removed_only': float(drops[1:].mean()), 'drop_at_k': float(drops[k])}


def evaluate_orders(x, shap_values, replacement, predict_raw, probability, *,
                    threshold, k, checkpoints, random_repeats, seed):
    """Fix target/ranking once; evaluate MoRF, reverse ranking and random orders."""
    import pandas as pd

    x = np.asarray(x, dtype=np.float64)
    shap_values = np.asarray(shap_values, dtype=np.float64)
    if shap_values.shape != x.shape or not np.isfinite(shap_values).all():
        raise ValueError('Need one finite signed SHAP value per input feature.')
    if (random_repeats < 1 or seed < 0 or not 0 < threshold < 1
            or not checkpoints or any(point < 1 or point > k for point in checkpoints)):
        raise ValueError('Invalid evaluation configuration.')
    original_raw = float(predict_raw(x[None, :])[0])
    target_class = int(original_raw >= threshold)
    oriented = shap_values * (1 if target_class else -1)
    morf = np.argsort(-oriented, kind='stable')
    # Notebook 04 uses sample index zero for a single sample.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 901]))
    orders = [morf, morf[::-1]] + [rng.permutation(len(x)) for _ in range(random_repeats)]
    descriptions = [('shap_morf', 0), ('shap_lorf', 0)] + [('random', r) for r in range(random_repeats)]
    blocks = [cumulative_feature_masks(x, order, replacement, k) for order in orders]
    raw = np.asarray(predict_raw(np.concatenate(blocks)), dtype=np.float64).reshape(len(orders), k + 1)
    prob = np.asarray(probability(raw.reshape(-1)), dtype=np.float64).reshape(raw.shape)
    if not np.isfinite(raw).all() or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError('Predictions must be finite and probabilities must be in [0, 1].')
    np.testing.assert_allclose(raw[:, 0], original_raw, atol=1e-6, rtol=0)
    raw_q, prob_q = fixed_class_score(raw, target_class), fixed_class_score(prob, target_class)
    curves, trials = [], []
    for t, (method, repeat) in enumerate(descriptions):
        for step in range(k + 1):
            curves.append({'method': method, 'repeat': repeat, 'step': step,
                           'masked_fraction': step / len(x),
                           'feature_index': int(orders[t][step - 1]) if step else None,
                           'raw_spoof_score': float(raw[t, step]),
                           'deployed_spoof_probability': float(prob[t, step]),
                           'fixed_class_raw_score': float(raw_q[t, step]),
                           'fixed_class_probability': float(prob_q[t, step]),
                           'raw_drop': float(raw_q[t, 0] - raw_q[t, step]),
                           'probability_drop': float(prob_q[t, 0] - prob_q[t, step])})
        for point in checkpoints:
            r, p = aopc_metrics(raw_q[t], point), aopc_metrics(prob_q[t], point)
            trials.append({'method': method, 'repeat': repeat, 'k': point,
                           'aopc_raw': r['paper'], 'aopc_raw_removed_only': r['removed_only'],
                           'raw_drop_at_k': r['drop_at_k'], 'aopc_probability': p['paper'],
                           'aopc_probability_removed_only': p['removed_only'],
                           'probability_drop_at_k': p['drop_at_k']})
    trials = pd.DataFrame(trials)
    summary = trials.groupby(['method', 'k'], as_index=False)[list(METRICS)].mean()
    return {'original_raw': original_raw, 'original_probability': float(prob[0, 0]),
            'target_class': target_class, 'oriented_shap': oriented, 'morf_order': morf,
            'curves': pd.DataFrame(curves), 'trials': trials, 'summary': summary}


def load_probability(artifacts, explicit_path):
    """Use calibration only when its artifact exists; label clipping explicitly."""
    path = explicit_path or artifacts.directory / 'probability_calibrator.joblib'
    selected = artifacts.metadata.get('calibration', {}).get('selected')
    note = None
    calibrator = None
    fingerprint = None
    if explicit_path is not None or path.is_file():
        import joblib
        path = path.resolve(strict=True)
        fingerprint = sha256(path.read_bytes()).hexdigest()
        calibrator = joblib.load(path)
        if selected is not None and bool(selected) != (calibrator is not None):
            raise ValueError('Calibration metadata and saved calibrator disagree; use the matching export.')
        if calibrator is not None and not callable(getattr(calibrator, 'predict', None)):
            raise ValueError('The calibrator must expose predict(raw_scores).')
    elif selected:
        note = ('Training metadata selects calibration, but probability_calibrator.joblib is missing. '
                'Raw AOPC is available. Secondary probability AOPC uses clipped raw scores, '
                'matching the current backend, rather than the notebook calibrator.')

    def probability(raw):
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        values = np.asarray(calibrator.predict(raw) if calibrator is not None else raw, dtype=np.float64)
        if values.shape != raw.shape or not np.isfinite(values).all():
            raise ValueError('Calibration returned invalid probabilities.')
        return np.clip(values, 0, 1)

    info = {'mode': 'calibrated' if calibrator is not None else 'clipped_raw',
            'calibrator_path': str(path) if fingerprint else None,
            'calibrator_sha256': fingerprint, 'note': note}
    return probability, info


def save_plots(result, output, title, probability_mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'shap_morf': '#275dad', 'random': '#888888', 'shap_lorf': '#cb6740'}
    names = {'shap_morf': 'SHAP MoRF', 'random': 'Random mean (30 orders)', 'shap_lorf': 'SHAP LoRF'}
    repeats = result['trials'].query("method == 'random'")['repeat'].nunique()
    names['random'] = f'Random mean ({repeats} orders)'
    mean_curves = result['curves'].groupby(['method', 'step'], as_index=False)[['raw_drop', 'probability_drop']].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, subtitle in zip(axes, ['raw_drop', 'probability_drop'],
                                   ['Raw score — primary', f'Probability ({probability_mode}) — secondary']):
        for method in METHODS:
            part = mean_curves[mean_curves['method'] == method]
            ax.plot(part['step'], part[metric], color=colors[method], label=names[method])
        ax.axhline(0, color='black', linewidth=.6)
        ax.set(xlabel='Cumulatively replaced features', ylabel='Fixed-class score drop', title=subtitle)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / 'perturbation_drop_curves.png', dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        part = result['summary'][result['summary']['method'] == method].sort_values('k')
        ax.plot(part['k'], part['aopc_raw'], marker='o', color=colors[method], label=names[method])
    ax.axhline(0, color='black', linewidth=.6)
    ax.set(xlabel='K features replaced', ylabel='Raw AOPC (K+1 denominator)', title=title)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'aopc_checkpoints.png', dpi=220)
    plt.close(fig)


def run_self_tests():
    """Keep protocol regression checks in this file; pytest is not required."""
    from contextlib import redirect_stderr
    from io import StringIO
    import tempfile
    from types import SimpleNamespace
    import unittest
    import pandas as pd

    def linear_result(**overrides):
        options = dict(x=np.array([3., 2., 1.]), shap_values=np.array([.3, .1, .02]),
                       replacement=np.zeros(3), predict_raw=lambda rows: rows @ [.1, .05, .02],
                       probability=lambda raw: np.clip(raw, 0, 1), threshold=.3, k=3,
                       checkpoints=[1, 2, 3], random_repeats=30, seed=42)
        options.update(overrides)
        return evaluate_orders(**options)

    class SemanticAopcChecks(unittest.TestCase):
        def test_known_linear_aopc_keeps_target_after_class_flip(self):
            result = linear_result()
            curve = result['curves'].query("method == 'shap_morf'")
            np.testing.assert_allclose(curve['raw_spoof_score'], [.42, .12, .02, 0])
            np.testing.assert_allclose(curve['raw_drop'], [0, .30, .40, .42])
            self.assertEqual(result['target_class'], 1)
            summary = result['summary'].query("method == 'shap_morf' and k == 3").iloc[0]
            self.assertAlmostEqual(summary['aopc_raw'], 1.12 / 4)
            self.assertAlmostEqual(summary['aopc_raw_removed_only'], 1.12 / 3)
            self.assertAlmostEqual(summary['raw_drop_at_k'], .42)

        def test_bonafide_signed_ranking_raw_scale_and_negative_drops(self):
            # The largest absolute contribution supports spoof: remove it last.
            result = linear_result(x=np.ones(3), shap_values=np.array([.6, -.1, -.08]),
                                   predict_raw=lambda rows: -.5 + rows @ [.6, -.1, -.08], threshold=.5)
            np.testing.assert_array_equal(result['morf_order'], [1, 2, 0])
            curve = result['curves'].query("method == 'shap_morf'")
            self.assertEqual(result['target_class'], 0)
            self.assertAlmostEqual(result['original_raw'], -.08)
            np.testing.assert_allclose(curve['raw_drop'], [0, .1, .18, -.42])
            self.assertAlmostEqual(curve['fixed_class_raw_score'].iloc[0], 1.08)
            self.assertAlmostEqual(curve['fixed_class_probability'].iloc[0], 1)
            self.assertLess(aopc_metrics(curve['fixed_class_raw_score'], 3)['paper'], 0)

        def test_cumulative_medians_preserve_input(self):
            x, baseline = np.array([3., 2., 1.]), np.array([10., 20., 30.])
            masked = cumulative_feature_masks(x, [1, 0, 2], baseline, 3)
            np.testing.assert_array_equal(masked, [[3, 2, 1], [3, 20, 1], [10, 20, 1], [10, 20, 30]])
            np.testing.assert_array_equal(x, [3, 2, 1])
            np.testing.assert_array_equal(baseline, [10, 20, 30])
            with self.assertRaisesRegex(ValueError, 'distinct'):
                cumulative_feature_masks(x, [1, 1], baseline, 2)

        def test_zero_baseline_change_random_averaging_and_reproducibility(self):
            unchanged = linear_result(replacement=np.array([3., 2., 1.]))
            np.testing.assert_allclose(unchanged['summary'][list(METRICS)], 0)
            result = linear_result()
            trials = result['trials'].query("method == 'random' and k == 3")
            self.assertEqual(len(trials), 30)
            random_mean = result['summary'].query("method == 'random' and k == 3")['aopc_raw'].iloc[0]
            self.assertAlmostEqual(random_mean, trials['aopc_raw'].mean())
            pd.testing.assert_frame_equal(result['curves'], linear_result()['curves'])
            np.testing.assert_allclose(result['curves'].query('step == 3')['raw_spoof_score'], 0)
            final = result['summary'].query('k == 3').set_index('method')['aopc_raw']
            self.assertGreater(final['shap_morf'], final['random'])
            self.assertGreater(final['random'], final['shap_lorf'])

        def test_missing_selected_calibrator_is_explicitly_clipped_raw(self):
            with tempfile.TemporaryDirectory(dir=HERE / 'results', prefix='aopc-self-test-') as folder:
                artifacts = SimpleNamespace(directory=Path(folder), metadata={'calibration': {'selected': True}})
                probability, info = load_probability(artifacts, None)
                np.testing.assert_allclose(probability([-.2, .4, 1.2]), [0, .4, 1])
                self.assertEqual(info['mode'], 'clipped_raw')
                self.assertIn('missing', info['note'])
                self.assertIsNone(info['calibrator_sha256'])
                with self.assertRaises(FileNotFoundError):
                    load_probability(artifacts, Path(folder) / 'explicit_missing.joblib')

        def test_matching_calibrator_changes_only_probability_metrics(self):
            import joblib
            from sklearn.isotonic import IsotonicRegression

            with tempfile.TemporaryDirectory(dir=HERE / 'results', prefix='aopc-self-test-') as folder:
                calibrator = IsotonicRegression(out_of_bounds='clip').fit([0., .5, 1.], [0., .8, 1.])
                joblib.dump(calibrator, Path(folder) / 'probability_calibrator.joblib')
                artifacts = SimpleNamespace(directory=Path(folder), metadata={'calibration': {'selected': True}})
                probability, info = load_probability(artifacts, None)
                self.assertEqual(info['mode'], 'calibrated')
                self.assertEqual(len(info['calibrator_sha256']), 64)
                calibrated, raw = linear_result(probability=probability), linear_result()
                np.testing.assert_allclose(calibrated['summary']['aopc_raw'], raw['summary']['aopc_raw'])
                self.assertFalse(np.allclose(calibrated['summary']['aopc_probability'], raw['summary']['aopc_probability']))

        def test_invalid_cli_stops_before_extraction(self):
            for arguments in (['--max-features', '0'], ['--max-features', '166'], ['--random-repeats', '0'],
                              ['--threads', '0'], ['--seed', '-1'],
                              ['--max-features', '5', '--checkpoints', '10'], ['--audio', 'one.wav', 'two.wav']):
                with self.subTest(arguments=arguments), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                    main(arguments)
                self.assertEqual(error.exception.code, 2)

        def test_plots_create_only_two_png_files(self):
            with tempfile.TemporaryDirectory(dir=HERE / 'results', prefix='aopc-self-test-') as folder:
                output = Path(folder)
                save_plots(linear_result(), output, 'AOPC self-test', 'clipped_raw')
                self.assertEqual({path.name for path in output.iterdir()},
                                 {'perturbation_drop_curves.png', 'aopc_checkpoints.png'})
                for path in output.iterdir():
                    self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SemanticAopcChecks)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def run(args):
    import pandas as pd
    import xgboost as xgb
    from app.voice_xai.semantic.artifact_loader import load_production_semantic_artifacts
    from app.voice_xai.semantic.feature_extraction import AcousticFeatureExtractor
    from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION
    from app.voice_xai.semantic.service import XgboostShapEvidenceProvider

    audio = args.audio
    if audio is None:
        audio = next(iter(sorted((HERE / 'audio').glob('*.wav'))), None)
        if audio is None:
            raise ValueError('No bundled audio/*.wav files found; specify --audio.')
    audio = audio.resolve(strict=True)
    if not audio.is_file():
        raise ValueError('--audio must name one recording.')
    output = args.output_dir or HERE / 'results' / 'semantic_aopc' / (
        f'{audio.stem}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}')
    output = output.resolve()
    if output.exists():
        raise ValueError(f'Result directory already exists; choose a new --output-dir: {output}')
    started = perf_counter()
    audio_hash = sha256(audio.read_bytes()).hexdigest()
    print(f'Audio: {audio}', flush=True)
    print(f'Loading semantic artifacts: {args.model_dir.resolve()}', flush=True)
    artifacts = load_production_semantic_artifacts(args.model_dir)
    artifacts.booster.set_param({'device': 'cpu', 'nthread': args.threads})
    probability, probability_info = load_probability(artifacts, args.calibrator)
    if probability_info['note']:
        print(f"NOTE: {probability_info['note']}", flush=True)
    print('Extracting the full-clip 165 acoustic features once...', flush=True)
    extract_started = perf_counter()
    features = AcousticFeatureExtractor().extract_file(audio)
    extraction_seconds = perf_counter() - extract_started
    evidence = XgboostShapEvidenceProvider(artifacts).extract_features(features)
    if sha256(audio.read_bytes()).hexdigest() != audio_hash:
        raise ValueError('Audio changed during extraction; retry with an unchanged recording.')

    def predict_raw(values):
        matrix = xgb.DMatrix(values, feature_names=list(FEATURE_NAMES))
        return artifacts.booster.predict(matrix, output_margin=True)

    print(f'Comparing SHAP MoRF, SHAP LoRF and {args.random_repeats} random orders...', flush=True)
    evaluation_started = perf_counter()
    result = evaluate_orders(evidence.feature_values, evidence.shap_values,
                             artifacts.imputer.statistics_, predict_raw, probability,
                             threshold=artifacts.threshold, k=args.max_features,
                             checkpoints=args.checkpoints, random_repeats=args.random_repeats,
                             seed=args.seed)
    evaluation_seconds = perf_counter() - evaluation_started
    np.testing.assert_allclose(result['original_raw'], evidence.predicted_value, atol=1e-6, rtol=0)
    target = 'SPOOF' if result['target_class'] else 'BONAFIDE'
    selected = result['summary'][result['summary']['k'] == args.max_features].set_index('method')
    advantage = float(selected.loc['shap_morf', 'aopc_raw'] - selected.loc['random', 'aopc_raw'])
    positive_count = int((result['oriented_shap'] > 1e-12).sum())
    report = (
        f'SEMANTIC SHAP AOPC — SINGLE AUDIO\nAudio: {audio.name}\n'
        f'Fixed original raw target: {target}\nRaw spoof score: {result["original_raw"]:.6f}\n'
        f'Probability ({probability_info["mode"]}): {result["original_probability"]:.6f}\n'
        f'K: {args.max_features}/{len(FEATURE_NAMES)}; random orders: {args.random_repeats}; seed: {args.seed}\n'
        'Baseline: saved fitted imputer medians; no refitting.\n'
        'AOPC = sum(drop[0:K+1]) / (K+1), including unchanged step 0.\n\n'
        + result['summary'][['method', 'k', 'aopc_raw', 'aopc_probability']].to_string(index=False, float_format=lambda v: f'{v:.6f}')
        + f'\n\nAt K={args.max_features}: SHAP-minus-random raw AOPC = {advantage:.6f}\n'
        + ('SHAP beat the random mean for this audio under this protocol.\n' if advantage > 0 else
           'SHAP did not beat the random mean for this audio under this protocol.\n')
        + f'Positive-support features: {positive_count}.\n'
        + ('Later replacements include neutral/opposing evidence at this K.\n' if positive_count < args.max_features else '')
        + 'Single-audio evidence only; no population bootstrap interval or universal pass threshold.\n'
        'This measures surrogate feature-space faithfulness, not teacher or waveform faithfulness.\n'
        'Median replacement can break feature correlations; negative drops are retained.\n'
        + (f'NOTE: {probability_info["note"]}\n' if probability_info['note'] else '')
        + f'\nResults: {output}\n')
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [('aopc_per_sample.csv', result['summary']),
                        ('aopc_per_random_repeat.csv', result['trials'])]:
        frame = frame.assign(analysis_id=audio.stem, fixed_raw_target_class=target,
                             probability_mode=probability_info['mode'])
        frame.to_csv(output / name, index=False)
    curves = result['curves'].copy()
    curves['newly_replaced_feature'] = curves.pop('feature_index').map(
        lambda j: FEATURE_NAMES[int(j)] if pd.notna(j) else '')
    curves.assign(analysis_id=audio.stem).to_csv(output / 'all_perturbation_curves.csv', index=False)
    order = result['morf_order']
    pd.DataFrame({'rank': np.arange(1, len(order) + 1),
                  'feature': np.asarray(FEATURE_NAMES)[order],
                  'raw_shap': evidence.shap_values[order], 'fixed_class_shap': result['oriented_shap'][order],
                  'original_feature_value': evidence.feature_values[order],
                  'replacement_value': artifacts.imputer.statistics_[order]}).to_csv(output / 'feature_ranking.csv', index=False)
    pd.DataFrame({'feature': FEATURE_NAMES, 'measured_value': features.values,
                  'imputed_value': evidence.feature_values,
                  'median_baseline': artifacts.imputer.statistics_}).to_csv(output / 'input_features.csv', index=False)
    random_variability = result['trials'].query("method == 'random'").groupby('k')[
        ['aopc_raw', 'aopc_probability']].agg(['mean', 'std'])
    random_variability.columns = ['_'.join(column) for column in random_variability.columns]
    random_variability.to_csv(output / 'random_repeat_variability.csv')
    save_plots(result, output, f'{audio.name} | fixed target: {target}', probability_info['mode'])
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'audio_path': str(audio),
        'audio_sha256': audio_hash, 'cleaned_duration_seconds': features.duration_seconds,
        'analysis_scope': 'single_full_clip', 'model_directory': str(artifacts.directory),
        'model_version': artifacts.model_version, 'model_artifact_fingerprint': artifacts.fingerprint,
        'extractor_version': FEATURE_EXTRACTION_VERSION, 'feature_names': list(FEATURE_NAMES),
        'imputed_feature_count': evidence.imputed_feature_count,
        'original_raw_spoof_score': result['original_raw'], 'fixed_raw_target_class': target,
        'threshold': artifacts.threshold, 'original_probability': result['original_probability'],
        'raw_vs_probability_class_disagreement': bool(result['target_class'] != int(result['original_probability'] >= artifacts.threshold)),
        'probability': probability_info, 'max_features': args.max_features, 'checkpoints': args.checkpoints,
        'random_repeats': args.random_repeats, 'seed': args.seed, 'threads': args.threads,
        'ranking': 'descending signed SHAP oriented to original raw predicted class; fixed once',
        'replacement': 'frozen fitted median-imputer statistics', 'positive_support_feature_count': positive_count,
        'main_aopc_convention': 'sum(drop[0:K+1])/(K+1); drop[0]=0',
        'additional_aopc_convention': 'sum(drop[1:K+1])/K; *_removed_only',
        'shap_additivity_error': abs(evidence.base_value + float(evidence.shap_values.sum()) - evidence.predicted_value),
        'shap_minus_random_raw_aopc': advantage, 'population_ci': None,
        'surrogate_test_fidelity': artifacts.fidelity(),
        'timing_seconds': {'feature_extraction': extraction_seconds, 'evaluation': evaluation_seconds,
                           'total_including_plots': perf_counter() - started},
        'software': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'xgboost', 'shap', 'scikit-learn', 'matplotlib')},
        'scope': 'surrogate feature-space faithfulness only; original teacher not perturbed; no audio editing',
    }
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'README_results.txt').write_text(report, encoding='utf-8')
    print('\n' + report, flush=True)
    return output


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.self_test:
        command.exit(0 if run_self_tests() else 1)
    if not 1 <= args.max_features <= 165:
        command.error('--max-features must be between 1 and 165.')
    if args.random_repeats < 1 or args.threads < 1 or args.seed < 0:
        command.error('--random-repeats and --threads must be positive; --seed must be nonnegative.')
    if args.checkpoints is None:
        args.checkpoints = [k for k in (5, 10, 20) if k <= args.max_features]
    if any(k < 1 or k > args.max_features for k in args.checkpoints):
        command.error('Every --checkpoints value must be between 1 and --max-features.')
    args.checkpoints = sorted(set(args.checkpoints + [args.max_features]))
    try:
        run(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        command.exit(1, f'AOPC validation failed: {error}\n')


if __name__ == '__main__':
    main()
