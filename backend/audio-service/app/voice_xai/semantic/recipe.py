"""Signal processing functions for acoustic and selected Glottal features.

Decode and clean the full clip once, then use that same waveform for all 148
base measurements and 20 selected Glottal measurements. Merge these groups into
165 unique model inputs while retaining shared base values. Load and cache the
DisVoice Glottal implementation with its required compatibility patches.
"""
from pathlib import Path
import importlib.util
import os
import re
import tempfile
import numpy as np
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from scipy.fft import dct, rfft, irfft
from scipy.signal import get_window
from scipy.stats import linregress
from functools import lru_cache
from app.models.real.glottal_compat import (
    apply_disvoice_compatibility_patches,
    patch_disvoice_gci_numpy2_bug,
)
from .schema import FEATURE_NAMES, PREPROCESSING_CONTRACT, SELECTED20
TARGET_SR = 16000


@lru_cache(maxsize=1)
def load_glottal_class():
    """Load the installed Glottal module without unrelated package extractors."""
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


def safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def prepare_audio(y, sr):
    """Clean a full decoded clip once; never crop or pad the source duration."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.ndim != 1 or y.size == 0 or sr <= 0:
        raise ValueError('Expected non-empty audio and a positive sample rate.')
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type='kaiser_fast')
        sr = TARGET_SR
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - np.mean(y)
    y, _ = librosa.effects.trim(y, top_db=PREPROCESSING_CONTRACT['silence_trim_top_db'])
    if len(y) < int(PREPROCESSING_CONTRACT['minimum_clean_duration_seconds'] * sr):
        raise ValueError('audio too short after preprocessing')
    peak = float(np.max(np.abs(y)))
    if not np.isfinite(peak) or peak <= 1e-8:
        raise ValueError('Audio is silent after preprocessing.')
    y = np.ascontiguousarray(y / peak, dtype=np.float32)
    return y, sr

def load_audio_path(path):
    y, sr = sf.read(str(path), dtype='float32', always_2d=False)
    return prepare_audio(y, sr)

def summarize_track(x, prefix, out):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out[f'{prefix}_mean'] = float(np.mean(x)) if len(x) else np.nan
    out[f'{prefix}_std'] = float(np.std(x)) if len(x) else np.nan

def compute_lfcc(y, sr, n_lfcc=20, n_filters=40, n_fft=512, hop_length=160, win_length=400):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True)) ** 2
    freqs = np.linspace(0.0, sr / 2.0, S.shape[0])
    edges = np.linspace(0.0, sr / 2.0, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)), dtype=np.float64)
    for i in range(n_filters):
        left, center, right = (edges[i], edges[i + 1], edges[i + 2])
        li = (freqs >= left) & (freqs <= center)
        ri = (freqs >= center) & (freqs <= right)
        if center > left:
            fb[i, li] = (freqs[li] - left) / (center - left)
        if right > center:
            fb[i, ri] = (right - freqs[ri]) / (right - center)
    energies = fb @ S
    log_e = np.log(np.maximum(energies, 1e-12))
    coeff = dct(log_e, type=2, axis=0, norm='ortho')
    return coeff[1:n_lfcc + 1]

def compute_framewise_cpp(y, sr, frame_ms=40, hop_ms=10, f0_min=60.0, f0_max=500.0):
    frame_len = int(round(sr * frame_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop).T
    window = get_window('hann', frame_len, fftbins=True)
    q = np.arange(frame_len) / sr
    q_search = (q >= 1.0 / f0_max) & (q <= 1.0 / f0_min)
    q_trend = (q >= 0.001) & (q <= min(0.05, q[-1]))
    qx = q[q_trend]
    if q_search.sum() < 2 or q_trend.sum() < 3:
        return np.array([], dtype=float)
    cpp = []
    for frame in frames:
        if np.sqrt(np.mean(frame ** 2) + 1e-12) < 1e-05:
            continue
        spec = np.abs(rfft(frame * window)) ** 2
        log_power_db = 10.0 * np.log10(np.maximum(spec, 1e-12))
        cep = irfft(log_power_db, n=frame_len)
        cy = cep[q_trend]
        if not np.all(np.isfinite(cy)):
            continue
        slope, intercept = np.polyfit(qx, cy, 1)
        search_idx = np.where(q_search)[0]
        peak_idx = search_idx[np.argmax(cep[search_idx])]
        baseline = slope * q[peak_idx] + intercept
        cpp.append(float(cep[peak_idx] - baseline))
    return np.asarray(cpp, dtype=float)

def praat_voice_features(y, sr):
    out = {'jitter': np.nan, 'shimmer': np.nan, 'hnr': np.nan, 'gci_count': np.nan, 'gci_mean_interval': np.nan, 'gci_std_interval': np.nan, 'gci_irregularity': np.nan, 'gci_jitter_gci': np.nan, 'formant_f1_mean': np.nan, 'formant_f1_std': np.nan, 'formant_f2_mean': np.nan, 'formant_f2_std': np.nan, 'formant_f3_mean': np.nan, 'formant_f3_std': np.nan}
    try:
        sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        pp = call(sound, 'To PointProcess (periodic, cc)', 75.0, 500.0)
        n_points = int(call(pp, 'Get number of points'))
        out['gci_count'] = float(n_points)
        if n_points >= 2:
            pulse_times = np.array([call(pp, 'Get time from index', i) for i in range(1, n_points + 1)], dtype=float)
            intervals = np.diff(pulse_times)
            intervals = intervals[(intervals > 1 / 500) & (intervals < 1 / 50)]
            if len(intervals):
                mu = float(np.mean(intervals))
                sd = float(np.std(intervals))
                out['gci_mean_interval'] = mu
                out['gci_std_interval'] = sd
                out['gci_jitter_gci'] = sd / (mu + 1e-12)
                if len(intervals) > 1:
                    out['gci_irregularity'] = float(np.mean(np.abs(np.diff(intervals))) / (mu + 1e-12))
        out['jitter'] = safe_float(call(pp, 'Get jitter (local)', 0.0, 0.0, 1 / 500, 1 / 75, 1.3))
        out['shimmer'] = safe_float(call([sound, pp], 'Get shimmer (local)', 0.0, 0.0, 1 / 500, 1 / 75, 1.3, 1.6))
        harmonicity = call(sound, 'To Harmonicity (cc)', 0.01, 75.0, 0.1, 1.0)
        out['hnr'] = safe_float(call(harmonicity, 'Get mean', 0.0, 0.0))
        formant = call(sound, 'To Formant (burg)', 0.0, 5.0, 5500.0, 0.025, 50.0)
        for i in (1, 2, 3):
            out[f'formant_f{i}_mean'] = safe_float(call(formant, 'Get mean', i, 0.0, 0.0, 'Hertz'))
            out[f'formant_f{i}_std'] = safe_float(call(formant, 'Get standard deviation', i, 0.0, 0.0, 'Hertz'))
    except Exception:
        pass
    return out

def pause_features_from_rms(rms, hop_length, sr, silence_db=-40.0, min_pause_s=0.15):
    rms = np.asarray(rms, dtype=float).reshape(-1)
    if len(rms) == 0 or np.max(rms) <= 1e-12:
        duration = len(rms) * hop_length / sr
        return {'pause_ratio': 1.0, 'pause_count': 1.0, 'pause_mean_duration': duration, 'pause_max_duration': duration}
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-12), ref=np.max)
    silent = db <= silence_db
    min_frames = max(1, int(round(min_pause_s * sr / hop_length)))
    durations = []
    start = None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        if (not s or i == len(silent) - 1) and start is not None:
            end = i if not s else i + 1
            n = end - start
            if n >= min_frames:
                durations.append(n * hop_length / sr)
            start = None
    return {'pause_ratio': float(np.mean(silent)), 'pause_count': float(len(durations)), 'pause_mean_duration': float(np.mean(durations)) if durations else 0.0, 'pause_max_duration': float(np.max(durations)) if durations else 0.0}

def extract_acoustic_features(y, sr):
    """Measure the full shared waveform without applying preprocessing again."""
    out = {}
    hop = 160
    n_fft = 512
    win = 400
    out.update(praat_voice_features(y, sr))
    lfcc = compute_lfcc(y, sr, n_lfcc=20, n_filters=40, n_fft=n_fft, hop_length=hop, win_length=win)
    for i in range(20):
        out[f'lfcc_{i + 1}'] = float(np.nanmean(lfcc[i])) if i < len(lfcc) else np.nan
    mfcc_all = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14, n_fft=n_fft, hop_length=hop, win_length=win, n_mels=40, fmin=20.0, fmax=min(7600.0, sr / 2 - 1))
    mfcc = mfcc_all[1:14]
    width = 9 if mfcc.shape[1] >= 9 else max(3, mfcc.shape[1] // 2 * 2 + 1)
    if width > mfcc.shape[1]:
        width = mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1
    if width < 3:
        d1 = np.zeros_like(mfcc)
        d2 = np.zeros_like(mfcc)
    else:
        d1 = librosa.feature.delta(mfcc, width=width, order=1, mode='nearest')
        d2 = librosa.feature.delta(mfcc, width=width, order=2, mode='nearest')
    for family, mat in [('mfcc', mfcc), ('dmfcc', d1), ('ddmfcc', d2)]:
        for i in range(13):
            summarize_track(mat[i], f'{family}_{i + 1}', out)
    try:
        f0, voiced_flag, _ = librosa.pyin(y, sr=sr, fmin=50.0, fmax=500.0, frame_length=2048, hop_length=hop)
        voiced = f0[np.isfinite(f0)]
        out['f0_mean'] = float(np.mean(voiced)) if len(voiced) else np.nan
        out['f0_std'] = float(np.std(voiced)) if len(voiced) else np.nan
        out['f0_median'] = float(np.median(voiced)) if len(voiced) else np.nan
        out['f0_range'] = float(np.ptp(voiced)) if len(voiced) else np.nan
        out['voiced_ratio'] = float(np.mean(voiced_flag.astype(float))) if voiced_flag is not None else np.nan
    except Exception:
        out.update({'f0_mean': np.nan, 'f0_std': np.nan, 'f0_median': np.nan, 'f0_range': np.nan, 'voiced_ratio': np.nan})
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    out['rms_mean'] = float(np.mean(rms))
    out['rms_std'] = float(np.std(rms))
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-12), ref=1.0)
    out['rms_dynamic_range_db'] = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    S_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window='hann'))
    tracks = {'spectral_centroid': librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0], 'spectral_bandwidth': librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0], 'spectral_rolloff': librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.85)[0], 'spectral_flatness': librosa.feature.spectral_flatness(S=S_mag)[0]}
    for name, track in tracks.items():
        summarize_track(track, name, out)
    norm_spec = S_mag / (np.linalg.norm(S_mag, axis=0, keepdims=True) + 1e-12)
    flux = np.sqrt(np.sum(np.diff(norm_spec, axis=1) ** 2, axis=0)) if norm_spec.shape[1] > 1 else np.array([0.0])
    summarize_track(flux, 'spectral_flux', out)
    try:
        contrast = librosa.feature.spectral_contrast(S=S_mag, sr=sr, n_bands=5, fmin=200.0)
        for b in range(contrast.shape[0]):
            summarize_track(contrast[b], f'spectral_contrast_band{b + 1}', out)
    except Exception:
        for b in range(6):
            out[f'spectral_contrast_band{b + 1}_mean'] = np.nan
            out[f'spectral_contrast_band{b + 1}_std'] = np.nan
    cpp = compute_framewise_cpp(y, sr)
    out['cpp_mean'] = float(np.mean(cpp)) if len(cpp) else np.nan
    out['cpp_std'] = float(np.std(cpp)) if len(cpp) else np.nan
    out.update(pause_features_from_rms(rms, hop, sr, silence_db=-40.0, min_pause_s=0.15))
    return out


DISVOICE_RAW_NAMES = ['global avg var GCI', 'global avg avg NAQ', 'global avg std NAQ', 'global avg avg QOQ', 'global avg std QOQ', 'global avg avg H1H2', 'global avg std H1H2', 'global avg avg HRF', 'global avg std HRF', 'global std var GCI', 'global std avg NAQ', 'global std std NAQ', 'global std avg QOQ', 'global std std QOQ', 'global std avg H1H2', 'global std std H1H2', 'global std avg HRF', 'global std std HRF', 'global skewness var GCI', 'global skewness avg NAQ', 'global skewness std NAQ', 'global skewness avg QOQ', 'global skewness std QOQ', 'global skewness avg H1H2', 'global skewness std H1H2', 'global skewness avg HRF', 'global skewness std HRF', 'global kurtosis var GCI', 'global kurtosis avg NAQ', 'global kurtosis std NAQ', 'global kurtosis avg QOQ', 'global kurtosis std QOQ', 'global kurtosis avg H1H2', 'global kurtosis std H1H2', 'global kurtosis avg HRF', 'global kurtosis std HRF']

def normalize_disvoice_name(name):
    name = name.lower().strip()
    name = re.sub('\\s+', '_', name)
    name = re.sub('[^a-z0-9_]', '', name)
    return f'glottal_{name}'
DISVOICE_FEATURE_NAMES = [normalize_disvoice_name(name) for name in DISVOICE_RAW_NAMES]
@lru_cache(maxsize=1)
def _get_glottal_extractor():
    return load_glottal_class()()

def extract_glottal_voice_quality(audio, sr=16000):
    features = {}
    feature_names = ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_median', 'voiced_fraction', 'jitter_local', 'shimmer_local', 'hnr_mean', 'hnr_std']
    try:
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
        f0_all = pitch.selected_array['frequency']
        voiced_mask = f0_all > 0
        f0 = f0_all[voiced_mask]
        if len(f0) > 0:
            features['f0_mean'] = float(np.mean(f0))
            features['f0_std'] = float(np.std(f0))
            features['f0_min'] = float(np.min(f0))
            features['f0_max'] = float(np.max(f0))
            features['f0_median'] = float(np.median(f0))
        else:
            features['f0_mean'] = np.nan
            features['f0_std'] = np.nan
            features['f0_min'] = np.nan
            features['f0_max'] = np.nan
            features['f0_median'] = np.nan
        features['voiced_fraction'] = float(np.mean(voiced_mask))
        point_process = parselmouth.praat.call(sound, 'To PointProcess (periodic, cc)', 60, 500)
        features['jitter_local'] = float(parselmouth.praat.call(point_process, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3))
        features['shimmer_local'] = float(parselmouth.praat.call([sound, point_process], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6))
        harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=60, silence_threshold=0.1, periods_per_window=4.5)
        hnr = harmonicity.values.flatten()
        hnr = hnr[np.isfinite(hnr) & (hnr > -100)]
        if len(hnr) > 0:
            features['hnr_mean'] = float(np.mean(hnr))
            features['hnr_std'] = float(np.std(hnr))
        else:
            features['hnr_mean'] = np.nan
            features['hnr_std'] = np.nan
    except Exception:
        for name in feature_names:
            features[name] = np.nan
    return features

def extract_glottal_spectral_features(y, sr=16000):
    features = {}
    try:
        spectrum = np.abs(np.fft.rfft(y)) ** 2
        freqs = np.fft.rfftfreq(len(y), d=1 / sr)
        mask = (freqs >= 100) & (freqs <= 5000)
        freq_selected = freqs[mask]
        power_selected = spectrum[mask]
        power_db = 10 * np.log10(power_selected + 1e-12)
        result = linregress(freq_selected, power_db)
        features['spectral_tilt'] = float(result.slope)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(centroid))
        features['spectral_centroid_std'] = float(np.std(centroid))
    except Exception:
        features['spectral_tilt'] = np.nan
        features['spectral_centroid_mean'] = np.nan
        features['spectral_centroid_std'] = np.nan
    return features

def extract_disvoice_glottal_exact(y, sr=TARGET_SR):
    """Encode the shared waveform as PCM16 only for DisVoice's file interface."""
    temp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as handle:
            temp_wav = handle.name
        sf.write(temp_wav, y, sr, subtype='PCM_16')
        values = _get_glottal_extractor().extract_features_file(temp_wav, static=True, plots=False, fmt='npy')
        values = np.asarray(values, dtype=np.float64).flatten()
        if len(values) != 36:
            raise ValueError(f'Expected 36 DisVoice features, got {len(values)}')
        finite_count = int(np.isfinite(values).sum())
        if finite_count != 36:
            raise ValueError(f'Invalid DisVoice extraction: {finite_count}/36 finite features')
        return values
    finally:
        if temp_wav is not None and os.path.exists(temp_wav):
            os.remove(temp_wav)

def extract_complete_glottal_features_exact(y, sr=TARGET_SR):
    """Use one prepared waveform for voice-quality, spectral, and DisVoice values."""
    features = {}
    features.update(extract_glottal_voice_quality(y, sr))
    features.update(extract_glottal_spectral_features(y, sr))
    disvoice_values = extract_disvoice_glottal_exact(y, sr)
    for name, value in zip(DISVOICE_FEATURE_NAMES, disvoice_values):
        features[name] = float(value)
    return features

def extract_selected20_features_exact(y, sr=TARGET_SR):
    all_features = extract_complete_glottal_features_exact(y, sr)
    selected = {}
    for feature_name in SELECTED20:
        if feature_name not in all_features:
            raise KeyError(f'Missing selected Glottal feature: {feature_name}')
        value = float(all_features[feature_name])
        if not np.isfinite(value):
            raise ValueError(f'Non-finite selected Glottal feature: {feature_name}={value}')
        selected[feature_name] = value
    if len(selected) != 20:
        raise RuntimeError(f'Expected 20 selected Glottal features, got {len(selected)}')
    return selected

def extract_feature_mapping(y, sr=TARGET_SR):
    """Merge full-clip feature groups in model order, retaining shared base names."""
    if sr != TARGET_SR or y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError('Expected a finite, prepared mono waveform at 16 kHz.')
    combined = extract_acoustic_features(y, sr)
    for name, value in extract_selected20_features_exact(y, sr).items():
        if name not in combined:
            combined[name] = value
    return {name: float(combined[name]) if np.isfinite(combined[name]) else np.nan
            for name in FEATURE_NAMES}
