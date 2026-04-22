import numpy as np
import librosa
from scipy.stats import entropy as scipy_entropy

from .config import LIBROSA_SR

GROUP_WEIGHTS = np.array(
    [0.7] * 40 +   # MFCCs (mean + std of 20 coefs)
    [1.0] * 12 +   # Chroma STFT
    [1.0] * 10 +   # Spectral (centroid, rolloff, flatness, contrast×7)
    [1.0] *  2 +   # Rhythm (tempo, onset strength)
    [1.0] *  2 +   # Dynamics (ZCR, RMS log)
    [1.3] *  4,    # Density (entropy, bandwidth, onset rate, H/N ratio)
    dtype=np.float32,
)


def acoustic_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract 70-dim acoustic feature vector. Resamples to LIBROSA_SR if needed."""
    if sr != LIBROSA_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=LIBROSA_SR)

    D      = librosa.stft(audio)
    S      = np.abs(D)
    mel    = librosa.feature.melspectrogram(S=S ** 2, sr=LIBROSA_SR)
    mel_db = librosa.power_to_db(mel)

    mfcc     = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    chroma   = librosa.feature.chroma_stft(S=S, sr=LIBROSA_SR)
    contrast = librosa.feature.spectral_contrast(S=S, sr=LIBROSA_SR, n_bands=6)

    onset_env = librosa.onset.onset_strength(S=mel_db, sr=LIBROSA_SR)
    tempo_arr = librosa.beat.beat_track(onset_envelope=onset_env, sr=LIBROSA_SR)[0]

    spec_centroid  = float(librosa.feature.spectral_centroid(S=S, sr=LIBROSA_SR).mean())
    spec_rolloff   = float(librosa.feature.spectral_rolloff(S=S, sr=LIBROSA_SR).mean())
    spec_flatness  = float(librosa.feature.spectral_flatness(S=S).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=LIBROSA_SR).mean())

    power_norm   = (S ** 2) / ((S ** 2).sum(axis=0, keepdims=True) + 1e-10)
    spec_entropy = float(scipy_entropy(power_norm.mean(axis=1) + 1e-10))
    onsets       = librosa.onset.onset_detect(onset_envelope=onset_env, sr=LIBROSA_SR)
    onset_rate   = len(onsets) / max(len(audio) / LIBROSA_SR, 1e-3)
    D_h, D_p     = librosa.decompose.hpss(D)
    rms_h = float(librosa.feature.rms(S=np.abs(D_h)).mean())
    rms_p = float(librosa.feature.rms(S=np.abs(D_p)).mean())
    hnr   = np.log10(max(rms_h, 1e-6) / max(rms_p, 1e-6))

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1),
        [spec_centroid, spec_rolloff, spec_flatness],
        contrast.mean(axis=1),
        [float(np.atleast_1d(tempo_arr)[0]),
         float(onset_env.mean()),
         float(librosa.feature.zero_crossing_rate(audio).mean()),
         float(np.log10(max(librosa.feature.rms(S=S).mean(), 1e-6)))],
        [spec_entropy, spec_bandwidth, onset_rate, hnr],
    ]).astype(np.float32)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-8)
