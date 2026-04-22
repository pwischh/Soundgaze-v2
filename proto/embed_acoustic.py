"""
Extract extended acoustic features from FMA Small, z-score standardize
per-dimension across the corpus, then apply group weights, and save to parquet.

Output:
  data/embeddings/fma_small_acoustic.parquet  — track_id, features (70-dim, standardized+weighted)
  data/embeddings/acoustic_scaler.npz         — mean, std, weights (for standardizing new clips)

Feature breakdown (70 dims):
  MFCCs      20 coef × mean + std  = 40  (group weight 0.7 — slightly downweighted)
  Chroma CQT 12 bins × mean        = 12
  Spectral   centroid, rolloff, flatness, contrast×7  = 10
  Rhythm     tempo, onset strength  =  2
  Dynamics   ZCR, RMS log          =  2
  Density    spectral entropy, bandwidth, onset rate, H/N ratio  = 4  (group weight 1.3)

Usage:
  python embed_acoustic.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from scipy.stats import entropy as scipy_entropy

from config import FMA_SMALL_DIR, EMBEDDINGS_DIR

SR           = 22050
DURATION     = 30.0
# v2: 70-dim with density features + group weights. Originals (v1) kept for comparison.
OUTPUT_PATH  = EMBEDDINGS_DIR / "fma_small_acoustic_v2.parquet"
SCALER_PATH  = EMBEDDINGS_DIR / "acoustic_scaler_v2.npz"

# Per-group cosine weights applied after z-scoring.
# MFCCs (40 dims) are downweighted so they don't dominate similarity.
# Density features (4 dims) are upweighted to amplify their signal.
GROUP_WEIGHTS = np.array(
    [0.7] * 40 +   # MFCCs (mean + std of 20 coefs)
    [1.0] * 12 +   # Chroma CQT
    [1.0] * 10 +   # Spectral (centroid, rolloff, flatness, contrast×7)
    [1.0] *  2 +   # Rhythm (tempo, onset strength)
    [1.0] *  2 +   # Dynamics (ZCR, RMS log)
    [1.3] *  4,    # Density (entropy, bandwidth, onset rate, H/N ratio)
    dtype=np.float32,
)


def extract_raw(path: Path) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION)
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return None

    if len(audio) < SR:
        return None

    # Compute STFT once; reuse S and mel for all downstream features.
    D      = librosa.stft(audio)
    S      = np.abs(D)
    mel    = librosa.feature.melspectrogram(S=S ** 2, sr=SR)
    mel_db = librosa.power_to_db(mel)

    mfcc     = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    chroma   = librosa.feature.chroma_stft(S=S, sr=SR)
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR, n_bands=6)

    onset_env = librosa.onset.onset_strength(S=mel_db, sr=SR)
    tempo_arr = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR)[0]

    spec_centroid  = float(librosa.feature.spectral_centroid(S=S, sr=SR).mean())
    spec_rolloff   = float(librosa.feature.spectral_rolloff(S=S, sr=SR).mean())
    spec_flatness  = float(librosa.feature.spectral_flatness(S=S).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=SR).mean())

    # Density features — all derived from already-computed transforms
    power_norm   = (S ** 2) / ((S ** 2).sum(axis=0, keepdims=True) + 1e-10)
    spec_entropy = float(scipy_entropy(power_norm.mean(axis=1) + 1e-10))
    onsets       = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR)
    onset_rate   = len(onsets) / max(len(audio) / SR, 1e-3)
    D_h, D_p     = librosa.decompose.hpss(D)
    rms_h = float(librosa.feature.rms(S=np.abs(D_h)).mean())
    rms_p = float(librosa.feature.rms(S=np.abs(D_p)).mean())
    hnr   = np.log10(max(rms_h, 1e-6) / max(rms_p, 1e-6))

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),                              # 40
        chroma.mean(axis=1),                                               # 12
        [spec_centroid, spec_rolloff, spec_flatness],                      #  3
        contrast.mean(axis=1),                                             #  7
        [float(np.atleast_1d(tempo_arr)[0]),
         float(onset_env.mean()),
         float(librosa.feature.zero_crossing_rate(audio).mean()),
         float(np.log10(max(librosa.feature.rms(S=S).mean(), 1e-6)))],    #  4
        [spec_entropy, spec_bandwidth, onset_rate, hnr],                   #  4 density
    ]).astype(np.float32)


def main() -> None:
    mp3_files = sorted(FMA_SMALL_DIR.rglob("*.mp3"))
    print(f"Found {len(mp3_files)} tracks.\n")

    track_ids: list[int]       = []
    raw_feats: list[np.ndarray] = []

    for i, path in enumerate(mp3_files):
        feat = extract_raw(path)
        if feat is None:
            continue
        track_ids.append(int(path.stem))
        raw_feats.append(feat)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(mp3_files)} ...")

    matrix = np.stack(raw_feats)                   # (N, 70)
    mu     = matrix.mean(axis=0)                   # (70,)
    sigma  = matrix.std(axis=0).clip(min=1e-8)     # (70,) — avoid div-by-zero

    standardized = (matrix - mu) / sigma * GROUP_WEIGHTS

    print(f"\nSaving scaler → {SCALER_PATH}")
    np.savez(SCALER_PATH, mean=mu, std=sigma, weights=GROUP_WEIGHTS)

    print(f"Saving {len(track_ids)} tracks → {OUTPUT_PATH}")
    records = [
        {"track_id": tid, "features": feat.tolist()}
        for tid, feat in zip(track_ids, standardized)
    ]
    pd.DataFrame(records).to_parquet(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
