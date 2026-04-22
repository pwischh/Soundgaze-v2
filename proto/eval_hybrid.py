"""
Evaluate CLAP + acoustic hybrid similarity at a sweep of alpha values.

hybrid_sim(A, B) = alpha * cosine(clap_A, clap_B) + (1-alpha) * cosine(acoustic_A, acoustic_B)

Implemented via weighted concatenation of unit-norm vectors so timbremetrics
can compute the hybrid in a single pass per alpha. CLAP inference is cached
after the first timbremetrics run so subsequent alphas are free.

Metrics:
  timbremetrics  — Spearman ρ, Kendall τ, Triplet Agreement, NDCG, MAE
  Covers80       — Top-1, Top-5, MRR  (requires --covers80)

Usage:
  python eval_hybrid.py [--covers80 /path/to/covers80]
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import librosa
import pandas as pd
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
import laion_clap
from timbremetrics import TimbreMetric

from config import EMBEDDINGS_DIR

CLAP_SR    = 48000
TIMBRE_SR  = 44100
LIBROSA_SR = 22050
DURATION   = 30.0
CLAP_LEN   = int(CLAP_SR * DURATION)

ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]

# "v2" = 70-dim with density features + group weights; "v1" = original 66-dim
ACOUSTIC_VERSION = "v2"

# Module-level caches — populated on first timbremetrics run
_clap_cache:     dict[bytes, np.ndarray] = {}
_acoustic_cache: dict[bytes, np.ndarray] = {}

# Corpus scaler — loaded once in main(), used everywhere
_scaler_mean:    np.ndarray | None = None
_scaler_std:     np.ndarray | None = None
_scaler_weights: np.ndarray | None = None


def standardize(v: np.ndarray) -> np.ndarray:
    if _scaler_mean is None:
        return v
    z = (v - _scaler_mean) / _scaler_std
    if _scaler_weights is not None:
        z = z * _scaler_weights
    return z


# ---------------------------------------------------------------------------
# Acoustic feature extraction  (70-dim, matches embed_acoustic.py)
# ---------------------------------------------------------------------------

def acoustic_features(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != LIBROSA_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=LIBROSA_SR)
        sr = LIBROSA_SR

    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))

    # Compute STFT once; reuse for all downstream features (matches embed_acoustic.py).
    D      = librosa.stft(audio)
    S      = np.abs(D)
    mel    = librosa.feature.melspectrogram(S=S ** 2, sr=sr)
    mel_db = librosa.power_to_db(mel)

    mfcc     = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    chroma   = librosa.feature.chroma_stft(S=S, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)

    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr)
    tempo_arr = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]

    spec_centroid  = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    spec_rolloff   = float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean())
    spec_flatness  = float(librosa.feature.spectral_flatness(S=S).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean())

    power_norm   = (S ** 2) / ((S ** 2).sum(axis=0, keepdims=True) + 1e-10)
    spec_entropy = float(scipy_entropy(power_norm.mean(axis=1) + 1e-10))
    onsets       = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate   = len(onsets) / max(len(audio) / sr, 1e-3)
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


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


# ---------------------------------------------------------------------------
# Timbremetrics hybrid callable with caching
# ---------------------------------------------------------------------------

def make_hybrid_callable(clap_model, alpha: float):
    """
    Weighted concatenation:  [sqrt(alpha)*norm(clap) | sqrt(1-alpha)*norm(acoustic)]
    cosine of this combined vector equals:
      alpha * cosine(clap_A, clap_B) + (1-alpha) * cosine(acoustic_A, acoustic_B)
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        key = x.numpy().tobytes()

        if key not in _clap_cache:
            audio_clap = librosa.resample(
                x.squeeze(0).numpy(), orig_sr=TIMBRE_SR, target_sr=CLAP_SR
            )
            emb = clap_model.get_audio_embedding_from_data(
                x=audio_clap.reshape(1, -1).astype(np.float32), use_tensor=False
            )
            _clap_cache[key] = emb[0]

            audio_ac = x.squeeze(0).numpy()
            raw = acoustic_features(audio_ac, TIMBRE_SR)
            _acoustic_cache[key] = standardize(raw)

        clap_n     = normalize(_clap_cache[key])
        acoustic_n = normalize(_acoustic_cache[key])

        combined = np.concatenate([
            np.sqrt(alpha)       * clap_n,
            np.sqrt(1.0 - alpha) * acoustic_n,
        ])
        return torch.from_numpy(combined.astype(np.float32))

    return fn


# ---------------------------------------------------------------------------
# Timbremetrics sweep
# ---------------------------------------------------------------------------

def run_timbremetrics(clap_model, device: str) -> None:
    metric = TimbreMetric(device=device, sample_rate=TIMBRE_SR)

    print(f"\n{'Alpha':<8} {'Spearman':>10} {'Kendall':>10} {'Triplet':>10} {'NDCG':>10} {'MAE':>10}")
    print("-" * 58)

    for alpha in ALPHAS:
        label = "CLAP only" if alpha == 1.0 else ("Acoustic" if alpha == 0.0 else f"α={alpha:.1f}")
        res = metric(make_hybrid_callable(clap_model, alpha))

        c = res.get("cosine", {})
        spearman = float(c.get("spearman_corr", float("nan")))
        kendall  = float(c.get("kendall_corr",  float("nan")))
        triplet  = float(c.get("triplet_agreement", float("nan")))
        ndcg     = float(c.get("ndcg_retrieve_sim", float("nan")))
        mae      = float(c.get("mae",              float("nan")))

        print(f"{label:<8} {spearman:>10.4f} {kendall:>10.4f} {triplet:>10.4f} {ndcg:>10.4f} {mae:>10.4f}")


# ---------------------------------------------------------------------------
# Covers80
# ---------------------------------------------------------------------------

def cosine_row(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    qnorm = np.linalg.norm(query)
    return (matrix @ query) / np.maximum(norms * qnorm, 1e-8)


def rank_of_partner(sims: np.ndarray, qi: int, pi: int) -> int:
    masked = sims.copy()
    masked[qi] = -np.inf
    return int(np.where(np.argsort(-masked) == pi)[0][0]) + 1


def run_covers80(covers_dir: Path, clap_model) -> None:
    covers32k = covers_dir / "covers32k"
    pairs: list[tuple[Path, Path]] = []
    for song_dir in sorted(covers32k.iterdir()):
        if not song_dir.is_dir():
            continue
        mp3s = sorted(song_dir.glob("*.mp3"))
        if len(mp3s) == 2:
            pairs.append((mp3s[0], mp3s[1]))

    all_paths   = [p for pair in pairs for p in pair]
    path_to_idx = {p: i for i, p in enumerate(all_paths)}
    print(f"\n── Covers80  ({len(pairs)} pairs, {len(all_paths)} tracks) ──\n")

    clap_embs: list[np.ndarray] = []
    ac_embs:   list[np.ndarray] = []
    valid:     list[bool]       = []

    for i, path in enumerate(all_paths):
        print(f"  [{i+1}/{len(all_paths)}] {path.name}", end="\r")
        try:
            audio_clap, _ = librosa.load(path, sr=CLAP_SR,    mono=True, duration=DURATION)
            audio_ac,   _ = librosa.load(path, sr=LIBROSA_SR, mono=True, duration=DURATION)
            clap_e = clap_model.get_audio_embedding_from_data(
                x=audio_clap.reshape(1, -1).astype(np.float32), use_tensor=False
            )[0]
            ac_e = standardize(acoustic_features(audio_ac, LIBROSA_SR))
            clap_embs.append(clap_e)
            ac_embs.append(ac_e)
            valid.append(True)
        except Exception:
            clap_embs.append(np.zeros(512,  dtype=np.float32))
            ac_embs.append(np.zeros(70,    dtype=np.float32))
            valid.append(False)
    print()

    clap_matrix = np.stack(clap_embs)
    ac_matrix   = np.stack(ac_embs)

    queries = []
    for a, b in pairs:
        ia, ib = path_to_idx[a], path_to_idx[b]
        if valid[ia] and valid[ib]:
            queries.append((ia, ib))
            queries.append((ib, ia))
    print(f"  {len(queries)} queries ({len(queries)//2} valid pairs)\n")

    print(f"  {'Label':<16} {'Top-1':>7} {'Top-5':>7} {'MRR':>8}")
    print("  " + "-" * 40)

    for alpha in ALPHAS:
        label = "CLAP only" if alpha == 1.0 else ("Acoustic" if alpha == 0.0 else f"Hybrid α={alpha:.1f}")
        clap_n = clap_matrix / np.maximum(np.linalg.norm(clap_matrix, axis=1, keepdims=True), 1e-8)
        ac_n   = ac_matrix   / np.maximum(np.linalg.norm(ac_matrix,   axis=1, keepdims=True), 1e-8)

        ranks = []
        for qi, pi in queries:
            sims = alpha * cosine_row(clap_n, clap_n[qi]) + (1 - alpha) * cosine_row(ac_n, ac_n[qi])
            ranks.append(rank_of_partner(sims, qi, pi))

        n    = len(ranks)
        top1 = sum(r == 1 for r in ranks) / n
        top5 = sum(r <= 5 for r in ranks) / n
        mrr  = float(np.mean([1.0 / r for r in ranks]))
        print(f"  {label:<16} {top1:>7.1%} {top5:>7.1%} {mrr:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--covers80", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scaler_path = EMBEDDINGS_DIR / f"acoustic_scaler_{ACOUSTIC_VERSION}.npz"
    if scaler_path.exists():
        global _scaler_mean, _scaler_std, _scaler_weights
        scaler = np.load(scaler_path)
        _scaler_mean    = scaler["mean"]
        _scaler_std     = scaler["std"]
        _scaler_weights = scaler["weights"] if "weights" in scaler else None
        print(f"Loaded acoustic scaler from {scaler_path}")
    else:
        print(f"WARNING: {scaler_path} not found — run embed_acoustic.py first. Acoustic features will be unscaled.")

    print("Loading CLAP model ...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    clap_model.load_ckpt()

    print("\n── timbremetrics alpha sweep ──")
    print("  (CLAP embeddings cached after first run)")
    run_timbremetrics(clap_model, device)

    if args.covers80:
        run_covers80(args.covers80, clap_model)


if __name__ == "__main__":
    main()
