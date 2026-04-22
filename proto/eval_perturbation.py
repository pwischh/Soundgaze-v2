"""
Perturbation robustness eval for hybrid sonic similarity.

For N sampled FMA tracks we apply controlled perturbations:
  - Pitch shift: -2, -1, +1, +2 semitones
  - Time stretch: 0.90x, 0.95x, 1.05x, 1.10x

Each perturbed version is embedded and queried against the full
precomputed FMA corpus. We report what rank the original gets.

Optimisations:
  - Audio loaded once at model SR, resampled for librosa (no double file read)
  - Perturbations applied once, resampled for handcrafted features
  - PANN: all perturbations batched in a single forward pass
  - Corpus pre-normalised so cosine sim is a single matmul

Usage:
  python eval_perturbation.py [--model pann|clap] [--n 100] [--seed 42]
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from panns_inference import AudioTagging
import laion_clap

from config import FMA_SMALL_DIR, EMBEDDINGS_DIR

PANN_SR      = 32000
CLAP_SR      = 48000
LIBROSA_SR   = 22050
DURATION     = 30.0
PANN_LEN     = int(PANN_SR    * DURATION)
CLAP_LEN     = int(CLAP_SR    * DURATION)
LIBROSA_LEN  = int(LIBROSA_SR * DURATION)

TEMPO_MAX    = 240.0
CENTROID_MAX = 8000.0
RMS_LOG_MIN  = -6.0
RMS_LOG_MAX  = 0.0

ALPHAS        = np.round(np.arange(0.0, 1.01, 0.05), 2).tolist()
PITCH_SHIFTS  = [-2, -1, 1, 2]
TIME_RATES    = [0.90, 0.95, 1.05, 1.10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fix_length(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.pad(audio, (0, target - len(audio)))


def load_audio(track_id: int, sr: int, length: int) -> np.ndarray | None:
    subdir = f"{track_id:06d}"[:3]
    path = FMA_SMALL_DIR / subdir / f"{track_id:06d}.mp3"
    if not path.exists():
        return None
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True, duration=DURATION)
        return fix_length(audio, length)
    except Exception:
        return None


def to_librosa(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=LIBROSA_SR)
    return fix_length(resampled, LIBROSA_LEN)


def handcrafted(audio: np.ndarray, sr: int) -> np.ndarray:
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_norm = float(np.clip(tempo / TEMPO_MAX, 0.0, 1.0))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_norm = float(np.clip(centroid.mean() / CENTROID_MAX, 0.0, 1.0))

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    contrast_mean = contrast.mean(axis=1)

    rms = librosa.feature.rms(y=audio)
    rms_log = np.log10(np.maximum(rms.mean(), 1e-6))
    rms_norm = float(np.clip((rms_log - RMS_LOG_MIN) / (RMS_LOG_MAX - RMS_LOG_MIN), 0.0, 1.0))

    return np.concatenate([[tempo_norm, centroid_norm], contrast_mean, [rms_norm]])


def cosine_sim(norm_matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    q = query / max(float(np.linalg.norm(query)), 1e-8)
    return norm_matrix @ q


def rank_of(sims: np.ndarray, target_idx: int) -> int:
    return int(np.where(np.argsort(-sims) == target_idx)[0][0]) + 1


def embed_clap_batch(model, audios: list[np.ndarray]) -> np.ndarray:
    x = np.stack(audios).astype(np.float32)
    return model.get_audio_embedding_from_data(x=x, use_tensor=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pann", "clap"], default="pann")
    parser.add_argument("--n",    type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_clap = args.model == "clap"
    neural_sr  = CLAP_SR  if use_clap else PANN_SR
    neural_len = CLAP_LEN if use_clap else PANN_LEN
    parquet    = "fma_small_clap.parquet" if use_clap else "fma_small_embeddings.parquet"
    emb_col    = "embedding"

    # --- Load & pre-normalise corpus ---
    print("Loading corpus ...")
    neural_df = pd.read_parquet(EMBEDDINGS_DIR / parquet)
    hc_df     = pd.read_parquet(EMBEDDINGS_DIR / "fma_small_handcrafted.parquet")
    merged    = neural_df.merge(hc_df, on="track_id")

    corpus_ids    = merged["track_id"].to_numpy()
    neural_corpus = np.stack(merged[emb_col].to_numpy())
    hc_corpus     = np.stack(merged["features"].to_numpy())
    id_to_idx     = {tid: i for i, tid in enumerate(corpus_ids)}

    neural_norm = neural_corpus / np.maximum(np.linalg.norm(neural_corpus, axis=1, keepdims=True), 1e-8)
    hc_norm     = hc_corpus     / np.maximum(np.linalg.norm(hc_corpus,     axis=1, keepdims=True), 1e-8)

    print(f"Corpus: {len(corpus_ids)} tracks\n")

    rng        = np.random.default_rng(args.seed)
    sample_ids = rng.choice(corpus_ids, size=min(args.n, len(corpus_ids)), replace=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_clap:
        print("Loading CLAP HTSAT-tiny ...")
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        model.load_ckpt()
    else:
        print(f"Loading CNN10 ({device}) ...")
        model = AudioTagging(checkpoint_path=None, device=device)

    PERT_LABELS = (
        [f"pitch {n:+d}st" for n in PITCH_SHIFTS] +
        [f"time  {r:.2f}x" for r in TIME_RATES]
    )
    results: dict[str, dict[float, list[int]]] = {
        lbl: {a: [] for a in ALPHAS} for lbl in PERT_LABELS
    }

    for i, track_id in enumerate(sample_ids):
        print(f"  [{i+1}/{len(sample_ids)}] track {track_id}", end="\r")

        orig_audio = load_audio(int(track_id), neural_sr, neural_len)
        if orig_audio is None:
            continue
        orig_idx = id_to_idx[track_id]

        # Build perturbed audios at neural_sr
        pert_neural: list[np.ndarray] = []
        for n_steps in PITCH_SHIFTS:
            a = librosa.effects.pitch_shift(orig_audio, sr=neural_sr, n_steps=n_steps)
            pert_neural.append(fix_length(a, neural_len))
        for rate in TIME_RATES:
            a = librosa.effects.time_stretch(orig_audio, rate=rate)
            pert_neural.append(fix_length(a, neural_len))

        # Inference
        try:
            if use_clap:
                embs = embed_clap_batch(model, pert_neural)          # (8, 512)
            else:
                batch = np.stack(pert_neural)
                _, embs = model.inference(batch)                     # (8, 2048)
        except Exception:
            continue

        # Handcrafted features (resample each perturbed audio to LIBROSA_SR)
        hc_vecs = [handcrafted(to_librosa(a, neural_sr), LIBROSA_SR) for a in pert_neural]

        # Score & record ranks
        for j, label in enumerate(PERT_LABELS):
            neural_sims = cosine_sim(neural_norm, embs[j])
            hc_sims     = cosine_sim(hc_norm,     hc_vecs[j])
            for alpha in ALPHAS:
                combined = alpha * neural_sims + (1.0 - alpha) * hc_sims
                results[label][alpha].append(rank_of(combined, orig_idx))

    n_tracks = len(next(iter(results.values()))[1.0])
    model_label = args.model.upper()
    print(f"\n\nPerturbation robustness  ({model_label}, n={n_tracks} tracks)\n")
    print("Top-1 accuracy by alpha\n")

    alpha_strs = [f"{a:.2f}" for a in ALPHAS]
    print(f"{'Perturbation':<14}" + "".join(f" {s:>5}" for s in alpha_strs) + "  best_α")
    print("-" * (14 + 6 * len(ALPHAS) + 8))

    for label in PERT_LABELS:
        row_top1 = {a: sum(r == 1 for r in results[label][a]) / len(results[label][a])
                    for a in ALPHAS}
        best_alpha = max(row_top1, key=row_top1.__getitem__)
        cells = "".join(f" {row_top1[a]:>5.1%}" for a in ALPHAS)
        print(f"{label:<14}{cells}  {best_alpha:.2f}")

    print()
    means = []
    for alpha in ALPHAS:
        all_ranks = [r for lbl in PERT_LABELS for r in results[lbl][alpha]]
        means.append(sum(r == 1 for r in all_ranks) / len(all_ranks))

    best_overall = ALPHAS[int(np.argmax(means))]
    mean_cells   = "".join(f" {m:>5.1%}" for m in means)
    print(f"{'mean (all pert)':<14}{mean_cells}  {best_overall:.2f}")


if __name__ == "__main__":
    main()
