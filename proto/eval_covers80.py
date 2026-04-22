"""
Evaluate hybrid similarity on the Covers80 dataset.

Dataset layout expected (standard covers80 distribution):
  <covers80_dir>/
    covers32k/
      <SongName>/
        artist1+album+track.mp3
        artist2+album+track.mp3   # exactly 2 files per directory = one pair

Usage:
  python eval_covers80.py --covers80 /path/to/covers80 [--model pann|clap]

What it reports:
  For each of 80 pairs we ask: is the partner the top-1 most similar track
  (excluding the query itself) across all 160 tracks?

  Top-1 accuracy = fraction of queries where partner is rank-1.
  Mean Reciprocal Rank (MRR) = mean of 1/rank across all queries.
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import librosa
from pathlib import Path
from panns_inference import AudioTagging
import laion_clap

PANN_SR    = 32000
CLAP_SR    = 48000
LIBROSA_SR = 22050
DURATION   = 30.0

TEMPO_MAX    = 240.0
CENTROID_MAX = 8000.0
RMS_LOG_MIN  = -6.0
RMS_LOG_MAX  = 0.0

ALPHAS = np.round(np.arange(0.0, 1.01, 0.05), 2).tolist()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_pann(path: Path, model: AudioTagging) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(path, sr=PANN_SR, mono=True, duration=DURATION)
        _, embedding = model.inference(audio.reshape(1, -1))
        return embedding[0]
    except Exception as e:
        print(f"  PANN SKIP {path.name}: {e}")
        return None


def extract_clap(path: Path, model) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(path, sr=CLAP_SR, mono=True, duration=DURATION)
        emb = model.get_audio_embedding_from_data(
            x=audio.reshape(1, -1).astype(np.float32), use_tensor=False
        )
        return emb[0]
    except Exception as e:
        print(f"  CLAP SKIP {path.name}: {e}")
        return None


def extract_handcrafted(path: Path) -> np.ndarray | None:
    try:
        audio, sr = librosa.load(path, sr=LIBROSA_SR, mono=True, duration=DURATION)
    except Exception as e:
        print(f"  HC SKIP {path.name}: {e}")
        return None

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


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_sim(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    return (matrix @ query) / np.maximum(norms * query_norm, 1e-8)


def rank_of_partner(sims: np.ndarray, query_idx: int, partner_idx: int) -> int:
    masked = sims.copy()
    masked[query_idx] = -np.inf
    order = np.argsort(-masked)
    return int(np.where(order == partner_idx)[0][0]) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--covers80", required=True, type=Path,
                        help="Path to covers80 root (contains covers32k/ subdir)")
    parser.add_argument("--model", choices=["pann", "clap"], default="pann")
    args = parser.parse_args()

    covers32k = args.covers80 / "covers32k"
    assert covers32k.exists(), f"covers32k not found under {args.covers80}"

    use_clap   = args.model == "clap"
    neural_dim = 512 if use_clap else 2048
    model_label = "CLAP" if use_clap else "PANN"

    # Discover pairs: each subdir has exactly 2 MP3s
    pairs: list[tuple[Path, Path]] = []
    for song_dir in sorted(covers32k.iterdir()):
        if not song_dir.is_dir():
            continue
        mp3s = sorted(song_dir.glob("*.mp3"))
        if len(mp3s) != 2:
            print(f"  SKIP {song_dir.name}: found {len(mp3s)} mp3(s), expected 2")
            continue
        pairs.append((mp3s[0], mp3s[1]))

    all_paths   = [p for pair in pairs for p in pair]
    path_to_idx = {p: i for i, p in enumerate(all_paths)}

    print(f"Covers80: {len(pairs)} pairs, {len(all_paths)} total tracks\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_clap:
        print("Loading CLAP HTSAT-tiny ...")
        neural_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        neural_model.load_ckpt()
        extract_neural = lambda path: extract_clap(path, neural_model)
    else:
        print(f"Loading CNN10 ({device}) ...")
        neural_model = AudioTagging(checkpoint_path=None, device=device)
        extract_neural = lambda path: extract_pann(path, neural_model)

    neural_embs = []
    hc_embs     = []
    valid_mask  = []

    for i, path in enumerate(all_paths):
        print(f"  [{i+1}/{len(all_paths)}] {path.name}", end="\r")
        n = extract_neural(path)
        h = extract_handcrafted(path)
        valid_mask.append(n is not None and h is not None)
        neural_embs.append(n if n is not None else np.zeros(neural_dim))
        hc_embs.append(h if h is not None else np.zeros(10))

    print()
    neural_matrix = np.stack(neural_embs)
    hc_matrix     = np.stack(hc_embs)

    queries = []
    for a, b in pairs:
        idx_a, idx_b = path_to_idx[a], path_to_idx[b]
        if valid_mask[idx_a] and valid_mask[idx_b]:
            queries.append((idx_a, idx_b))
            queries.append((idx_b, idx_a))

    print(f"Evaluating {len(queries)} queries ({len(queries)//2} pairs, both directions)\n")

    results: dict[float, list[int]] = {a: [] for a in ALPHAS}

    for query_idx, partner_idx in queries:
        neural_sims = cosine_sim(neural_matrix, neural_matrix[query_idx])
        hc_sims     = cosine_sim(hc_matrix,     hc_matrix[query_idx])

        for alpha in ALPHAS:
            combined = alpha * neural_sims + (1 - alpha) * hc_sims
            results[alpha].append(rank_of_partner(combined, query_idx, partner_idx))

    print(f"{'Mode':<22} {'Top-1':>6} {'Top-5':>6} {'MRR':>7}")
    print("-" * 45)
    for alpha in ALPHAS:
        ranks = results[alpha]
        n     = len(ranks)
        top1  = sum(r == 1 for r in ranks) / n
        top5  = sum(r <= 5 for r in ranks) / n
        mrr   = np.mean([1.0 / r for r in ranks])

        if alpha == 1.0:
            label = f"{model_label} only"
        elif alpha == 0.0:
            label = "Handcrafted only"
        else:
            label = f"Hybrid (α={alpha})"

        print(f"{label:<22} {top1:>6.1%} {top5:>6.1%} {mrr:>7.4f}")


if __name__ == "__main__":
    main()
