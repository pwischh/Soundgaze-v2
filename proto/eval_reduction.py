"""
Evaluate UMAP and t-SNE projections of CLAP embeddings (512-dim → 3-dim).

Metrics reported for each representation:
  KNN Recall       — fraction of original k-NN preserved in projection (k=10)
  Trustworthiness  — fraction of projected k-NN that were truly close in original space
  Hubness          — k-occurrence skewness, max k-occ, zero-occ %
  Covers80         — Top-1, Top-5, MRR  (requires --covers80)
  timbremetrics    — Spearman ρ, Kendall τ, Triplet Agreement, NDCG, MAE
                     (CLAP-512 and UMAP-3D only; t-SNE has no transform for new clips)

Usage:
  python eval_reduction.py [--covers80 /path/to/covers80] [--skip-tsne]
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
import laion_clap
import umap as umap_lib
from sklearn.manifold import TSNE, trustworthiness
from timbremetrics import TimbreMetric

from config import EMBEDDINGS_DIR

CLAP_SR   = 48000
DURATION  = 30.0
CLAP_LEN  = int(CLAP_SR * DURATION)
TIMBRE_SR = 44100
K_HUB     = 10
K_RECALL  = 10


# ---------------------------------------------------------------------------
# KNN recall + trustworthiness
# ---------------------------------------------------------------------------

def knn_indices(matrix: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) array of k nearest neighbor indices (cosine, excluding self)."""
    normed = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    sim = normed.astype(np.float32) @ normed.astype(np.float32).T
    np.fill_diagonal(sim, -np.inf)
    # argpartition gives top-k in arbitrary order; sort for consistency
    top_k = np.argpartition(-sim, k, axis=1)[:, :k]
    return top_k


def knn_recall(orig: np.ndarray, reduced: np.ndarray, k: int) -> float:
    """Mean fraction of original k-NN preserved after reduction."""
    nn_orig    = knn_indices(orig,    k)
    nn_reduced = knn_indices(reduced, k)
    recalls = [
        len(set(nn_orig[i]) & set(nn_reduced[i])) / k
        for i in range(len(orig))
    ]
    return float(np.mean(recalls))


def projection_report(name: str, orig: np.ndarray, reduced: np.ndarray) -> None:
    recall = knn_recall(orig, reduced, K_RECALL)
    # sklearn trustworthiness uses Euclidean by default; pass precomputed ranks for cosine
    tw = trustworthiness(orig, reduced, n_neighbors=K_RECALL, metric="cosine")
    print(f"  {name:<22} KNN-recall={recall:.4f}  trustworthiness={tw:.4f}")


# ---------------------------------------------------------------------------
# Hubness
# ---------------------------------------------------------------------------

def compute_koccurrence(matrix: np.ndarray, k: int) -> np.ndarray:
    normed = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    sim = (normed.astype(np.float32) @ normed.astype(np.float32).T)
    np.fill_diagonal(sim, -np.inf)
    top_k = np.argpartition(-sim, k, axis=1)[:, :k]
    return np.bincount(top_k.ravel(), minlength=len(matrix))


def skewness(x: np.ndarray) -> float:
    return float(np.mean(((x - x.mean()) / x.std()) ** 3))


def hubness_report(name: str, matrix: np.ndarray) -> None:
    kocc = compute_koccurrence(matrix, K_HUB)
    print(f"  {name:<22} skew={skewness(kocc):>6.3f}  max_k={kocc.max():>4}  zero={( kocc == 0).mean():>5.1%}")


# ---------------------------------------------------------------------------
# Covers80
# ---------------------------------------------------------------------------

def load_clap_audio(path: Path) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(path, sr=CLAP_SR, mono=True, duration=DURATION)
        if len(audio) < CLAP_LEN:
            audio = np.pad(audio, (0, CLAP_LEN - len(audio)))
        return audio[:CLAP_LEN]
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return None


def cosine_row(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    qnorm = np.linalg.norm(query)
    return (matrix @ query) / np.maximum(norms * qnorm, 1e-8)


def rank_of_partner(sims: np.ndarray, qi: int, pi: int) -> int:
    masked = sims.copy()
    masked[qi] = -np.inf
    return int(np.where(np.argsort(-masked) == pi)[0][0]) + 1


def eval_covers80(covers_dir: Path, clap_model, reducer_umap) -> None:
    covers32k = covers_dir / "covers32k"
    pairs: list[tuple[Path, Path]] = []
    for song_dir in sorted(covers32k.iterdir()):
        if not song_dir.is_dir():
            continue
        mp3s = sorted(song_dir.glob("*.mp3"))
        if len(mp3s) == 2:
            pairs.append((mp3s[0], mp3s[1]))

    all_paths = [p for pair in pairs for p in pair]
    path_to_idx = {p: i for i, p in enumerate(all_paths)}
    print(f"\n── Covers80  ({len(pairs)} pairs, {len(all_paths)} tracks) ──")

    raw_embs: list[np.ndarray] = []
    valid: list[bool] = []
    for i, path in enumerate(all_paths):
        print(f"  [{i+1}/{len(all_paths)}] {path.name}", end="\r")
        audio = load_clap_audio(path)
        if audio is None:
            valid.append(False)
            raw_embs.append(np.zeros(512, dtype=np.float32))
            continue
        emb = clap_model.get_audio_embedding_from_data(
            x=audio.reshape(1, -1).astype(np.float32), use_tensor=False
        )
        raw_embs.append(emb[0])
        valid.append(True)
    print()

    clap_matrix = np.stack(raw_embs)
    umap_matrix = reducer_umap.transform(clap_matrix)

    # t-SNE fitted on covers80 tracks only — self-contained so no transform needed
    print("  Fitting t-SNE on covers80 tracks ...")
    tsne_covers = TSNE(n_components=3, metric="cosine", random_state=42,
                       perplexity=min(30, len(clap_matrix) // 4), n_jobs=-1)
    tsne_matrix = tsne_covers.fit_transform(clap_matrix)

    queries = []
    for a, b in pairs:
        ia, ib = path_to_idx[a], path_to_idx[b]
        if valid[ia] and valid[ib]:
            queries.append((ia, ib))
            queries.append((ib, ia))
    print(f"  {len(queries)} queries ({len(queries)//2} valid pairs)\n")

    def report(name: str, matrix: np.ndarray) -> None:
        ranks = [rank_of_partner(cosine_row(matrix, matrix[qi]), qi, pi) for qi, pi in queries]
        n = len(ranks)
        top1 = sum(r == 1 for r in ranks) / n
        top5 = sum(r <= 5 for r in ranks) / n
        mrr  = float(np.mean([1.0 / r for r in ranks]))
        print(f"  {name:<22} Top-1={top1:>6.1%}  Top-5={top5:>6.1%}  MRR={mrr:.4f}")

    report("CLAP (512-dim)", clap_matrix)
    report("UMAP (3-dim)",   umap_matrix)
    report("t-SNE (3-dim)",  tsne_matrix)


# ---------------------------------------------------------------------------
# timbremetrics
# ---------------------------------------------------------------------------

def make_clap_callable(clap_model):
    def fn(x: torch.Tensor) -> torch.Tensor:
        audio = librosa.resample(x.squeeze(0).numpy(), orig_sr=TIMBRE_SR, target_sr=CLAP_SR)
        emb = clap_model.get_audio_embedding_from_data(
            x=audio.reshape(1, -1).astype(np.float32), use_tensor=False
        )
        return torch.from_numpy(emb[0].copy())
    return fn


def make_umap_callable(clap_model, reducer):
    def fn(x: torch.Tensor) -> torch.Tensor:
        audio = librosa.resample(x.squeeze(0).numpy(), orig_sr=TIMBRE_SR, target_sr=CLAP_SR)
        emb = clap_model.get_audio_embedding_from_data(
            x=audio.reshape(1, -1).astype(np.float32), use_tensor=False
        )
        reduced = reducer.transform(emb)[0].astype(np.float32)
        return torch.from_numpy(reduced.copy())
    return fn


def print_timbre_comparison(clap_res: dict, umap_res: dict) -> None:
    print(f"\n{'Metric':<40} {'CLAP-512':>10} {'UMAP-3D':>10}")
    print("-" * 62)
    for distance in ["cosine", "l2"]:
        c_sub = clap_res.get(distance, {})
        u_sub = umap_res.get(distance, {})
        for metric in sorted(c_sub.keys()):
            label = f"{distance}/{metric}"
            c_val = float(c_sub.get(metric, float("nan")))
            u_val = float(u_sub.get(metric, float("nan")))
            print(f"{label:<40} {c_val:>10.4f} {u_val:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--covers80", type=Path, default=None)
    parser.add_argument("--skip-tsne", action="store_true",
                        help="Skip t-SNE (takes several minutes on CPU)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading CLAP embeddings ...")
    df = pd.read_parquet(EMBEDDINGS_DIR / "fma_small_clap.parquet")
    ids = df["track_id"].to_numpy()
    emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    print(f"  {len(ids)} tracks, {emb.shape[1]}-dim\n")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("Fitting UMAP (cosine, 3-dim) ...")
    reducer_umap = umap_lib.UMAP(n_components=3, metric="cosine", random_state=42, verbose=False)
    umap_emb = reducer_umap.fit_transform(emb)
    print(f"  Done. shape={umap_emb.shape}\n")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    tsne_emb = None
    if not args.skip_tsne:
        print("Fitting t-SNE (cosine, 3-dim) — this may take several minutes ...")
        tsne = TSNE(n_components=3, metric="cosine", random_state=42,
                    n_jobs=-1, verbose=1, perplexity=30)
        tsne_emb = tsne.fit_transform(emb)
        print(f"  Done. shape={tsne_emb.shape}\n")

    # ── KNN recall + trustworthiness ─────────────────────────────────────────
    print("── KNN Recall & Trustworthiness (k=10) ──")
    projection_report("UMAP (3-dim)", emb, umap_emb)
    if tsne_emb is not None:
        projection_report("t-SNE (3-dim)", emb, tsne_emb)

    # ── Hubness ──────────────────────────────────────────────────────────────
    print("\n── Hubness (k=10) ──")
    hubness_report("CLAP original (512-dim)", emb)
    hubness_report("UMAP (3-dim)", umap_emb)
    if tsne_emb is not None:
        hubness_report("t-SNE (3-dim)", tsne_emb)

    # ── timbremetrics ────────────────────────────────────────────────────────
    print("\n── timbremetrics (CLAP-512 vs UMAP-3D) ──")
    print("  t-SNE excluded: sklearn TSNE has no transform() for new clips\n")
    print("Loading CLAP model ...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    clap_model.load_ckpt()

    metric = TimbreMetric(device=device, sample_rate=TIMBRE_SR)

    print("Running timbremetrics on CLAP-512 ...")
    clap_res = metric(make_clap_callable(clap_model))

    print("Running timbremetrics on UMAP-3D ...")
    umap_res = metric(make_umap_callable(clap_model, reducer_umap))

    print_timbre_comparison(clap_res, umap_res)

    # ── Covers80 ─────────────────────────────────────────────────────────────
    if args.covers80:
        eval_covers80(args.covers80, clap_model, reducer_umap)


if __name__ == "__main__":
    main()
