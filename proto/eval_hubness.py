"""
Hubness analysis for PANN and CLAP embedding spaces.

Hubness: in high-dimensional spaces, a few "hub" points appear in nearly
everyone's k-nearest-neighbors, distorting similarity search. Skewness of
the k-occurrence distribution > 3 indicates problematic hubness.

Usage:
  python eval_hubness.py
"""

import numpy as np
import pandas as pd

from config import EMBEDDINGS_DIR

PANN_PATH = EMBEDDINGS_DIR / "fma_small_embeddings.parquet"
CLAP_PATH = EMBEDDINGS_DIR / "fma_small_clap.parquet"
K = 10


def compute_koccurrence(matrix: np.ndarray, k: int) -> np.ndarray:
    normed = matrix.astype(np.float32)
    norms = np.linalg.norm(normed, axis=1, keepdims=True)
    normed = normed / np.maximum(norms, 1e-8)

    sim = normed @ normed.T                      # (N, N) float32
    np.fill_diagonal(sim, -np.inf)

    top_k = np.argpartition(-sim, k, axis=1)[:, :k]   # (N, k)
    return np.bincount(top_k.ravel(), minlength=len(matrix))


def skewness(x: np.ndarray) -> float:
    z = (x - x.mean()) / max(float(x.std()), 1e-8)
    return float(np.mean(z ** 3))


def report(name: str, kocc: np.ndarray, track_ids: np.ndarray) -> None:
    sk    = skewness(kocc)
    flag  = "  [PROBLEMATIC > 3.0]" if sk > 3.0 else ""
    top5  = np.argsort(-kocc)[:5]

    print(f"{name}")
    print(f"  Skewness:         {sk:.3f}{flag}")
    print(f"  Mean k-occ:       {kocc.mean():.2f}")
    print(f"  Max k-occ:        {int(kocc.max())}")
    print(f"  Zero-occ tracks:  {(kocc == 0).mean() * 100:.1f}%")
    top5_str = ", ".join(f"track {track_ids[i]} (k={kocc[i]})" for i in top5)
    print(f"  Top-5 hubs:       {top5_str}")
    print()


def main() -> None:
    print("Loading PANN embeddings ...")
    pann_df = pd.read_parquet(PANN_PATH)
    pann_matrix = np.stack(pann_df["embedding"].to_numpy())
    pann_ids    = pann_df["track_id"].to_numpy()

    print("Loading CLAP embeddings ...")
    clap_df = pd.read_parquet(CLAP_PATH)
    clap_matrix = np.stack(clap_df["embedding"].to_numpy())
    clap_ids    = clap_df["track_id"].to_numpy()

    print(f"\nComputing k-occurrence (k={K}) ...\n")

    for name, matrix, ids in [
        (f"PANN CNN10  ({pann_matrix.shape[1]}-dim,  n={len(pann_matrix)})", pann_matrix, pann_ids),
        (f"CLAP HTSAT  ({clap_matrix.shape[1]}-dim, n={len(clap_matrix)})", clap_matrix, clap_ids),
    ]:
        kocc = compute_koccurrence(matrix, K)
        report(name, kocc, ids)


if __name__ == "__main__":
    main()
