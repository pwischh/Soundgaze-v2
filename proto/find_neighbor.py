"""
Pick a random track and find its closest neighbor using the CLAP + acoustic
hybrid similarity (alpha=0.5).

hybrid_sim(A, B) = 0.5 * cosine(clap_A, clap_B) + 0.5 * cosine(acoustic_A, acoustic_B)
Implemented via weighted concatenation so a single matmul cosine suffices.
"""

import numpy as np
import pandas as pd

from config import EMBEDDINGS_DIR, FMA_METADATA_DIR

# "v2" = 70-dim with density features + group weights; "v1" = original 66-dim
ACOUSTIC_VERSION = "v2"

CLAP_PATH     = EMBEDDINGS_DIR / "fma_small_clap.parquet"
ACOUSTIC_PATH = EMBEDDINGS_DIR / f"fma_small_acoustic_{ACOUSTIC_VERSION}.parquet"
SCALER_PATH   = EMBEDDINGS_DIR / f"acoustic_scaler_{ACOUSTIC_VERSION}.npz"
TRACKS_CSV    = FMA_METADATA_DIR / "tracks.csv"

ALPHA = 0.5


def normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def load_tracks() -> pd.DataFrame:
    df = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
    return pd.DataFrame({"title": df[("track", "title")], "artist": df[("artist", "name")]})


def track_label(track_id: int, meta: pd.DataFrame) -> str:
    if track_id in meta.index:
        row = meta.loc[track_id]
        return f"{row['artist']} - {row['title']}"
    return f"Track {track_id}"


def main() -> None:
    print("Loading CLAP embeddings ...")
    clap_df = pd.read_parquet(CLAP_PATH)
    clap_df = clap_df.set_index("track_id")

    print("Loading acoustic embeddings ...")
    ac_df = pd.read_parquet(ACOUSTIC_PATH)
    ac_df = ac_df.set_index("track_id")

    # Inner join — only tracks present in both parquets
    common_ids = clap_df.index.intersection(ac_df.index)
    clap_df = clap_df.loc[common_ids]
    ac_df   = ac_df.loc[common_ids]
    ids = common_ids.to_numpy()
    print(f"Corpus: {len(ids)} tracks (intersection of CLAP + acoustic)\n")

    clap_emb = np.stack(clap_df["embedding"].to_numpy()).astype(np.float32)
    ac_emb   = np.stack(ac_df["features"].to_numpy()).astype(np.float32)

    # Acoustic features are already z-scored in the parquet (from embed_acoustic.py),
    # but load scaler as a sanity check
    if SCALER_PATH.exists():
        print(f"Scaler found at {SCALER_PATH} — acoustic features already standardized.")
    else:
        print("WARNING: acoustic_scaler.npz not found — features may be unscaled.")

    # Weighted concatenation: cosine([√α·norm(clap) | √(1-α)·norm(ac)])
    # == α·cosine(clap) + (1-α)·cosine(ac)
    hybrid = np.concatenate([
        np.sqrt(ALPHA)       * normalize_rows(clap_emb),
        np.sqrt(1.0 - ALPHA) * normalize_rows(ac_emb),
    ], axis=1)

    normed = normalize_rows(hybrid)

    print("Loading track metadata ...")
    meta = load_tracks()

    # Pre-normalize component halves for breakdown display
    clap_n = normalize_rows(clap_emb)
    ac_n   = normalize_rows(ac_emb)

    rng = np.random.default_rng()
    query_idx = int(rng.integers(len(ids)))
    query_id  = int(ids[query_idx])

    sims = normed @ normed[query_idx]
    sims[query_idx] = -1

    top5_idx = np.argsort(-sims)[:5]

    print(f"\nQuery: [{query_id}] {track_label(query_id, meta)}\n")
    print(f"  {'Rank':<6} {'Track':<50} {'Hybrid':>8} {'CLAP':>8} {'Acoustic':>10}")
    print("  " + "-" * 84)
    for rank, idx in enumerate(top5_idx, 1):
        tid        = int(ids[idx])
        hybrid_cos = float(sims[idx])
        clap_cos   = float(clap_n[idx] @ clap_n[query_idx])
        ac_cos     = float(ac_n[idx]   @ ac_n[query_idx])
        label      = track_label(tid, meta)[:48]
        print(f"  {rank:<6} [{tid}] {label:<48} {hybrid_cos:>8.4f} {clap_cos:>8.4f} {ac_cos:>10.4f}")


if __name__ == "__main__":
    main()
