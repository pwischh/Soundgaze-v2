"""
Pick a random track from the embeddings and find its closest neighbor
by cosine similarity. Prints both track names.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import EMBEDDINGS_DIR, FMA_METADATA_DIR

EMBEDDINGS_PATH = EMBEDDINGS_DIR / "fma_small_embeddings.parquet"
TRACKS_CSV = FMA_METADATA_DIR / "tracks.csv"


def load_tracks() -> pd.DataFrame:
    # tracks.csv has 2 header rows forming a MultiIndex
    df = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
    # Flatten to just the columns we need
    titles = df[("track", "title")]
    artists = df[("artist", "name")]
    return pd.DataFrame({"title": titles, "artist": artists})


def track_label(track_id: int, meta: pd.DataFrame) -> str:
    if track_id in meta.index:
        row = meta.loc[track_id]
        return f"{row['artist']} - {row['title']}"
    return f"Track {track_id}"


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    return normalized @ normalized.T


def main() -> None:
    print("Loading embeddings ...")
    df = pd.read_parquet(EMBEDDINGS_PATH)
    track_ids = df["track_id"].to_numpy()
    embeddings = np.stack(df["embedding"].to_numpy())

    print("Loading track metadata ...")
    meta = load_tracks()

    rng = np.random.default_rng()
    query_idx = rng.integers(len(track_ids))
    query_id = track_ids[query_idx]

    # Cosine similarity of query against all tracks
    query_vec = embeddings[query_idx]
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    sims = (embeddings @ query_vec) / np.maximum(norms * query_norm, 1e-8)

    # Mask out the query itself and pick highest similarity
    sims[query_idx] = -1
    neighbor_idx = int(np.argmax(sims))
    neighbor_id = track_ids[neighbor_idx]

    print(f"\nQuery:    [{query_id}] {track_label(query_id, meta)}")
    print(f"Neighbor: [{neighbor_id}] {track_label(neighbor_id, meta)}  (sim={sims[neighbor_idx]:.4f})")


if __name__ == "__main__":
    main()
