"""
SoundgazeV2 — FastAPI backend
Serves precomputed DR embeddings, k-NN queries, eval submissions, and metrics.

Prerequisites (run Parker's pipeline scripts first):
  data/reduced/track_ids.npy          — (N,) int array of FMA track IDs
  data/reduced/pca_3d.npy             — (N, 3) PCA-reduced coords
  data/reduced/tsne_3d.npy            — (N, 3) t-SNE-reduced coords
  data/reduced/umap_3d.npy            — (N, 3) UMAP-reduced coords
  data/raw/fma_metadata/tracks.csv    — FMA metadata (title, artist, genre)
  data/metrics.json                   — precomputed eval metrics (from evaluate.py)

Run:
  pip install -r requirements-api.txt
  uvicorn api:app --reload --port 8000
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA = Path(__file__).parent / "data"
REDUCED_DIR = DATA / "reduced"
EVAL_LOG = DATA / "eval_results.jsonl"

METHODS = ["pca", "tsne", "umap"]
METRICS_SUPPORTED = ["cosine", "euclidean", "manhattan"]

# ---------------------------------------------------------------------------
# In-memory state (populated at startup)
# ---------------------------------------------------------------------------

track_ids: np.ndarray = np.array([], dtype=int)
embeddings: dict[str, np.ndarray] = {}
normalized_xyz: dict[str, np.ndarray] = {}
metadata: dict[int, dict] = {}
knn_indexes: dict[tuple[str, str], NearestNeighbors] = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="SoundgazeV2 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_data() -> None:
    global track_ids, metadata

    # --- Track IDs ---
    ids_path = REDUCED_DIR / "track_ids.npy"
    if not ids_path.exists():
        print(f"WARNING: {ids_path} not found — run Parker's reduce.py first.")
        return
    track_ids = np.load(ids_path)

    # --- FMA metadata ---
    tracks_csv = DATA / "raw/fma_metadata/tracks.csv"
    if tracks_csv.exists():
        df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
        for tid in track_ids:
            tid = int(tid)
            row = df.loc[tid] if tid in df.index else None
            metadata[tid] = {
                "title":  str(row[("track",  "title")])    if row is not None else f"Track {tid}",
                "artist": str(row[("artist", "name")])     if row is not None else "Unknown",
                "genre":  str(row[("track",  "genre_top")]) if row is not None else "Unknown",
            }
    else:
        print(f"WARNING: {tracks_csv} not found — using placeholder metadata.")
        for tid in track_ids:
            metadata[int(tid)] = {"title": f"Track {int(tid)}", "artist": "Unknown", "genre": "Unknown"}

    # --- Reduced embeddings + k-NN indexes ---
    for method in METHODS:
        path = REDUCED_DIR / f"{method}_3d.npy"
        if not path.exists():
            print(f"WARNING: {path} not found — skipping {method}.")
            continue

        emb = np.load(path)                        # (N, 3)
        embeddings[method] = emb

        # Normalize to [0, 1] once at load time — frontend centers to [-0.5, 0.5]
        lo, hi = emb.min(axis=0), emb.max(axis=0)
        normalized_xyz[method] = (emb - lo) / np.maximum(hi - lo, 1e-8)

        for metric in METRICS_SUPPORTED:
            knn = NearestNeighbors(metric=metric, algorithm="brute")
            knn.fit(emb)
            knn_indexes[(method, metric)] = knn

    loaded = [m for m in METHODS if m in embeddings]
    print(f"Loaded {len(track_ids)} tracks · methods: {loaded}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track_record(i: int, method: str) -> dict:
    tid = int(track_ids[i])
    return {
        "track_id": tid,
        "xyz":      normalized_xyz[method][i].tolist(),
        **metadata.get(tid, {"title": f"Track {tid}", "artist": "Unknown", "genre": "Unknown"}),
    }


def _validate(method: str, metric: str) -> None:
    if method not in embeddings:
        raise HTTPException(400, f"method '{method}' not available — run reduce.py first")
    if metric not in METRICS_SUPPORTED:
        raise HTTPException(400, f"metric must be one of {METRICS_SUPPORTED}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "loaded_methods": list(embeddings.keys()),
        "track_count": len(track_ids),
    }


@app.get("/points")
def get_points(method: str = "umap") -> dict:
    """Return all 3D track points for a given DR method."""
    _validate(method, "cosine")
    tracks = [_track_record(i, method) for i in range(len(track_ids))]
    return {"tracks": tracks}


@app.get("/similar")
def get_similar(
    track_id: int,
    method: str = "umap",
    metric: str = "cosine",
    k: int = 10,
) -> dict:
    """Return k nearest neighbors for a track under a given method + metric."""
    _validate(method, metric)

    idx_arr = np.where(track_ids == track_id)[0]
    if len(idx_arr) == 0:
        raise HTTPException(404, f"track_id {track_id} not found")
    row = int(idx_arr[0])

    knn = knn_indexes[(method, metric)]
    query = embeddings[method][row].reshape(1, -1)
    _, indices = knn.kneighbors(query, n_neighbors=k + 1)

    neighbors = [
        _track_record(int(i), method)
        for i in indices[0]
        if int(track_ids[i]) != track_id
    ][:k]
    return {"neighbors": neighbors}


@app.get("/metrics")
def get_metrics() -> dict:
    """Return precomputed quantitative metrics for all DR methods."""
    path = DATA / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json not found — run evaluate.py first")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Human evaluation
# ---------------------------------------------------------------------------

class EvalRating(BaseModel):
    track_id: int
    score: Literal[0, 1]

class EvalPayload(BaseModel):
    evaluator_name: str
    seed_track_id:  int
    method:         str
    metric:         str
    k:              int
    timestamp:      str
    ratings:        list[EvalRating]

@app.post("/eval/submit")
def submit_eval(payload: EvalPayload) -> dict:
    """Append one human evaluation session to eval_results.jsonl."""
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG, "a") as f:
        f.write(payload.model_dump_json() + "\n")
    return {"ok": True}

@app.get("/eval/results")
def get_eval_results() -> dict:
    """Return all submitted evaluation sessions (for analysis)."""
    if not EVAL_LOG.exists():
        return {"results": []}
    results = [json.loads(line) for line in EVAL_LOG.read_text().splitlines() if line.strip()]
    return {"results": results, "count": len(results)}
