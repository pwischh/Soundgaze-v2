"""
Soundgaze-v2 FastAPI backend.

Run from the project root:
    uvicorn backend.api:app --reload --port 8000

On first run, missing artifacts are generated automatically (takes ~hours on CPU).
Subsequent starts load from disk and are fast.
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import pipeline
from .config import METHODS


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load_state()
    yield


app = FastAPI(title="Soundgaze-v2", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track_record(idx: int, method: str) -> dict:
    tid = int(pipeline.track_ids[idx])
    return {
        "track_id": tid,
        "xyz": pipeline.xyz[method][idx].tolist(),
        **pipeline.metadata.get(tid, {"title": f"Track {tid}", "artist": "Unknown", "genre": "Unknown"}),
    }


def _require_method(method: str) -> None:
    if method not in pipeline.xyz:
        raise HTTPException(400, f"method '{method}' not available; choose from {list(pipeline.xyz.keys())}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "loaded_methods": list(pipeline.xyz.keys()),
        "track_count": int(len(pipeline.track_ids)),
    }


@app.get("/points")
def get_points(method: str = "umap") -> dict:
    _require_method(method)
    return {"tracks": [_track_record(i, method) for i in range(len(pipeline.track_ids))]}


@app.get("/similar")
def get_similar(track_id: int, method: str = "umap", k: int = 10) -> dict:
    _require_method(method)
    if pipeline.knn_index is None:
        raise HTTPException(503, "k-NN index not ready")

    idx_arr = np.where(pipeline.track_ids == track_id)[0]
    if len(idx_arr) == 0:
        raise HTTPException(404, f"track_id {track_id} not found")
    row = int(idx_arr[0])

    dists, indices = pipeline.knn_index.kneighbors(
        pipeline.hybrid_emb[row].reshape(1, -1), n_neighbors=k + 1
    )

    neighbors = []
    for dist, idx in zip(dists[0], indices[0]):
        if int(pipeline.track_ids[idx]) == track_id:
            continue
        rec = _track_record(int(idx), method)
        rec["score"] = float(1.0 - dist)  # cosine distance → similarity
        neighbors.append(rec)
        if len(neighbors) == k:
            break

    return {"neighbors": neighbors}
