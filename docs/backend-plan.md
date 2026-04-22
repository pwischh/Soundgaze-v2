# Soundgaze-v2 Backend Plan

## Overview

Lightweight FastAPI backend — no database, no Docker. On first run the backend generates all needed artifacts from the raw FMA Small audio files and caches them to disk. Subsequent starts are fast (just load cached files into memory).

---

## Data Pipeline (startup, lazy generation)

```
FMA Small MP3s
    ├── CLAP inference (laion-clap, HTSAT-tiny, 48kHz, 30s)
    │       └── data/embeddings/fma_small_clap.parquet      [track_id, embedding (512-dim)]
    │
    └── Acoustic extraction (librosa, 22050Hz, 30s) + z-score standardization
            └── data/embeddings/fma_small_acoustic.parquet  [track_id, features (70-dim)]
                data/embeddings/acoustic_scaler.npz         [mean (66,), std (66,)]

    Hybrid vectors (α=0.5, built in memory at startup):
        hybrid = [√0.5 · norm(clap) | √0.5 · norm(acoustic)]  →  (582-dim)
        └── k-NN index built on hybrid vectors (cosine)

    3D reductions (run on CLAP-512 embeddings):
        ├── UMAP (cosine, 3D)  →  data/reduced/umap_3d.npy   (N, 3)
        ├── t-SNE (cosine, 3D) →  data/reduced/tsne_3d.npy   (N, 3)
        └── PCA  (3D)          →  data/reduced/pca_3d.npy    (N, 3)

        track_ids.npy (N,) saved alongside reduced files
```

Each artifact is only generated if its file does not already exist. Logic is adapted from `proto/embed_clap.py`, `proto/embed_acoustic.py`, and `proto/eval_reduction.py` — no imports from proto at runtime.

---

## In-Memory State (after startup)

| Variable | Type | Description |
|---|---|---|
| `track_ids` | `np.ndarray (N,)` | FMA track ID for each row |
| `hybrid_emb` | `np.ndarray (N, 582)` | Concatenated `[√0.5·norm(clap) \| √0.5·norm(acoustic)]` vectors |
| `xyz` | `dict[method → np.ndarray (N,3)]` | [0,1]-normalized 3D coords per reduction method (per-axis min-max) |
| `metadata` | `dict[track_id → {title, artist, genre}]` | From `fma_metadata/tracks.csv` |
| `knn_index` | `NearestNeighbors` | Single brute-force cosine index on `hybrid_emb` |

---

## Endpoints

### `GET /health`
Returns server status, loaded methods, and track count.

```json
{ "status": "ok", "loaded_methods": ["umap", "tsne", "pca"], "track_count": 8000 }
```

---

### `GET /points?method=umap`
Returns all N tracks with 3D coordinates for the given reduction method. Intended to be called once per method and cached client-side.

**Query params:** `method` — `umap` | `tsne` | `pca` (default: `umap`)

**Response:**
```json
{
  "tracks": [
    { "track_id": 2, "title": "...", "artist": "...", "genre": "...", "xyz": [0.41, 0.72, 0.18] },
    ...
  ]
}
```

---

### `GET /similar?track_id=2&method=umap&k=10`
Returns k nearest neighbors in **high-dimensional CLAP space** (512-dim cosine). Neighbors are the same regardless of method — only the returned `xyz` values differ.

**Query params:**
- `track_id` — FMA track ID
- `method` — `umap` | `tsne` | `pca` (determines which xyz to attach to results)
- `k` — number of neighbors (default: 10)

**Response:**
```json
{
  "neighbors": [
    { "track_id": 55, "title": "...", "artist": "...", "genre": "...", "xyz": [0.38, 0.69, 0.21], "score": 0.93 },
    ...
  ]
}
```

`score` is cosine similarity in hybrid space (higher = more similar).

---

## Frontend Data Flow

```
App loads
  └── GET /points?method=umap → cache all N points for umap
  └── (lazy) GET /points?method=tsne  when user switches to t-SNE
  └── (lazy) GET /points?method=pca   when user switches to PCA

User clicks a point
  └── GET /similar?track_id=N&method=umap&k=10
       └── neighbors returned with xyz → highlight immediately (no cache lookup needed)

User switches method (with track selected)
  └── if new method's points already cached:
        re-use xyz from cache → no API call needed
  └── else:
        GET /points?method=new + GET /similar?...&method=new in parallel
```

---

## Project Structure

All backend code lives in a new `backend/` directory at the project root:

```
Soundgaze-v2/
├── backend/
│   ├── api.py               # FastAPI app — startup pipeline + all endpoints
│   ├── requirements.txt     # Backend dependencies
│   ├── .venv/               # Virtual environment (not committed)
│   └── data/                # Symlink or copy of ../data (shared with proto pipeline)
├── frontend/                # Next.js app (unchanged structure)
├── proto/                   # Research scripts (reference only, not imported)
└── docs/
```

## Setup & Running

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

First run will generate CLAP embeddings and 3D reductions — this takes several hours on CPU. Subsequent starts load from disk and are fast.

## Files

| File | Action |
|---|---|
| `backend/api.py` | Create — self-contained startup pipeline + all endpoints |
| `backend/requirements.txt` | Create — `fastapi`, `uvicorn`, `numpy`, `pandas`, `scikit-learn`, `pyarrow`, `laion-clap`, `umap-learn`, `librosa` |
| `frontend/app/lib/api.ts` | Flip `USE_PLACEHOLDERS = false`; add `method` param to `fetchSimilar`; remove eval functions |
| `frontend/` (EvalModal, eval routes) | Remove eval UI components |

---

## Key Design Decisions

**Hybrid similarity (α=0.5).** Benchmarking showed the CLAP + acoustic hybrid at α=0.5 outperforms pure CLAP across all timbremetrics (Spearman 0.4796 vs 0.4455, +7.7%). The weighted concatenation trick `[√α·norm(clap) | √(1-α)·norm(acoustic)]` means cosine similarity on the combined vector exactly equals `α·cosine(clap) + (1-α)·cosine(acoustic)` — no special distance function needed.

**Similarity is method-agnostic.** The hybrid 582-dim vector determines neighbors. The reduction method only controls the visual layout. Switching UMAP/t-SNE/PCA changes how the space *looks* but not which tracks are considered similar — consistent with our evaluations (3D k-NN gave 0% Top-1; hybrid 582-dim gives better-than-CLAP performance).

**Single k-NN index.** One `NearestNeighbors(metric='cosine', algorithm='brute')` index on `hybrid_emb` serves all similarity queries. No per-method indexes needed.

**xyz returned in /similar.** Avoids a client-side cache lookup or a second round trip. Neighbors are immediately placeable in the 3D view on first click.

**No database, no Docker.** All state is in-memory numpy arrays. Artifacts persist to disk as `.npy` / `.parquet` files.
