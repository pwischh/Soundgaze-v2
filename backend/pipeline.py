"""
Data pipeline for Soundgaze-v2.

Generates any missing artifacts and loads everything into module-level state.
Can be run standalone to pre-generate all artifacts before starting the server:

    python -m backend.pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa

from .config import (
    CLAP_PATH, ACOUSTIC_PATH, SCALER_PATH, REDUCED_DIR, META_DIR,
    FMA_SMALL_DIR, CLAP_SR, LIBROSA_SR, DURATION, CLAP_LEN, CLAP_BATCH, METHODS,
)
from .features import acoustic_features, normalize_rows, GROUP_WEIGHTS

# ---------------------------------------------------------------------------
# Module-level state (populated by load_state)
# ---------------------------------------------------------------------------

track_ids:  np.ndarray = np.array([], dtype=np.int64)
hybrid_emb: np.ndarray = np.empty((0, 582), dtype=np.float32)
xyz:        dict[str, np.ndarray] = {}
metadata:   dict[int, dict]       = {}
knn_index                         = None  # sklearn NearestNeighbors


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_clap(model) -> None:
    print("Generating CLAP embeddings ...")
    CLAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    mp3_files = sorted(FMA_SMALL_DIR.rglob("*.mp3"))
    print(f"  Found {len(mp3_files)} tracks.")

    records: list[dict]       = []
    batch_audio: list[np.ndarray] = []
    batch_ids:   list[int]        = []

    def _flush() -> None:
        if not batch_audio:
            return
        x    = np.stack(batch_audio)
        embs = model.get_audio_embedding_from_data(x=x, use_tensor=False)
        for tid, emb in zip(batch_ids, embs):
            records.append({"track_id": tid, "embedding": emb.tolist()})
        batch_audio.clear()
        batch_ids.clear()

    def _fix_length(a: np.ndarray) -> np.ndarray:
        return a[:CLAP_LEN] if len(a) >= CLAP_LEN else np.pad(a, (0, CLAP_LEN - len(a)))

    for i, path in enumerate(mp3_files):
        try:
            audio, _ = librosa.load(path, sr=CLAP_SR, mono=True, duration=DURATION)
            batch_audio.append(_fix_length(audio))
            batch_ids.append(int(path.stem))
        except Exception:
            continue
        if len(batch_audio) >= CLAP_BATCH:
            _flush()
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(mp3_files)} ...")

    _flush()
    pd.DataFrame(records).to_parquet(CLAP_PATH, index=False)
    print(f"  Saved {len(records)} embeddings → {CLAP_PATH}")


def generate_acoustic() -> None:
    print("Generating acoustic v2 embeddings ...")
    ACOUSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    mp3_files = sorted(FMA_SMALL_DIR.rglob("*.mp3"))
    print(f"  Found {len(mp3_files)} tracks.")

    tid_list:  list[int]        = []
    raw_feats: list[np.ndarray] = []

    for i, path in enumerate(mp3_files):
        try:
            audio, _ = librosa.load(path, sr=LIBROSA_SR, mono=True, duration=DURATION)
            if len(audio) < LIBROSA_SR:
                continue
            tid_list.append(int(path.stem))
            raw_feats.append(acoustic_features(audio, LIBROSA_SR))
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(mp3_files)} ...")

    matrix = np.stack(raw_feats)
    mu     = matrix.mean(axis=0)
    sigma  = matrix.std(axis=0).clip(min=1e-8)
    standardized = (matrix - mu) / sigma * GROUP_WEIGHTS

    np.savez(SCALER_PATH, mean=mu, std=sigma, weights=GROUP_WEIGHTS)
    records = [{"track_id": t, "features": f.tolist()} for t, f in zip(tid_list, standardized)]
    pd.DataFrame(records).to_parquet(ACOUSTIC_PATH, index=False)
    print(f"  Saved {len(records)} embeddings → {ACOUSTIC_PATH}")


def generate_reductions(hybrid: np.ndarray, ids: np.ndarray) -> None:
    import umap as umap_lib
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    REDUCED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REDUCED_DIR / "track_ids.npy", ids)

    if not (REDUCED_DIR / "pca_3d.npy").exists():
        print("Fitting PCA (3-dim) ...")
        emb = PCA(n_components=3, random_state=42).fit_transform(hybrid)
        np.save(REDUCED_DIR / "pca_3d.npy", emb.astype(np.float32))

    if not (REDUCED_DIR / "umap_3d.npy").exists():
        print("Fitting UMAP (cosine, 3-dim) ...")
        emb = umap_lib.UMAP(
            n_components=3, metric="cosine", random_state=42, verbose=False
        ).fit_transform(hybrid)
        np.save(REDUCED_DIR / "umap_3d.npy", emb.astype(np.float32))

    if not (REDUCED_DIR / "tsne_3d.npy").exists():
        print("Fitting t-SNE (cosine, 3-dim) — may take several minutes ...")
        emb = TSNE(
            n_components=3, metric="cosine", random_state=42, n_jobs=-1, perplexity=30,
        ).fit_transform(hybrid)
        np.save(REDUCED_DIR / "tsne_3d.npy", emb.astype(np.float32))


# ---------------------------------------------------------------------------
# State loader (called by api.py lifespan and __main__)
# ---------------------------------------------------------------------------

def load_state() -> None:
    global track_ids, hybrid_emb, knn_index
    from sklearn.neighbors import NearestNeighbors

    # ── CLAP ──────────────────────────────────────────────────────────────────
    if not CLAP_PATH.exists():
        import laion_clap
        print("Loading CLAP model ...")
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        model.load_ckpt()
        generate_clap(model)
    else:
        print(f"CLAP found: {CLAP_PATH}")

    # ── Acoustic v2 ───────────────────────────────────────────────────────────
    if not ACOUSTIC_PATH.exists():
        generate_acoustic()
    else:
        print(f"Acoustic v2 found: {ACOUSTIC_PATH}")

    # ── Build hybrid matrix ───────────────────────────────────────────────────
    print("Loading embeddings ...")
    clap_df = pd.read_parquet(CLAP_PATH).set_index("track_id")
    ac_df   = pd.read_parquet(ACOUSTIC_PATH).set_index("track_id")

    common  = clap_df.index.intersection(ac_df.index)
    track_ids = common.to_numpy(dtype=np.int64)

    clap_mat = np.stack(clap_df.loc[common]["embedding"].to_numpy()).astype(np.float32)
    ac_mat   = np.stack(ac_df.loc[common]["features"].to_numpy()).astype(np.float32)

    hybrid_emb = np.concatenate([
        np.sqrt(0.5) * normalize_rows(clap_mat),
        np.sqrt(0.5) * normalize_rows(ac_mat),
    ], axis=1)  # (N, 582)
    print(f"Corpus: {len(track_ids)} tracks, hybrid dim={hybrid_emb.shape[1]}")

    # ── 3D reductions ─────────────────────────────────────────────────────────
    if any(not (REDUCED_DIR / f"{m}_3d.npy").exists() for m in METHODS):
        generate_reductions(hybrid_emb, track_ids)

    for method in METHODS:
        path = REDUCED_DIR / f"{method}_3d.npy"
        if path.exists():
            emb   = np.load(path)
            lo, hi = emb.min(axis=0), emb.max(axis=0)
            xyz[method] = (emb - lo) / np.maximum(hi - lo, 1e-8)

    print(f"3D methods loaded: {list(xyz.keys())}")

    # ── k-NN index ────────────────────────────────────────────────────────────
    print("Building k-NN index ...")
    knn_index = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_index.fit(hybrid_emb)

    # ── FMA metadata ──────────────────────────────────────────────────────────
    tracks_csv = META_DIR / "tracks.csv"
    if tracks_csv.exists():
        df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
        for tid in track_ids:
            tid = int(tid)
            if tid in df.index:
                row = df.loc[tid]
                metadata[tid] = {
                    "title":  str(row[("track",  "title")]),
                    "artist": str(row[("artist", "name")]),
                    "genre":  str(row[("track",  "genre_top")]),
                }
            else:
                metadata[tid] = {"title": f"Track {tid}", "artist": "Unknown", "genre": "Unknown"}
    else:
        for tid in track_ids:
            metadata[int(tid)] = {"title": f"Track {int(tid)}", "artist": "Unknown", "genre": "Unknown"}

    print("Ready.")


if __name__ == "__main__":
    load_state()
