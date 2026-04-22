"""
Evaluate 3D dimensionality reduction of the hybrid CLAP + acoustic embedding (582-dim, α=0.5).

Reductions tested: UMAP, t-SNE, PCA (all → 3D, fit on hybrid-582 corpus)

Metrics:
  KNN Recall        — fraction of hybrid-582 k-NN preserved in 3D projection (k=10)
  Trustworthiness   — fraction of 3D k-NN truly close in hybrid-582 space
  Hubness           — k-occurrence skewness (k=10)
  timbremetrics     — Spearman ρ, Kendall τ, Triplet, NDCG, MAE
                      (Hybrid-582, UMAP-3D, PCA-3D; t-SNE excluded — no transform())
  Covers80          — Top-1, Top-5, MRR  (requires --covers80)

3D plots: matplotlib scatter, 2000-track sample, colored by genre_top

Usage:
  python eval_hybrid_3d.py [--covers80 /path/to/covers80] [--skip-tsne] [--no-plot]
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import librosa
from scipy.stats import entropy as scipy_entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

import laion_clap
import umap as umap_lib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from timbremetrics import TimbreMetric

from config import EMBEDDINGS_DIR, FMA_METADATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAP_SR   = 48000
TIMBRE_SR = 44100
LIBROSA_SR = 22050
DURATION  = 30.0
ALPHA     = 0.5
K         = 10

# "v2" = 70-dim with density features + group weights; "v1" = original 66-dim
ACOUSTIC_VERSION = "v2"

GENRE_COLORS = {
    "Hip-Hop":       "#e6194b",
    "Pop":           "#f58231",
    "Folk":          "#3cb44b",
    "Experimental":  "#911eb4",
    "Rock":          "#4363d8",
    "International": "#42d4f4",
    "Electronic":    "#f032e6",
    "Instrumental":  "#bfef45",
}
DEFAULT_COLOR = "#aaaaaa"

# ---------------------------------------------------------------------------
# Module-level caches and scaler for timbremetrics callables
# ---------------------------------------------------------------------------

_clap_cache:      dict[bytes, np.ndarray] = {}
_acoustic_cache:  dict[bytes, np.ndarray] = {}
_scaler_mean:     np.ndarray | None       = None
_scaler_std:      np.ndarray | None       = None
_scaler_weights:  np.ndarray | None       = None


def standardize(v: np.ndarray) -> np.ndarray:
    if _scaler_mean is None:
        return v
    z = (v - _scaler_mean) / _scaler_std
    if _scaler_weights is not None:
        z = z * _scaler_weights
    return z


# ---------------------------------------------------------------------------
# Feature helpers  (match embed_acoustic.py / eval_hybrid.py exactly)
# ---------------------------------------------------------------------------

def acoustic_features(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != LIBROSA_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=LIBROSA_SR)
        sr = LIBROSA_SR
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))

    # Compute STFT once; reuse for all downstream features (matches embed_acoustic.py).
    D      = librosa.stft(audio)
    S      = np.abs(D)
    mel    = librosa.feature.melspectrogram(S=S ** 2, sr=sr)
    mel_db = librosa.power_to_db(mel)

    mfcc     = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    chroma   = librosa.feature.chroma_stft(S=S, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)

    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr)
    tempo_arr = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]

    spec_centroid  = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    spec_rolloff   = float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean())
    spec_flatness  = float(librosa.feature.spectral_flatness(S=S).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean())

    power_norm   = (S ** 2) / ((S ** 2).sum(axis=0, keepdims=True) + 1e-10)
    spec_entropy = float(scipy_entropy(power_norm.mean(axis=1) + 1e-10))
    onsets       = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate   = len(onsets) / max(len(audio) / sr, 1e-3)
    D_h, D_p     = librosa.decompose.hpss(D)
    rms_h = float(librosa.feature.rms(S=np.abs(D_h)).mean())
    rms_p = float(librosa.feature.rms(S=np.abs(D_p)).mean())
    hnr   = np.log10(max(rms_h, 1e-6) / max(rms_p, 1e-6))

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1),
        [spec_centroid, spec_rolloff, spec_flatness],
        contrast.mean(axis=1),
        [float(np.atleast_1d(tempo_arr)[0]),
         float(onset_env.mean()),
         float(librosa.feature.zero_crossing_rate(audio).mean()),
         float(np.log10(max(librosa.feature.rms(S=S).mean(), 1e-6)))],
        [spec_entropy, spec_bandwidth, onset_rate, hnr],
    ]).astype(np.float32)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-8)


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def build_hybrid(clap_mat: np.ndarray, ac_mat: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    return np.concatenate([
        np.sqrt(alpha)       * normalize_rows(clap_mat),
        np.sqrt(1.0 - alpha) * normalize_rows(ac_mat),
    ], axis=1)


# ---------------------------------------------------------------------------
# KNN recall + trustworthiness  (from eval_reduction.py)
# ---------------------------------------------------------------------------

def knn_indices(matrix: np.ndarray, k: int) -> np.ndarray:
    normed = normalize_rows(matrix.astype(np.float32))
    sim = normed @ normed.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k, axis=1)[:, :k]


def knn_recall(orig: np.ndarray, reduced: np.ndarray, k: int) -> float:
    nn_orig    = knn_indices(orig,    k)
    nn_reduced = knn_indices(reduced, k)
    return float(np.mean([
        len(set(nn_orig[i]) & set(nn_reduced[i])) / k
        for i in range(len(orig))
    ]))


def projection_report(name: str, orig: np.ndarray, reduced: np.ndarray) -> tuple[float, float]:
    recall = knn_recall(orig, reduced, K)
    tw = trustworthiness(orig, reduced, n_neighbors=K, metric="cosine")
    return recall, tw


# ---------------------------------------------------------------------------
# Hubness  (from eval_reduction.py)
# ---------------------------------------------------------------------------

def compute_koccurrence(matrix: np.ndarray, k: int) -> np.ndarray:
    normed = normalize_rows(matrix.astype(np.float32))
    sim = normed @ normed.T
    np.fill_diagonal(sim, -np.inf)
    top_k = np.argpartition(-sim, k, axis=1)[:, :k]
    return np.bincount(top_k.ravel(), minlength=len(matrix))


def skewness(x: np.ndarray) -> float:
    return float(np.mean(((x - x.mean()) / x.std()) ** 3))


def hubness_report(name: str, matrix: np.ndarray) -> tuple[float, int, float]:
    kocc = compute_koccurrence(matrix, K)
    return skewness(kocc), int(kocc.max()), float((kocc == 0).mean())


# ---------------------------------------------------------------------------
# timbremetrics callables
# ---------------------------------------------------------------------------

def make_hybrid_callable(clap_model):
    def fn(x: torch.Tensor) -> torch.Tensor:
        key = x.numpy().tobytes()
        if key not in _clap_cache:
            audio_clap = librosa.resample(x.squeeze(0).numpy(), orig_sr=TIMBRE_SR, target_sr=CLAP_SR)
            _clap_cache[key] = clap_model.get_audio_embedding_from_data(
                x=audio_clap.reshape(1, -1).astype(np.float32), use_tensor=False
            )[0]
            _acoustic_cache[key] = standardize(acoustic_features(x.squeeze(0).numpy(), TIMBRE_SR))
        clap_n = normalize_vec(_clap_cache[key])
        ac_n   = normalize_vec(_acoustic_cache[key])
        combined = np.concatenate([np.sqrt(ALPHA) * clap_n, np.sqrt(1.0 - ALPHA) * ac_n])
        return torch.from_numpy(combined.astype(np.float32))
    return fn


def make_umap_callable(clap_model, reducer):
    hybrid_fn = make_hybrid_callable(clap_model)
    def fn(x: torch.Tensor) -> torch.Tensor:
        h = hybrid_fn(x).numpy().reshape(1, -1)
        return torch.from_numpy(reducer.transform(h)[0].astype(np.float32))
    return fn


def make_pca_callable(clap_model, reducer):
    hybrid_fn = make_hybrid_callable(clap_model)
    def fn(x: torch.Tensor) -> torch.Tensor:
        h = hybrid_fn(x).numpy().reshape(1, -1)
        return torch.from_numpy(reducer.transform(h)[0].astype(np.float32))
    return fn


def format_timbre_table(hybrid_res: dict, umap_res: dict, pca_res: dict) -> str:
    lines = [f"{'Metric':<40} {'Hybrid-582':>12} {'UMAP-3D':>10} {'PCA-3D':>10}", "-" * 74]
    for metric in ["spearman_corr", "kendall_corr", "triplet_agreement", "ndcg_retrieve_sim", "mae"]:
        label = metric.replace("_corr", "").replace("_", " ")
        h = float(hybrid_res.get("cosine", {}).get(metric, float("nan")))
        u = float(umap_res.get("cosine",  {}).get(metric, float("nan")))
        p = float(pca_res.get("cosine",   {}).get(metric, float("nan")))
        lines.append(f"  {label:<38} {h:>12.4f} {u:>10.4f} {p:>10.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Covers80
# ---------------------------------------------------------------------------

def cosine_row(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    qnorm = np.linalg.norm(query)
    return (matrix @ query) / np.maximum(norms * qnorm, 1e-8)


def rank_of_partner(sims: np.ndarray, qi: int, pi: int) -> int:
    masked = sims.copy()
    masked[qi] = -np.inf
    return int(np.where(np.argsort(-masked) == pi)[0][0]) + 1


def covers80_report(name: str, matrix: np.ndarray, queries: list[tuple[int, int]]) -> tuple[float, float, float]:
    ranks = [rank_of_partner(cosine_row(matrix, matrix[qi]), qi, pi) for qi, pi in queries]
    n    = len(ranks)
    top1 = sum(r == 1 for r in ranks) / n
    top5 = sum(r <= 5 for r in ranks) / n
    mrr  = float(np.mean([1.0 / r for r in ranks]))
    return top1, top5, mrr


def eval_covers80(covers_dir: Path, clap_model, reducer_umap, reducer_pca) -> None:
    covers32k = covers_dir / "covers32k"
    pairs: list[tuple[Path, Path]] = []
    for song_dir in sorted(covers32k.iterdir()):
        if not song_dir.is_dir():
            continue
        mp3s = sorted(song_dir.glob("*.mp3"))
        if len(mp3s) == 2:
            pairs.append((mp3s[0], mp3s[1]))

    all_paths   = [p for pair in pairs for p in pair]
    path_to_idx = {p: i for i, p in enumerate(all_paths)}
    print(f"  {len(pairs)} pairs, {len(all_paths)} tracks")

    clap_embs: list[np.ndarray] = []
    ac_embs:   list[np.ndarray] = []
    valid:     list[bool]       = []

    for i, path in enumerate(all_paths):
        print(f"  [{i+1}/{len(all_paths)}] {path.name}", end="\r")
        try:
            audio_clap, _ = librosa.load(path, sr=CLAP_SR,    mono=True, duration=DURATION)
            audio_ac,   _ = librosa.load(path, sr=LIBROSA_SR, mono=True, duration=DURATION)
            clap_e = clap_model.get_audio_embedding_from_data(
                x=audio_clap.reshape(1, -1).astype(np.float32), use_tensor=False
            )[0]
            ac_e = standardize(acoustic_features(audio_ac, LIBROSA_SR))
            clap_embs.append(clap_e)
            ac_embs.append(ac_e)
            valid.append(True)
        except Exception:
            clap_embs.append(np.zeros(512, dtype=np.float32))
            ac_embs.append(np.zeros(66,  dtype=np.float32))
            valid.append(False)
    print()

    clap_mat   = np.stack(clap_embs)
    ac_mat     = np.stack(ac_embs)
    hybrid_mat = build_hybrid(clap_mat, ac_mat)
    umap_mat   = reducer_umap.transform(hybrid_mat)
    pca_mat    = reducer_pca.transform(hybrid_mat)

    print("  Fitting t-SNE on covers80 tracks ...")
    tsne_covers = TSNE(n_components=3, metric="cosine", random_state=42,
                       perplexity=min(30, len(hybrid_mat) // 4), n_jobs=-1)
    tsne_mat = tsne_covers.fit_transform(hybrid_mat)

    queries = []
    for a, b in pairs:
        ia, ib = path_to_idx[a], path_to_idx[b]
        if valid[ia] and valid[ib]:
            queries.append((ia, ib))
            queries.append((ib, ia))
    print(f"  {len(queries)} queries ({len(queries)//2} valid pairs)")

    return {
        "Hybrid (582-dim)": covers80_report("Hybrid (582-dim)", hybrid_mat, queries),
        "UMAP (3-dim)":     covers80_report("UMAP (3-dim)",     umap_mat,   queries),
        "PCA (3-dim)":      covers80_report("PCA (3-dim)",      pca_mat,    queries),
        "t-SNE (3-dim)":    covers80_report("t-SNE (3-dim)",    tsne_mat,   queries),
    }


# ---------------------------------------------------------------------------
# 3D scatter plots
# ---------------------------------------------------------------------------

def make_plots(
    umap_emb: np.ndarray,
    tsne_emb: np.ndarray | None,
    pca_emb:  np.ndarray,
    ids:      np.ndarray,
    genre_map: dict,
    n_sample: int = 2000,
) -> None:
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(ids), min(n_sample, len(ids)), replace=False)

    genres  = [genre_map.get(int(ids[i]), "Unknown") for i in sample_idx]
    colors  = [GENRE_COLORS.get(g, DEFAULT_COLOR) for g in genres]

    panels = [("UMAP", umap_emb)]
    if tsne_emb is not None:
        panels.append(("t-SNE", tsne_emb))
    panels.append(("PCA", pca_emb))

    fig = plt.figure(figsize=(7 * len(panels), 6))
    for i, (name, emb) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        pts = emb[sample_idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colors, s=4, alpha=0.6, linewidths=0)
        ax.set_title(f"Hybrid 582→3D  [{name}]", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=8, label=g)
        for g, c in GENRE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = Path(__file__).parent / "hybrid_3d_projections.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--covers80",  type=Path, default=None)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--no-plot",   action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Load acoustic scaler ───────────────────────────────────────────────
    global _scaler_mean, _scaler_std, _scaler_weights
    scaler_path = EMBEDDINGS_DIR / f"acoustic_scaler_{ACOUSTIC_VERSION}.npz"
    if scaler_path.exists():
        scaler = np.load(scaler_path)
        _scaler_mean    = scaler["mean"]
        _scaler_std     = scaler["std"]
        _scaler_weights = scaler["weights"] if "weights" in scaler else None
        print(f"Loaded acoustic scaler from {scaler_path}")
    else:
        print(f"WARNING: {scaler_path} not found — acoustic features will be unscaled.")

    # ── Load parquets ──────────────────────────────────────────────────────
    print("Loading CLAP embeddings ...")
    clap_df = pd.read_parquet(EMBEDDINGS_DIR / "fma_small_clap.parquet").set_index("track_id")
    print("Loading acoustic embeddings ...")
    ac_df   = pd.read_parquet(EMBEDDINGS_DIR / f"fma_small_acoustic_{ACOUSTIC_VERSION}.parquet").set_index("track_id")

    ids = clap_df.index.intersection(ac_df.index)
    clap_emb = np.stack(clap_df.loc[ids]["embedding"].to_numpy()).astype(np.float32)
    ac_emb   = np.stack(ac_df.loc[ids]["features"].to_numpy()).astype(np.float32)
    print(f"  {len(ids)} tracks (intersection)\n")

    hybrid = build_hybrid(clap_emb, ac_emb)   # (N, 582)

    # ── Genre metadata for plots ───────────────────────────────────────────
    genre_map: dict[int, str] = {}
    tracks_csv = FMA_METADATA_DIR / "tracks.csv"
    if tracks_csv.exists():
        meta_df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
        for tid in ids:
            if tid in meta_df.index:
                g = meta_df.loc[tid][("track", "genre_top")]
                genre_map[int(tid)] = str(g) if pd.notna(g) else "Unknown"

    # ── Dimensionality reduction ───────────────────────────────────────────
    print("Fitting UMAP (cosine, 3D) on hybrid-582 ...")
    reducer_umap = umap_lib.UMAP(n_components=3, metric="cosine", random_state=42, verbose=False)
    umap_emb = reducer_umap.fit_transform(hybrid)
    print(f"  Done. shape={umap_emb.shape}\n")

    tsne_emb = None
    if not args.skip_tsne:
        print("Fitting t-SNE (cosine, 3D) on hybrid-582 — may take several minutes ...")
        tsne = TSNE(n_components=3, metric="cosine", random_state=42, n_jobs=-1,
                    perplexity=30, verbose=1)
        tsne_emb = tsne.fit_transform(hybrid)
        print(f"  Done. shape={tsne_emb.shape}\n")

    print("Fitting PCA (3D) on hybrid-582 ...")
    reducer_pca = PCA(n_components=3, random_state=42)
    pca_emb = reducer_pca.fit_transform(hybrid)
    print(f"  Done. shape={pca_emb.shape}\n")

    # ── Compute (silently) ────────────────────────────────────────────────
    print("Computing KNN recall & trustworthiness ...")
    knn_results: dict[str, tuple[float, float]] = {
        "UMAP (3-dim)": projection_report("UMAP (3-dim)", hybrid, umap_emb),
        "PCA (3-dim)":  projection_report("PCA (3-dim)",  hybrid, pca_emb),
    }
    if tsne_emb is not None:
        knn_results["t-SNE (3-dim)"] = projection_report("t-SNE (3-dim)", hybrid, tsne_emb)

    print("Computing hubness ...")
    hub_results: dict[str, tuple[float, int, float]] = {
        "Hybrid (582-dim)": hubness_report("Hybrid (582-dim)", hybrid),
        "UMAP (3-dim)":     hubness_report("UMAP (3-dim)",     umap_emb),
        "PCA (3-dim)":      hubness_report("PCA (3-dim)",      pca_emb),
    }
    if tsne_emb is not None:
        hub_results["t-SNE (3-dim)"] = hubness_report("t-SNE (3-dim)", tsne_emb)

    print("Loading CLAP model ...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    clap_model.load_ckpt()

    metric = TimbreMetric(device=device, sample_rate=TIMBRE_SR)
    print("Running timbremetrics — Hybrid-582 ...")
    hybrid_res = metric(make_hybrid_callable(clap_model))
    print("Running timbremetrics — UMAP-3D ...")
    umap_res = metric(make_umap_callable(clap_model, reducer_umap))
    print("Running timbremetrics — PCA-3D ...")
    pca_res = metric(make_pca_callable(clap_model, reducer_pca))

    covers_results: dict | None = None
    if args.covers80:
        print("Running Covers80 ...")
        covers_results = eval_covers80(args.covers80, clap_model, reducer_umap, reducer_pca)

    if not args.no_plot:
        print("Generating 3D scatter plots ...")
        make_plots(umap_emb, tsne_emb, pca_emb, np.array(ids), genre_map)

    # ── Summary ───────────────────────────────────────────────────────────
    sep = "=" * 74
    print(f"\n\n{sep}")
    print("  RESULTS SUMMARY")
    print(sep)

    print("\n── KNN Recall & Trustworthiness (k=10, vs hybrid-582) ──")
    for name, (recall, tw) in knn_results.items():
        print(f"  {name:<22} KNN-recall={recall:.4f}  trustworthiness={tw:.4f}")

    print("\n── Hubness (k=10) ──")
    for name, (skew, max_k, zero) in hub_results.items():
        print(f"  {name:<22} skew={skew:>6.3f}  max_k={max_k:>4}  zero={zero:>5.1%}")

    print("\n── timbremetrics (Hybrid-582 vs UMAP-3D vs PCA-3D) ──")
    print("  t-SNE excluded: no transform() for new clips")
    print(format_timbre_table(hybrid_res, umap_res, pca_res))

    if covers_results is not None:
        print("\n── Covers80 ──")
        print(f"  {'Method':<22} {'Top-1':>7} {'Top-5':>7} {'MRR':>8}")
        print("  " + "-" * 46)
        for name, (top1, top5, mrr) in covers_results.items():
            print(f"  {name:<22} {top1:>7.1%} {top5:>7.1%} {mrr:>8.4f}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
