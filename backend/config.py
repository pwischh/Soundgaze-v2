from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent   # Soundgaze-v2/
DATA_DIR      = ROOT / "data"
EMB_DIR       = DATA_DIR / "embeddings"
FMA_SMALL_DIR = DATA_DIR / "raw" / "fma_small"
META_DIR      = DATA_DIR / "raw" / "fma_metadata"

CLAP_SR    = 48000
LIBROSA_SR = 22050
DURATION   = 30.0
CLAP_LEN   = int(CLAP_SR * DURATION)
CLAP_BATCH = 16

METHODS = ["umap", "tsne", "pca"]

# Set to True to cap the corpus at DEV_LIMIT tracks for fast frontend testing.
# Dev mode writes to separate _dev files so the full corpus files are never touched.
DEV_MODE  = True
DEV_LIMIT = 200

_suffix       = "_dev" if DEV_MODE else ""
CLAP_PATH     = EMB_DIR / "fma_small_clap.parquet"                        # always full
ACOUSTIC_PATH = EMB_DIR / f"fma_small_acoustic_v2{_suffix}.parquet"
SCALER_PATH   = EMB_DIR / f"acoustic_scaler_v2{_suffix}.npz"
REDUCED_DIR   = DATA_DIR / f"reduced{_suffix}"
