import os
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent   # Soundgaze-v2/
DATA_DIR      = ROOT / "data"
EMB_DIR       = DATA_DIR / "embeddings"
FMA_SMALL_DIR = DATA_DIR / "raw" / "fma_small"
META_DIR      = DATA_DIR / "raw" / "fma_metadata"
SLIM_META_PATH = DATA_DIR / "fma_slim_metadata.json"

CLAP_SR    = 48000
LIBROSA_SR = 22050
DURATION   = 30.0
CLAP_LEN   = int(CLAP_SR * DURATION)
CLAP_BATCH = 16

METHODS = ["umap", "tsne", "pca"]

# DEV_MODE can be overridden by the DEV_MODE environment variable.
# Set DEV_MODE=true in Railway variables to use the committed 200-track dev artifacts.
DEV_MODE  = os.getenv("DEV_MODE", "false").lower() == "true"
DEV_LIMIT = 200

_suffix       = "_dev" if DEV_MODE else ""
CLAP_PATH     = EMB_DIR / f"fma_small_clap{_suffix}.parquet"
ACOUSTIC_PATH = EMB_DIR / f"fma_small_acoustic_v2{_suffix}.parquet"
SCALER_PATH   = EMB_DIR / f"acoustic_scaler_v2{_suffix}.npz"
REDUCED_DIR   = DATA_DIR / f"reduced{_suffix}"
