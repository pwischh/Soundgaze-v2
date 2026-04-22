from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent   # Soundgaze-v2/
DATA_DIR      = ROOT / "data"
EMB_DIR       = DATA_DIR / "embeddings"
REDUCED_DIR   = DATA_DIR / "reduced"
FMA_SMALL_DIR = DATA_DIR / "raw" / "fma_small"
META_DIR      = DATA_DIR / "raw" / "fma_metadata"

CLAP_PATH     = EMB_DIR / "fma_small_clap.parquet"
ACOUSTIC_PATH = EMB_DIR / "fma_small_acoustic_v2.parquet"
SCALER_PATH   = EMB_DIR / "acoustic_scaler_v2.npz"

CLAP_SR    = 48000
LIBROSA_SR = 22050
DURATION   = 30.0
CLAP_LEN   = int(CLAP_SR * DURATION)
CLAP_BATCH = 16

METHODS = ["umap", "tsne", "pca"]
