from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Soundgaze-v2/

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

FMA_SMALL_DIR = RAW_DIR / "fma_small"
FMA_METADATA_DIR = RAW_DIR / "fma_metadata"

# Ensure dirs exist when config is imported
for _d in [RAW_DIR, EMBEDDINGS_DIR, FMA_SMALL_DIR, FMA_METADATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
