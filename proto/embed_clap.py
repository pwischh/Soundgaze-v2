"""
Generate CLAP embeddings for FMA Small and save to parquet.
Output: data/embeddings/fma_small_clap.parquet
Columns: track_id (int), embedding (list[float], 512-dim)

Usage:
  python embed_clap.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import laion_clap
from pathlib import Path

from config import FMA_SMALL_DIR, EMBEDDINGS_DIR

CLAP_SR     = 48000
DURATION    = 30.0
CLAP_LEN    = int(CLAP_SR * DURATION)  # 1_440_000
BATCH_SIZE  = 16
OUTPUT_PATH = EMBEDDINGS_DIR / "fma_small_clap.parquet"


def fix_length(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.pad(audio, (0, target - len(audio)))


def load_audio(path: Path) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(path, sr=CLAP_SR, mono=True, duration=DURATION)
        return fix_length(audio, CLAP_LEN)
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return None


def main() -> None:
    print("Loading CLAP model ...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt()

    mp3_files = sorted(FMA_SMALL_DIR.rglob("*.mp3"))
    print(f"Found {len(mp3_files)} tracks.\n")

    records = []
    batch_audios: list[np.ndarray] = []
    batch_ids: list[int] = []

    def flush_batch() -> None:
        if not batch_audios:
            return
        x = np.stack(batch_audios)  # (B, CLAP_LEN)
        embs = model.get_audio_embedding_from_data(x=x, use_tensor=False)  # (B, 512)
        for tid, emb in zip(batch_ids, embs):
            records.append({"track_id": tid, "embedding": emb.tolist()})
        batch_audios.clear()
        batch_ids.clear()

    for i, path in enumerate(mp3_files):
        audio = load_audio(path)
        if audio is None:
            continue

        batch_audios.append(audio)
        batch_ids.append(int(path.stem))

        if len(batch_audios) >= BATCH_SIZE:
            flush_batch()

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(mp3_files)} tracks processed ...")

    flush_batch()

    print(f"\nSaving {len(records)} embeddings -> {OUTPUT_PATH}")
    pd.DataFrame(records).to_parquet(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
