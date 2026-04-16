"""
Generate CNN10 embeddings for FMA Small and save to parquet.
Output: data/embeddings/fma/fma_small_embeddings.parquet
Columns: track_id (int), embedding (list[float], 2048-dim)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import torch
from panns_inference import AudioTagging
from pathlib import Path

from config import FMA_SMALL_DIR, EMBEDDINGS_DIR

SAMPLE_RATE = 32000  # CNN10 expects 32kHz
OUTPUT_PATH = EMBEDDINGS_DIR / "fma_small_embeddings.parquet"


def load_audio(path: Path) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=30.0)
        return audio.reshape(1, -1)  # (1, samples)
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return None


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CNN10 model ...")
    model = AudioTagging(checkpoint_path=None, device=device)

    mp3_files = sorted(FMA_SMALL_DIR.rglob("*.mp3"))
    print(f"Found {len(mp3_files)} tracks.\n")

    records = []
    for i, path in enumerate(mp3_files):
        track_id = int(path.stem)
        audio = load_audio(path)
        if audio is None:
            continue

        _, embedding = model.inference(audio)  # embedding: (1, 2048)
        records.append({"track_id": track_id, "embedding": embedding[0].tolist()})

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(mp3_files)} tracks processed ...")

    print(f"\nSaving {len(records)} embeddings -> {OUTPUT_PATH}")
    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
