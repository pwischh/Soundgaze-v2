"""
Extract handcrafted audio features for FMA Small and save to parquet.
Output: data/embeddings/fma_small_handcrafted.parquet
Columns: track_id (int), features (list[float], 10-dim)

Feature vector layout (10 dims):
  [0]     tempo (normalized 0-1, clipped at 240 BPM)
  [1]     spectral centroid mean (Hz, normalized 0-1, clipped at 8000 Hz)
  [2-8]   spectral contrast mean over 7 bands
  [9]     RMS energy mean (log-scaled, normalized 0-1)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
from pathlib import Path

from config import FMA_SMALL_DIR, EMBEDDINGS_DIR

SAMPLE_RATE = 22050
OUTPUT_PATH = EMBEDDINGS_DIR / "fma_small_handcrafted.parquet"

TEMPO_MAX = 240.0
CENTROID_MAX = 8000.0
RMS_LOG_MIN = -6.0
RMS_LOG_MAX = 0.0


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_norm = float(np.clip(tempo / TEMPO_MAX, 0.0, 1.0))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_norm = float(np.clip(centroid.mean() / CENTROID_MAX, 0.0, 1.0))

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    contrast_mean = contrast.mean(axis=1)  # 7 values

    rms = librosa.feature.rms(y=audio)
    rms_log = np.log10(np.maximum(rms.mean(), 1e-6))
    rms_norm = float(np.clip((rms_log - RMS_LOG_MIN) / (RMS_LOG_MAX - RMS_LOG_MIN), 0.0, 1.0))

    return np.concatenate([[tempo_norm, centroid_norm], contrast_mean, [rms_norm]])


def main() -> None:
    mp3_files = sorted(FMA_SMALL_DIR.rglob("*.mp3"))
    print(f"Found {len(mp3_files)} tracks.\n")

    records = []
    for i, path in enumerate(mp3_files):
        track_id = int(path.stem)
        try:
            audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=30.0)
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue

        features = extract_features(audio, sr)
        records.append({"track_id": track_id, "features": features.tolist()})

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(mp3_files)} tracks processed ...")

    print(f"\nSaving {len(records)} feature vectors -> {OUTPUT_PATH}")
    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
