"""
Benchmark PANN and CLAP against human timbre similarity ratings
using the timbremetrics library (21 psychoacoustic datasets, 334 clips).

Reports Spearman rho, Kendall tau, and Triplet Agreement for both models.

Usage:
  python eval_timbremetrics.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import librosa
from panns_inference import AudioTagging
import laion_clap
from timbremetrics import TimbreMetric

TIMBRE_SR    = 44100
PANN_SR      = 32000
CLAP_SR      = 48000
PANN_MIN_LEN = PANN_SR * 2  # CNN14 needs ≥2s to avoid pooling collapse


def pann_callable(model: AudioTagging):
    def fn(x: torch.Tensor) -> torch.Tensor:
        audio = librosa.resample(
            x.squeeze(0).numpy(), orig_sr=TIMBRE_SR, target_sr=PANN_SR
        )
        if len(audio) < PANN_MIN_LEN:
            audio = np.pad(audio, (0, PANN_MIN_LEN - len(audio)))
        _, emb = model.inference(audio.reshape(1, -1))
        return torch.from_numpy(emb[0].copy())
    return fn


def clap_callable(model):
    def fn(x: torch.Tensor) -> torch.Tensor:
        audio = librosa.resample(
            x.squeeze(0).numpy(), orig_sr=TIMBRE_SR, target_sr=CLAP_SR
        )
        emb = model.get_audio_embedding_from_data(
            x=audio.reshape(1, -1).astype(np.float32), use_tensor=False
        )
        return torch.from_numpy(emb[0].copy())
    return fn


def print_results(pann_res: dict, clap_res: dict) -> None:
    print(f"\n{'Metric':<40} {'PANN':>8} {'CLAP':>8}")
    print("-" * 58)
    for distance in ["cosine", "l2"]:
        p_sub = pann_res.get(distance, {})
        c_sub = clap_res.get(distance, {})
        for metric in sorted(p_sub.keys()):
            p = p_sub.get(metric, float("nan"))
            c = c_sub.get(metric, float("nan"))
            label = f"{distance}/{metric}"
            print(f"{label:<40} {float(p):>8.4f} {float(c):>8.4f}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading PANN CNN10 ...")
    pann_model = AudioTagging(checkpoint_path=None, device=device)

    print("Loading CLAP HTSAT-tiny ...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    clap_model.load_ckpt()

    metric = TimbreMetric(device=device, sample_rate=TIMBRE_SR)

    print("\nRunning PANN ...")
    pann_res = metric(pann_callable(pann_model))

    print("Running CLAP ...")
    clap_res = metric(clap_callable(clap_model))

    print_results(pann_res, clap_res)


if __name__ == "__main__":
    main()
