#!/usr/bin/env python3
"""CWRU preprocessing: raw vibration -> FFT feature vectors an MCU can replicate.

Pipeline per window (mirrors planned on-device CMSIS-DSP chain):
  1024-sample window (hop 512) -> hann -> rfft -> |mag| of bins 1..512
  -> average-pool x4 -> 128 log1p features -> global standardize.

Outputs: data/features.npz (X float32 [N,128], y int64 [N]), data/norm.json
(mean/std for firmware), data/classes.json.
Usage: python preprocess.py    Apache-2.0."""
import json
import numpy as np
import scipy.io as sio
from pathlib import Path

WIN, HOP, NBINS = 1024, 512, 128
ROOT = Path(__file__).parent
OUT = ROOT / "data"

def de_signal(mat_path):
    """Extract drive-end accelerometer time series from a CWRU .mat file."""
    md = sio.loadmat(mat_path)
    keys = [k for k in md if k.endswith("_DE_time")]
    if not keys:
        raise KeyError(f"{mat_path.name}: no *_DE_time key (keys: {list(md)[:8]})")
    return md[keys[0]].ravel().astype(np.float32)

def featurize(sig):
    n = (len(sig) - WIN) // HOP + 1
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = sig[idx] * np.hanning(WIN).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))[:, 1:513]        # drop DC -> 512
    pooled = mag.reshape(n, NBINS, 4).mean(axis=2)              # 512 -> 128
    return np.log1p(pooled).astype(np.float32)

def main():
    mats = sorted((OUT / "cwru").glob("*.mat"))
    assert mats, "run download_data.py first"
    classes = sorted({m.stem.rsplit("_", 1)[0] for m in mats})
    Xs, ys = [], []
    for m in mats:
        cls = m.stem.rsplit("_", 1)[0]
        f = featurize(de_signal(m))
        Xs.append(f)
        ys.append(np.full(len(f), classes.index(cls)))
        print(f"{m.name}: {len(f)} windows -> class '{cls}'")
    X, y = np.concatenate(Xs), np.concatenate(ys)
    mean, std = X.mean(), X.std()               # global (scalar) norm: trivial on MCU
    X = (X - mean) / std
    np.savez_compressed(OUT / "features.npz", X=X, y=y)
    (OUT / "norm.json").write_text(json.dumps({"mean": float(mean), "std": float(std),
                                               "win": WIN, "hop": HOP, "nbins": NBINS}))
    (OUT / "classes.json").write_text(json.dumps(classes))
    print(f"\nX {X.shape}, y {y.shape}, {len(classes)} classes: {classes}")
    print(f"norm: mean={mean:.4f} std={std:.4f}  (saved for firmware)")

if __name__ == "__main__":
    main()
