#!/usr/bin/env python3
"""CWRU preprocessing: raw vibration -> FFT feature vectors an MCU can replicate.

Per window (mirrors planned on-device CMSIS-DSP chain):
  1024-sample window (hop 512) -> hann -> rfft -> |mag| bins 1..512
  -> average-pool x4 -> 128 log1p features.

Each window is tagged with its class AND its motor load (group), so training
can hold out a whole load with no leakage. Normalization (mean/std) is computed
LATER, in train.py, from training windows only — computing it here over all
data would leak test statistics.

Outputs: data/features.npz (X [N,128] float32, y [N], load [N]), data/classes.json.
Usage: python preprocess.py    Apache-2.0."""
import json
import numpy as np
import scipy.io as sio
from pathlib import Path

WIN, HOP, NBINS = 1024, 512, 128
ROOT = Path(__file__).parent
OUT = ROOT / "data"

def de_signal(mat_path):
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

def parse_name(stem):
    # e.g. "ir_007_load1_106" -> class "ir_007", load 1
    parts = stem.split("_")
    load = int([p for p in parts if p.startswith("load")][0][4:])
    cls = "_".join(parts[:-2])  # drop loadN and filenum
    return cls, load

def main():
    mats = sorted((OUT / "cwru").glob("*.mat"))
    assert mats, "run download_data.py first"
    classes = sorted({parse_name(m.stem)[0] for m in mats})
    Xs, ys, loads = [], [], []
    for m in mats:
        cls, load = parse_name(m.stem)
        f = featurize(de_signal(m))
        Xs.append(f)
        ys.append(np.full(len(f), classes.index(cls)))
        loads.append(np.full(len(f), load))
        print(f"{m.name}: {len(f)} windows -> class '{cls}', load {load}")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    load = np.concatenate(loads)
    np.savez_compressed(OUT / "features.npz", X=X, y=y, load=load)
    (OUT / "classes.json").write_text(json.dumps(classes))
    print(f"\nX {X.shape}, y {y.shape}, {len(classes)} classes, loads {sorted(set(load))}")

if __name__ == "__main__":
    main()
