#!/usr/bin/env python3
"""export_raw_windows.py — export RAW 1024-sample vibration windows for the
held-out load (load 3), for the live replay demo.

These are the raw samples a sensor+ADC would produce (BEFORE any Hann window or
FFT). The demo streams them to the board, which then runs the full on-chip
pipeline (Hann + FFT features + INT8 + classify). Using only the held-out load
means the demo runs on data the model never trained on.

Output: data/raw_windows.npz (X_raw [N,1024] float32, y [N], load [N]).
Usage: python export_raw_windows.py    Apache-2.0."""
import numpy as np
import scipy.io as sio
from pathlib import Path

WIN, HOP = 1024, 512
HELD_OUT_LOAD = 3          # must match train.py
ROOT = Path(__file__).parent
OUT = ROOT / "data"


def de_signal(mat_path):
    md = sio.loadmat(mat_path)
    keys = [k for k in md if k.endswith("_DE_time")]
    if not keys:
        raise KeyError(f"{mat_path.name}: no *_DE_time key")
    return md[keys[0]].ravel().astype(np.float32)


def parse_name(stem):
    parts = stem.split("_")
    load = int([p for p in parts if p.startswith("load")][0][4:])
    cls = "_".join(parts[:-2])
    return cls, load


def raw_windows(sig):
    n = (len(sig) - WIN) // HOP + 1
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return sig[idx].astype(np.float32)        # raw, pre-Hann


def main():
    import json
    classes = json.loads((OUT / "classes.json").read_text())
    mats = sorted((OUT / "cwru").glob("*.mat"))
    assert mats, "run download_data.py first"
    Xs, ys, loads = [], [], []
    for m in mats:
        cls, load = parse_name(m.stem)
        if load != HELD_OUT_LOAD:
            continue
        w = raw_windows(de_signal(m))
        Xs.append(w)
        ys.append(np.full(len(w), classes.index(cls)))
        loads.append(np.full(len(w), load))
        print(f"{m.name}: {len(w)} raw windows -> '{cls}', load {load}")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    load = np.concatenate(loads)
    np.savez_compressed(OUT / "raw_windows.npz", X_raw=X, y=y, load=load)
    print(f"\nraw_windows.npz: X_raw {X.shape}, y {y.shape}, held-out load {HELD_OUT_LOAD}")


if __name__ == "__main__":
    main()
