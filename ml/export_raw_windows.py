#!/usr/bin/env python3
"""export_raw_windows.py — export raw TEST windows as 12-bit ADC counts for the
live replay demo.

Uses the per-file test split (last 30% of each recording, no overlap with train)
so the demo streams data the model never trained on. Each sample is a 12-bit
signed ADC count (fixed +/-8g full-scale) -- exactly what the board's ADC would
produce and what the on-board FFT path consumes.

Output: data/raw_windows.npz (X_raw [N,1024] float counts, y [N], load [N]).
Run after preprocess.py. Usage: python export_raw_windows.py    Apache-2.0."""
import numpy as np
from pathlib import Path
from preprocess import de_signal, adc_quantize, WIN

ROOT = Path(__file__).parent
OUT = ROOT / "data"


def main():
    import json
    d = np.load(OUT / "features.npz")
    y, load, split, src, start = d["y"], d["load"], d["split"], d["src"], d["start"]
    files = json.loads((OUT / "files.json").read_text())
    test = np.where(split == 1)[0]

    sig_cache = {}
    X = np.empty((len(test), WIN), dtype=np.float32)
    for row, idx in enumerate(test):
        fi, s = int(src[idx]), int(start[idx])
        if fi not in sig_cache:
            sig_cache[fi] = adc_quantize(de_signal(OUT / "cwru" / files[fi]))
        X[row] = sig_cache[fi][s:s + WIN]
    np.savez_compressed(OUT / "raw_windows.npz",
                        X_raw=X, y=y[test], load=load[test])
    print(f"raw_windows.npz: X_raw {X.shape} (12-bit ADC counts), test windows only")


if __name__ == "__main__":
    main()
