#!/usr/bin/env python3
"""CWRU preprocessing: raw vibration -> 12-bit-ADC -> FFT feature vectors.

Pipeline (mirrors the on-device chain in firmware/features.h):
  DE channel only, use only the 12 kHz data.
  1) simulate the ADC: quantize the raw signal to 12-bit signed counts,
     fixed full-scale +/-8 g (ADC_FS), clip to [-2048, 2047].  A real MCU ADC
     hands you integer counts, not floats -- this injects that quantization.
  2) turn each 1024-sample window (hop 512) into a 128-number "frequency
     fingerprint" the model actually trains on:
       - Hann: taper the window's edges to zero (reduces FFT edge artifacts)
       - rFFT: Fast Fourier Transform -> how much vibration energy is present
         at each frequency, like a graphic equalizer. Gives 512 frequency
         bins (bins 1..512, DC/bin-0 dropped).
       - |mag|: keep just the magnitude (strength) of each bin, discard phase.
       - average-pool x4: average every 4 neighboring bins together,
         512 -> 128 numbers. Shrinks the vector to fit the tiny on-device
         model, trading off some frequency resolution.
       - log1p: log-compress each of the 128 numbers, so very loud and very
         quiet frequency content are both represented fairly (like decibels).
  (standardization mean/std is fit LATER, train-only, in train.py.)

Train/test split: per FILE, first 70% of the recording (in time) = train,
last 30% = test, no shuffling. Any window straddling the 70% border is DROPPED
so no train window shares a single raw sample with any test window. (Order
inside each group is shuffled later, in train.py.)

Outputs data/features.npz - the ready-to-train dataset, and the input to
train.py - with, per kept window:
  X [N,128] float32 features, y [N] class, load [N], split [N] (0=train,1=test),
  src [N] file index, start [N] sample offset.
Also two small lookup tables, since features.npz stores everything as plain
numbers (not names) to stay small and fast to load:
  data/classes.json - the 10 class names in order, so y's class number (e.g. 4)
    can be translated back to its name (e.g. "ir_021").
  data/files.json   - the list of source .mat filenames in order, so src's file
    number can be translated back to the actual recording filename.
Usage: python preprocess.py    Apache-2.0."""
import json
import numpy as np
import scipy.io as sio
from pathlib import Path

WIN, HOP, NBINS = 1024, 512, 128
ADC_BITS = 12
ADC_FS = 8.0                       # ADC full-scale = +/- 8 g (fixed, global)
ADC_LEVELS = 1 << (ADC_BITS - 1)   # 2048
ADC_MIN, ADC_MAX = -ADC_LEVELS, ADC_LEVELS - 1   # -2048 .. 2047
TRAIN_FRAC = 0.70
ROOT = Path(__file__).parent
OUT = ROOT / "data"


def de_signal(mat_path):
    md = sio.loadmat(mat_path)
    keys = [k for k in md if k.endswith("_DE_time")]   # drive-end only
    if not keys:
        raise KeyError(f"{mat_path.name}: no *_DE_time key")
    return md[keys[0]].ravel().astype(np.float64)


def adc_quantize(sig):
    """float g -> 12-bit signed ADC counts (fixed +/-8 g full-scale), as float."""
    q = np.round(sig / ADC_FS * ADC_LEVELS)
    q = np.clip(q, ADC_MIN, ADC_MAX)
    return q.astype(np.float32)


def windows_and_split(sig_len):
    """window start offsets + split label (0 train, 1 test, -1 drop straddler)."""
    n = (sig_len - WIN) // HOP + 1
    starts = HOP * np.arange(n)
    ends = starts + WIN
    border = int(TRAIN_FRAC * sig_len)
    split = np.full(n, -1, dtype=np.int8)
    split[ends <= border] = 0        # fully before border -> train
    split[starts >= border] = 1      # fully after border  -> test
    return starts, split


def featurize(counts, starts):
    idx = starts[:, None] + np.arange(WIN)[None, :]
    frames = counts[idx] * np.hanning(WIN).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))[:, 1:513]        # drop DC -> 512
    pooled = mag.reshape(len(starts), NBINS, 4).mean(axis=2)   # 512 -> 128
    return np.log1p(pooled).astype(np.float32)


def parse_name(stem):
    parts = stem.split("_")
    load = int([p for p in parts if p.startswith("load")][0][4:])
    cls = "_".join(parts[:-2])
    return cls, load


def main():
    mats = sorted((OUT / "cwru").glob("*.mat"))
    assert mats, "run download_data.py first"
    classes = sorted({parse_name(m.stem)[0] for m in mats})
    files = [m.name for m in mats]
    Xs, ys, loads, splits, srcs, starts_all = [], [], [], [], [], []
    for fi, m in enumerate(mats):
        cls, load = parse_name(m.stem)
        counts = adc_quantize(de_signal(m))
        starts, split = windows_and_split(len(counts))
        keep = split >= 0
        starts_k, split_k = starts[keep], split[keep]
        feats = featurize(counts, starts_k)
        Xs.append(feats)
        ys.append(np.full(len(starts_k), classes.index(cls), dtype=np.int64))
        loads.append(np.full(len(starts_k), load, dtype=np.int64))
        splits.append(split_k.astype(np.int64))
        srcs.append(np.full(len(starts_k), fi, dtype=np.int64))
        starts_all.append(starts_k.astype(np.int64))
        ntr = int((split_k == 0).sum()); nte = int((split_k == 1).sum())
        ndrop = int((split == -1).sum())
        print(f"{m.name:26s} {cls:7s} load{load}: train {ntr:4d} | test {nte:4d} | dropped {ndrop} straddler(s)")
    X = np.concatenate(Xs); y = np.concatenate(ys); load = np.concatenate(loads)
    split = np.concatenate(splits); src = np.concatenate(srcs); start = np.concatenate(starts_all)
    np.savez_compressed(OUT / "features.npz", X=X, y=y, load=load,
                        split=split, src=src, start=start)
    (OUT / "classes.json").write_text(json.dumps(classes))
    (OUT / "files.json").write_text(json.dumps(files))
    ntr, nte = int((split == 0).sum()), int((split == 1).sum())
    print(f"\nX {X.shape}  |  train {ntr}  test {nte}  ({100*nte/(ntr+nte):.0f}% test)")
    print(f"ADC: {ADC_BITS}-bit signed, full-scale +/-{ADC_FS} g, counts in [{ADC_MIN},{ADC_MAX}]")
    print(f"{len(classes)} classes: {classes}")


if __name__ == "__main__":
    main()
