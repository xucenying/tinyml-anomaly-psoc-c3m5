#!/usr/bin/env python3
"""Train the CWRU fault MLP on the per-file 70/30 temporal split.

Split comes from preprocess.py (split: 0=train, 1=test). First 70% of every
recording trains, last 30% tests, straddling windows dropped -> no train window
shares a raw sample with any test window. Normalization is fit on TRAIN only.

Architecture: 128 -> 96 -> 48 -> n_classes (softmax), FULLY_CONNECTED + SOFTMAX
only (ops already in the firmware).

Outputs:
  model_fp32.keras     - the trained FP32 model; input to quantize.py.
  data/norm.json        - mean/std (fit on TRAIN only) + window constants.
    Any new data (incl. on-board) must be normalized with these exact
    numbers, or the model's predictions won't make sense.
  data/val_split.npz    - the train/test window indices (ti, vi) actually
    used, so later steps can reload the same split without recomputing it.
  data/fp32_acc.json    - the FP32 test accuracy + which split method was
    used, saved as a reference to compare against later (e.g. after INT8
    quantization).
Also prints model.summary(), per-epoch training progress, and the final
test accuracy on the held-out 30% of each recording.

Usage: python train.py    Apache-2.0."""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).parent
d = np.load(ROOT / "data" / "features.npz")
X, y, split = d["X"], d["y"], d["split"]
classes = json.loads((ROOT / "data" / "classes.json").read_text())

ti = np.where(split == 0)[0]     # train windows
vi = np.where(split == 1)[0]     # test windows

# shuffle WITHIN each group (the temporal 70/30 split already decided membership;
# this only randomizes order inside train and inside test). Fixed seed = reproducible.
rng = np.random.default_rng(0)
rng.shuffle(ti)
rng.shuffle(vi)

print(f"train windows: {len(ti)} | test windows: {len(vi)} "
      f"({100*len(vi)/(len(ti)+len(vi)):.0f}% test)")

# normalization fit on TRAIN only (no leakage)
mean = float(X[ti].mean())
std  = float(X[ti].std())
Xn = (X - mean) / std

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128,)),
    tf.keras.layers.Dense(96, activation="relu"),
    tf.keras.layers.Dense(48, activation="relu"),
    tf.keras.layers.Dense(len(classes), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
model.fit(Xn[ti], y[ti], validation_data=(Xn[vi], y[vi]),
          epochs=30, batch_size=128, verbose=2)

loss, acc = model.evaluate(Xn[vi], y[vi], verbose=0)
print(f"\nTest (last-30%-of-each-file) accuracy: {acc*100:.2f}%  [12-bit ADC, no overlap]")

model.save(ROOT / "model_fp32.keras")
np.savez(ROOT / "data" / "val_split.npz", vi=vi, ti=ti)
(ROOT / "data" / "norm.json").write_text(json.dumps(
    {"mean": mean, "std": std, "win": 1024, "hop": 512, "nbins": 128}))
(ROOT / "data" / "fp32_acc.json").write_text(json.dumps(
    {"fp32_acc": float(acc), "split": "per-file-70-30-temporal"}))
print("saved model_fp32.keras, norm.json (train-only stats)")
