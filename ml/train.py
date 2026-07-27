#!/usr/bin/env python3
"""Train the CWRU fault MLP with a leakage-free leave-one-load-out split.

Train on motor loads {0,1,2}, test on held-out load {3}. Train and test share
no recording and no operating condition, so validation accuracy reflects real
generalization. Normalization is fit on TRAIN windows only.

Architecture: 128 -> 96 -> 48 -> n_classes (softmax), FULLY_CONNECTED + SOFTMAX
only (ops already in the firmware). Usage: python train.py    Apache-2.0."""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

HELD_OUT_LOAD = 3   # train on the other loads, test on this one

ROOT = Path(__file__).parent
d = np.load(ROOT / "data" / "features.npz")
X, y, load = d["X"], d["y"], d["load"]
classes = json.loads((ROOT / "data" / "classes.json").read_text())

ti = np.where(load != HELD_OUT_LOAD)[0]     # train indices
vi = np.where(load == HELD_OUT_LOAD)[0]     # held-out load = validation
print(f"train windows: {len(ti)} (loads {sorted(set(load[ti]))}) | "
      f"val windows: {len(vi)} (held-out load {HELD_OUT_LOAD})")

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
print(f"\nHeld-out-load (load {HELD_OUT_LOAD}) accuracy: {acc*100:.2f}%  "
      f"[leakage-free]")

model.save(ROOT / "model_fp32.keras")
np.savez(ROOT / "data" / "val_split.npz", vi=vi, ti=ti)
(ROOT / "data" / "norm.json").write_text(json.dumps(
    {"mean": mean, "std": std, "win": 1024, "hop": 512, "nbins": 128}))
(ROOT / "data" / "fp32_acc.json").write_text(json.dumps(
    {"fp32_acc": float(acc), "held_out_load": HELD_OUT_LOAD}))
print("saved model_fp32.keras, norm.json (train-only stats)")
