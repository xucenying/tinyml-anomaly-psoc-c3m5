#!/usr/bin/env python3
"""Train a tiny MLP fault classifier on CWRU FFT features.

Architecture: 128 -> 96 -> 48 -> n_classes (softmax). FULLY_CONNECTED +
SOFTMAX only — ops already registered in the firmware, ideal for the
CMSIS-NN comparison. ~18k params ≈ 18 KB INT8.

NOTE (honest caveat, also for the README): train/val windows come from the
same recordings, so accuracy is optimistic vs. unseen operating conditions.
Good enough for the optimization story; a load-condition split is the
stretch upgrade. Usage: python train.py    Apache-2.0."""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).parent
d = np.load(ROOT / "data" / "features.npz")
X, y = d["X"], d["y"]
classes = json.loads((ROOT / "data" / "classes.json").read_text())

rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
n_val = len(X) // 5
vi, ti = idx[:n_val], idx[n_val:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128,)),
    tf.keras.layers.Dense(96, activation="relu"),
    tf.keras.layers.Dense(48, activation="relu"),
    tf.keras.layers.Dense(len(classes), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
model.fit(X[ti], y[ti], validation_data=(X[vi], y[vi]),
          epochs=30, batch_size=128, verbose=2)

loss, acc = model.evaluate(X[vi], y[vi], verbose=0)
print(f"\nFP32 validation accuracy: {acc*100:.2f}%")
model.save(ROOT / "model_fp32.keras")
np.savez(ROOT / "data" / "val_split.npz", vi=vi, ti=ti)
(ROOT / "data" / "fp32_acc.json").write_text(json.dumps({"fp32_acc": float(acc)}))
print("saved model_fp32.keras")
