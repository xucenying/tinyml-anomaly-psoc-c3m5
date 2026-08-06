#!/usr/bin/env python3
"""Full-integer INT8 quantization + C-array export.

Shrinks the trained FP32 model down to INT8 - the small, fast format that
actually runs on the chip - and proves the shrink didn't cost accuracy.

Steps:
  1) load model_fp32.keras + test data, apply the same train-only
     normalization used in train.py.
  2) convert to two .tflite versions: FP32 (kept as a baseline) and full
     INT8. The INT8 conversion needs a "representative dataset" (300 sample
     training windows) so the converter can pick the right scale for
     squeezing each layer's numbers into 8-bit integers (requantization).
  3) evaluate both versions on the held-out test windows, so you can see
     whether shrinking to INT8 actually hurt accuracy.
  4) print + save (data/quant_report.json) a comparison table: file size and
     accuracy for FP32 vs INT8, the size ratio, and the accuracy delta.
  5) export both .tflite models as C header files the firmware can compile
     in directly, since the microcontroller can't read .tflite files itself:
     ../firmware/model_data.h (INT8) and model_data_fp32.h (FP32).

Usage: python quantize.py    Apache-2.0."""
import json, subprocess, sys
import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).parent
d = np.load(ROOT / "data" / "features.npz")
X, y = d["X"], d["y"]
# apply the SAME train-only normalization used in train.py (no leakage)
norm = json.loads((ROOT / "data" / "norm.json").read_text())
X = (X - norm["mean"]) / norm["std"]
sp = np.load(ROOT / "data" / "val_split.npz")
vi, ti = sp["vi"], sp["ti"]
model = tf.keras.models.load_model(ROOT / "model_fp32.keras")

def rep_data():
    for i in np.random.default_rng(0).choice(ti, 300, replace=False):
        yield [X[i:i+1].astype(np.float32)]

# --- FP32 tflite (Phase-2 baseline) ---
conv = tf.lite.TFLiteConverter.from_keras_model(model)
fp32 = conv.convert()
(ROOT / "model_fp32.tflite").write_bytes(fp32)

# --- full-INT8 tflite ---
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
int8 = conv.convert()
(ROOT / "model_int8.tflite").write_bytes(int8)

def tflite_acc(blob, quant):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    correct = 0
    for i in vi:
        x = X[i:i+1].astype(np.float32)
        if quant:
            s, zp = inp["quantization"]
            x = np.clip(np.round(x / s + zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        it.invoke()
        correct += int(np.argmax(it.get_tensor(out["index"])) == y[i])
    return correct / len(vi)

a32, a8 = tflite_acc(fp32, False), tflite_acc(int8, True)
print(f"\n| model | size (B) | val acc |")
print(f"|---|---|---|")
print(f"| FP32 | {len(fp32)} | {a32*100:.2f}% |")
print(f"| INT8 | {len(int8)} | {a8*100:.2f}% |")
print(f"accuracy delta: {(a32-a8)*100:+.2f} pp; size ratio: {len(fp32)/len(int8):.1f}x")
(ROOT / "data" / "quant_report.json").write_text(json.dumps(
    {"fp32_bytes": len(fp32), "int8_bytes": len(int8),
     "fp32_acc": a32, "int8_acc": a8}))

# --- export C arrays for firmware (INT8 for rungs 2-5, FP32 for rung 1) ---
conv_script = ROOT.parent / "firmware" / "convert_tflite_to_c.py"
subprocess.run([sys.executable, str(conv_script), str(ROOT / "model_int8.tflite"),
                str(ROOT.parent / "firmware" / "model_data.h")], check=True)
subprocess.run([sys.executable, str(conv_script), str(ROOT / "model_fp32.tflite"),
                str(ROOT.parent / "firmware" / "model_data_fp32.h")], check=True)
print("wrote ../firmware/model_data.h (INT8) and model_data_fp32.h (FP32)")
