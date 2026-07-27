#!/usr/bin/env python3
"""In-process end-to-end check of the raw-window demo: protocol roundtrip +
board FFT + classify + debounced alert. No sockets. Apache-2.0."""
import json, sys
from pathlib import Path
import numpy as np
from protocol import pack_frame, read_frame, PAYLOAD_LEN, RAW_DIM
from board_sim import Board

ROOT = Path(__file__).resolve().parent
labels = json.loads((ROOT.parent / "ml/data/classes.json").read_text())
d = np.load(ROOT.parent / "ml/data/raw_windows.npz")
X, y = d["X_raw"].astype(np.float32), d["y"]
by_class = {lab: X[y == i] for i, lab in enumerate(labels)}

rng = np.random.default_rng(1)
def take(lab, n):
    pool = by_class[lab]; idx = rng.integers(0, len(pool), n)
    return [(lab, pool[i]) for i in idx]
plan = take("normal", 6) + take("or_021", 14) + take("normal", 6)
fault_onset = 6

# protocol roundtrip (bytes in == bytes out, CRC valid)
def reader_from(buf):
    pos = [0]
    def rd(n):
        b = buf[pos[0]:pos[0]+n]; pos[0] += n; return b
    return rd
w0 = plan[0][1]
frame = pack_frame(w0)
assert len(frame) == 2 + 2 + PAYLOAD_LEN + 2, "frame size"
win_back, ok = read_frame(reader_from(frame))
assert ok, "CRC failed on roundtrip"
assert np.allclose(win_back, w0, atol=0), "payload mismatch"
print(f"protocol roundtrip OK  (frame {len(frame)} B, {RAW_DIM} raw samples, CRC valid)")

board = Board(ROOT.parent / "ml/model_int8.tflite")
correct = alert_at = false_alerts = 0
alert_at = None
print(f"{'seq':>3} {'true':>8} {'pred':>8} {'conf':>5} {'fft':>7} {'inf':>7}  state")
for i, (true_lab, window) in enumerate(plan):
    # go through the wire format exactly as the host+board would
    _, ok = read_frame(reader_from(pack_frame(window)))
    assert ok
    line = board.step(window).split()
    pred, conf, fft_c, inf_c, state = line[3], line[4], int(line[5]), int(line[6]), line[7]
    correct += (pred == true_lab)
    if state == "ALERT":
        if alert_at is None and i >= fault_onset: alert_at = i
        if i < fault_onset: false_alerts += 1
    mark = "  <== fault injected" if i == fault_onset else ""
    print(f"{i:>3} {true_lab:>8} {pred:>8} {conf:>5} {fft_c:>7} {inf_c:>7}  {state}{mark}")

n = len(plan)
print(f"\nwindows: {n} | pred-accuracy {correct}/{n} = {100*correct/n:.0f}%")
if alert_at is not None:
    print(f"fault at frame {fault_onset} (or_021) -> ALERT at {alert_at} "
          f"= {alert_at-fault_onset+1} frames latency")
else:
    print("NO ALERT raised")
print(f"false alerts during healthy stretch: {false_alerts}")
print("RESULT:", "PASS" if (alert_at is not None and false_alerts == 0) else "CHECK")
